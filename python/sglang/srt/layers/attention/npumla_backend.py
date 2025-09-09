from __future__ import annotations

"""
Support attention backend for NpuMLA.

#TODO
Enable speculative sampling in NpuMLA
"""

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu

from sglang.global_config import global_config
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.dp_attention import (
    DPPaddingMode,
    get_attention_dp_rank,
    get_attention_dp_size,
    get_attention_tp_size,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
    from sglang.srt.speculative.spec_info import SpecInfo


MAX_SEQ_LEN = 4096


@dataclass
class NpuMLAMetadata:
    block_kv_indices: Optional[torch.Tensor] = None
    mc2_mask: Optional[torch.Tensor] = None
    q_lens_list: Optional[list[int]] = None
    kv_lens_list: Optional[list[int]] = None

    def __init__(
        self,
        block_kv_indices: Optional[torch.Tensor] = None,
        seq_lens_list=None,
        forward_batch: ForwardBatch = None,
    ):
        self.block_kv_indices = block_kv_indices
        self.q_lens_list = seq_lens_list or [1]

        if (
            forward_batch.forward_mode == ForwardMode.EXTEND
            or forward_batch.global_num_tokens_cpu is None
        ):
            tp_size = get_attention_tp_size()
            self.kv_lens_list = np.cumsum(self.q_lens_list).tolist()
            if forward_batch.dp_padding_mode == DPPaddingMode.MAX_LEN:
                # DPPaddingMode.MAX_LEN
                self.kv_lens_list[-1] = forward_batch.input_ids.size(0)
            else:
                # DPPaddingMode.SUM_LEN
                self.kv_lens_list[-1] = (
                    (self.kv_lens_list[-1] - 1) // tp_size + 1
                ) * tp_size
        elif forward_batch.forward_mode.is_target_verify():
            draft_token_num = forward_batch.spec_info.draft_token_num
            kv_lens = np.array(seq_lens_list) + draft_token_num
            self.kv_lens_list = kv_lens.tolist()
            valid_bs = len(self.q_lens_list)
            self.q_lens_list = list(
                range(
                    draft_token_num, draft_token_num * (valid_bs + 1), draft_token_num
                )
            )
            if forward_batch.dp_padding_mode is not None:
                fake_tokens = (
                    forward_batch.global_num_tokens_cpu[0] - self.q_lens_list[-1]
                )
                if fake_tokens > 0:
                    self.q_lens_list[-1] = forward_batch.global_num_tokens_cpu[0]
                    # self.q_lens_list.append(forward_batch.global_num_tokens_cpu[0])
                    # self.kv_lens_list.append(0)
            self.block_kv_indices = self.block_kv_indices[: len(self.q_lens_list)]
        elif forward_batch.forward_mode.is_draft_extend():
            tp_size = get_attention_tp_size()
            self.kv_lens_list = copy.copy(self.q_lens_list)
            self.q_lens_list = np.cumsum(forward_batch.extend_seq_lens_cpu).tolist()
            align_global_num_tokens = (
                (forward_batch.extend_num_tokens + tp_size - 1) // tp_size * tp_size
            )
            fake_tokens = align_global_num_tokens - self.q_lens_list[-1]
            if fake_tokens > 0:
                # todo (zyj) TND QS of each batch computed by actual_seq_lengths should be in range [0, 16]
                self.q_lens_list.append(align_global_num_tokens)
                self.kv_lens_list.append(0)
            self.block_kv_indices = self.block_kv_indices[: len(self.q_lens_list)]
        elif forward_batch.global_num_tokens_for_logprob_cpu is not None:
            self.kv_lens_list = copy.copy(self.q_lens_list)
            self.q_lens_list = list(range(1, len(self.q_lens_list) + 1))
            vaild_batch = forward_batch.global_num_tokens_for_logprob_cpu[
                get_attention_dp_rank()
            ]
            assert vaild_batch <= forward_batch.batch_size
            self.mc2_mask = torch.zeros(
                forward_batch.batch_size,
                dtype=torch.bool,
                device=block_kv_indices.device,
            )
            self.mc2_mask[:vaild_batch].fill_(True)


def create_npumla_kv_indices(
    bs,
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride,
    kv_indices_ptr_stride,
    page_size,
):
    seq_lens_max = page_kernel_lens_ptr.max()
    block_table = (
        req_to_token_ptr[req_pool_indices_ptr, :seq_lens_max][:, ::page_size]
        // page_size
    )
    kv_indices_ptr[: block_table.shape[0], : block_table.shape[1]].copy_(block_table)


class NpuMLABackend(TorchNativeAttnBackend):
    """npumla attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
    ):
        super().__init__(model_runner)
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.skip_prefill = skip_prefill
        self.forward_metadata: Optional[NpuMLAMetadata] = None
        self.page_size = model_runner.page_size
        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        if "deepseek" in model_runner.model_config.hf_config.architectures[0].lower():
            self.kv_lora_rank = model_runner.model_config.kv_lora_rank
            self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
            self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
            self.v_head_dim = model_runner.model_config.v_head_dim
            self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
            self.scaling = model_runner.model_config.scaling

        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

        self.mask_length = 2048
        self.attn_mask = ~torch.tril(
            torch.ones(
                (self.mask_length, self.mask_length),
                dtype=torch.bool,
                device=model_runner.device,
            )
        )

        max_bs = model_runner.req_to_token_pool.size
        self.kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

        self.q_indptr_decode = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=model_runner.device
        )
        max_total_tokens = model_runner.server_args.max_total_tokens or MAX_SEQ_LEN
        self.max_seqlen_pad = max_total_tokens // model_runner.server_args.page_size

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            bs = forward_batch.input_ids.size(0)
            block_kv_indices = torch.full(
                (bs, self.max_seqlen_pad),
                -1,
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )
            create_npumla_kv_indices(
                forward_batch.batch_size,
                self.req_to_token,
                forward_batch.req_pool_indices,
                (
                    forward_batch.seq_lens + self.num_draft_tokens
                    if forward_batch.forward_mode.is_target_verify()
                    else forward_batch.seq_lens
                ),
                None,
                block_kv_indices,
                self.req_to_token.stride(0),
                self.max_seqlen_pad,
                self.page_size,
            )
            self.forward_metadata = NpuMLAMetadata(
                block_kv_indices,
                forward_batch.seq_lens_cpu.tolist(),
                forward_batch,
            )
        else:
            self.forward_metadata = NpuMLAMetadata(
                None,
                forward_batch.extend_seq_lens_cpu,
                forward_batch,
            )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        pass

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
        forward_batch: ForwardBatch,
    ):
        self.forward_metadata = NpuMLAMetadata(
            torch.full(
                (bs, self.max_seqlen_pad),
                0,
                dtype=torch.int32,
                device=seq_lens.device,
            ),
            [1] * bs,
            forward_batch,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if k is not None:
            if save_kv_cache:
                if k_rope is not None:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        k,
                        k_rope,
                    )
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        k,
                        v,
                    )
        padding_bs = forward_batch.input_ids.size(0)
        if q_rope is not None:
            q_nope = q.view(padding_bs, -1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                padding_bs, -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            reshape_q = q.view(padding_bs, -1, layer.tp_q_head_num, layer.head_dim)
            q_nope = reshape_q[..., : layer.v_head_dim]
            q_rope = reshape_q[..., layer.v_head_dim :]
            if q_rope.numel() == 0:
                q_rope = None

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        if q_rope is None:
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        elif (
            hasattr(forward_batch.token_to_kv_pool, "enable_kv_cache_seperated")
            and forward_batch.token_to_kv_pool.enable_kv_cache_seperated
        ):
            v_cache = k_rope
        else:
            v_cache = k_cache

        o = self._run_npu_forward_decode(
            (q_nope, q_rope),
            k_cache if save_kv_cache else k,
            v_cache,
            layer,
            forward_batch,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = False,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if (
            forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
            or forward_batch.forward_mode.is_target_verify()
        ):
            if k_rope is not None:
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        k,
                        k_rope,
                    )
                    k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                        layer.layer_id
                    )
                    v_cache = k_cache
                else:
                    k_cache = k
                    v_cache = k_rope

                o = self._run_npu_forward_decode(
                    (q, q_rope), k_cache, v_cache, layer, forward_batch
                )
                return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            else:
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        k,
                        v,
                    )
                use_gqa = layer.tp_q_head_num != layer.tp_k_head_num
                return self._run_npu_forward_extend(
                    q, k, v, layer, forward_batch, use_gqa
                )
        else:
            raise NotImplementedError(f"unsupported {forward_batch.forward_mode=}")

    def _run_npu_forward_extend(self, q, k, v, layer, forward_batch, use_gqa=False):
        """
        q: (b*s, N, q_dim=192)
        k: (b*s, N, k_dim=192)
        v: (b*s, N, v_dim=128)
        """
        if not isinstance(q, Tuple):
            if q.ndim == 2:
                q = q.view(q.shape[0], self.num_local_heads, -1)
            bs_qlen, q_heads, q_dim = q.size()
        else:
            q_heads = self.num_local_heads
            q_dim = self.qk_rope_head_dim
        if not isinstance(k, Tuple):
            _, k_heads, k_dim = k.size()
        else:
            k_heads = self.num_local_heads
            k_dim = self.qk_rope_head_dim
        bs_qlen, v_heads, v_dim = v.size()

        if use_gqa:
            attn_output = torch.empty(
                bs_qlen, q_heads, v_dim, device=q.device, dtype=q.dtype
            )
            q_len_offset = 0
            for q_len in forward_batch.seq_len:
                attn_output[q_len_offset : q_len_offset + q_len] = (
                    torch.ops.npu.npu_fused_infer_attention_score(
                        q[None, q_len_offset : q_len_offset + q_len],
                        k[None, q_len_offset : q_len_offset + q_len],
                        v[None, q_len_offset : q_len_offset + q_len],
                        num_heads=q_heads,
                        num_key_value_heads=k_heads,
                        input_layout="BSND",  # todo, TND not supports q_heads!=k_heads
                        atten_mask=self.attn_mask.unsqueeze(0),
                        sparse_mode=3,
                        scale=layer.scaling,
                        next_tokens=0,
                    )[0]
                )
                q_len_offset += q_len
        else:  # MHA
            if q_dim != v_dim:
                if isinstance(k, Tuple):
                    q_nope, q_rope = q
                else:
                    q_nope, q_rope = q.split(
                        [self.v_head_dim, self.qk_rope_head_dim], dim=-1
                    )
                if isinstance(k, Tuple):
                    k_nope, k_rope = k
                else:
                    k_nope, k_rope = k.split(
                        [self.v_head_dim, self.qk_rope_head_dim], dim=-1
                    )

                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q_nope,
                    k_nope,
                    v,
                    query_rope=q_rope,
                    key_rope=k_rope,
                    num_heads=q_heads,
                    input_layout="TND",
                    atten_mask=self.attn_mask,
                    sparse_mode=3,
                    actual_seq_lengths=self.forward_metadata.kv_lens_list,  # cumsum
                    actual_seq_lengths_kv=self.forward_metadata.kv_lens_list,  # cumsum
                    scale=layer.scaling,
                    next_tokens=0,
                )
            else:
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q,
                    k,
                    v,
                    num_heads=q_heads,
                    input_layout="TND",
                    atten_mask=self.attn_mask,
                    sparse_mode=3,
                    actual_seq_lengths=self.forward_metadata.kv_lens_list,  # cumsum
                    actual_seq_lengths_kv=self.forward_metadata.kv_lens_list,  # cumsum
                    scale=layer.scaling,
                    next_tokens=0,
                )
            attn_output = attn_output[..., : layer.v_head_dim]

        return attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _run_npu_forward_decode(self, q, k_cache, v_cache, layer, forward_batch):
        """
        q: (b, s, N, q_dim=576)
        k_cache: (tokens_capticy, 1, k_dim=576)
        """
        if not isinstance(q, torch.Tensor):
            q_nope, q_rope = q
        else:
            q_nope, q_rope = q.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        n, q_nope_dim = q_nope.shape[-2:]

        if q_rope is not None:  # MLA
            if not forward_batch.token_to_kv_pool.enable_kv_cache_seperated:
                k_nope = k_cache[..., :q_nope_dim]
                k_rope = k_cache[..., q_nope_dim:]
            else:
                k_nope = k_cache
                k_rope = v_cache

            q_nope = q_nope.view(-1, n, self.kv_lora_rank)
            q_rope = q_rope.view(-1, n, self.qk_rope_head_dim)

            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                k_nope,
                k_nope,
                query_rope=q_rope,
                key_rope=k_rope,
                num_heads=n,
                num_key_value_heads=1,
                input_layout="TND",
                scale=layer.scaling,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=self.forward_metadata.block_kv_indices,
                block_size=self.page_size,
                actual_seq_lengths=self.forward_metadata.q_lens_list,  # cumsum
                actual_seq_lengths_kv=self.forward_metadata.kv_lens_list,  # non-cumsum
            )
        else:  # MHA
            _, k_heads, k_dim = k_cache.size()

            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                k_cache.view(-1, self.page_size, k_heads * k_dim),
                v_cache.view(-1, self.page_size, k_heads * k_dim),
                num_heads=n,
                num_key_value_heads=k_heads,
                input_layout="BSND",
                atten_mask=None,
                block_size=self.page_size,
                block_table=self.forward_metadata.block_kv_indices,
                actual_seq_lengths_kv=self.forward_metadata.kv_lens_list,  # non-cumsum
                scale=layer.scaling,
            )
        attn_output = attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        return attn_output


class NpuMLAAttnMultiStepDraftBackend:
    """
    Wrap multiple NpuMLA attention backends as one for multiple consecutive
    draft decoding steps
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps

        self.attn_backends = []
        for _ in range(self.speculative_num_steps):
            self.attn_backends.append(NpuMLABackend(model_runner))

    def common_template(self, forward_batch: ForwardBatch, call_fn: Callable):
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                forward_batch=forward_batch,
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=None,
            )

        self.common_template(forward_batch, call_fn)
