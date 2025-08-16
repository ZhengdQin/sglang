from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch_npu
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class ForwardMetadata:

    # calculated map for kv positions [bs * maxseqlen]
    block_tables: Optional[torch.Tensor] = None

    # seq len inputs
    extend_seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_int: Optional[torch.Tensor] = None


class AscendAttnBackend(AttentionBackend):

    def gen_attention_mask(self, max_seq_len: int, dtype=torch.float16):
        mask_flag = torch.tril(
            torch.ones((max_seq_len, max_seq_len), dtype=torch.bool)
        ).view(max_seq_len, max_seq_len)
        mask_flag = ~mask_flag
        if dtype == torch.float16:
            mask_value = torch.finfo(torch.float32).min
        else:
            mask_value = 1
        self.mask = (
            torch.masked_fill(
                torch.zeros(size=(max_seq_len, max_seq_len)), mask_flag, mask_value
            )
            .to(dtype)
            .to(self.device)
        )
        self.mask_len = max_seq_len

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = ForwardMetadata()
        self.device = model_runner.device
        self.gen_attention_mask(128, model_runner.dtype)
        self.page_size = model_runner.page_size
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        if self.use_mla:
            self.kv_lora_rank = model_runner.model_config.kv_lora_rank
            self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.graph_metadata = {}
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.graph_mode = False
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.mtp_mask = torch.tril(torch.ones(2048, 2048, dtype=torch.bool)).npu()
        self.mtp_mask = ~self.mtp_mask

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        self.forward_metadata = ForwardMetadata()
        seq_lens_max = forward_batch.seq_lens.max()
        if forward_batch.forward_mode.is_target_verify():
            seq_lens_max += self.speculative_num_draft_tokens
        self.forward_metadata.block_tables = (
            forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, :seq_lens_max
            ][:, :: self.page_size]
            // self.page_size
        )
        if forward_batch.extend_seq_lens is not None:
            self.forward_metadata.extend_seq_lens_cpu_int = (
                forward_batch.extend_seq_lens.cpu().int()
            )
        self.forward_metadata.seq_lens_cpu_int = forward_batch.seq_lens_cpu.int()
        self.forward_metadata.seq_lens_cpu_list_cumsum = torch.cumsum(
            forward_batch.seq_lens_cpu, dim=0
        ).tolist()
        if forward_batch.forward_mode.is_target_verify():
            self.forward_metadata.seq_lens_cpu_int += self.speculative_num_draft_tokens

        self.graph_mode = False

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.graph_metadata = {
            "block_tables": torch.empty(
                (max_bs, self.max_context_len // self.page_size),
                dtype=torch.int32,
                device=self.device,
            ),
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        metadata = ForwardMetadata()

        metadata.block_tables = self.graph_metadata["block_tables"][:bs, :]
        metadata.seq_lens_cpu_list = seq_lens.cpu().int().tolist()

        self.graph_metadata[bs] = metadata
        self.forward_metadata = metadata

        self.graph_mode = True

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        metadata = self.graph_metadata[bs]
        max_len = seq_lens_cpu[:bs].max().item()
        max_seq_pages = (max_len + self.page_size - 1) // self.page_size

        metadata.block_tables[:bs, :max_seq_pages].copy_(
            self.req_to_token[req_pool_indices[:bs], :max_len][:, :: self.page_size]
            // self.page_size
        )
        metadata.block_tables[:bs, max_seq_pages:].fill_(0)
        metadata.block_tables[bs:, :].fill_(0)

        self.forward_metadata = metadata

        self.graph_mode = True

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            if _use_mlapo:
                save_kv_cache = False
            return self.forward_mtp(q, k, v, layer, forward_batch, save_kv_cache)
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        if not self.use_mla:
            query = q.view(-1, layer.tp_q_head_num * layer.qk_head_dim)
            output = torch.empty(
                (query.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                dtype=query.dtype,
                device=query.device,
            )

            torch_npu._npu_flash_attention_qlens(
                query=query,
                key_cache=k_cache,
                value_cache=v_cache,
                mask=self.mask,
                block_table=self.forward_metadata.block_tables,
                seq_len=self.forward_metadata.extend_seq_lens_cpu_int,
                context_lens=self.forward_metadata.seq_lens_cpu_int,
                scale_value=layer.scaling,
                num_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                out=output,
            )
            return output
        else:
            mask = self.attn_mask_builder.get_attn_mask(2048, q.dtype, q.device).to(
                torch.int8
            )
            num_token_padding = q.shape[0]
            q, k, v = [
                data[: forward_batch.num_token_non_padded_cpu] for data in [q, k, v]
            ]
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                q,
                k,
                v,
                atten_mask=mask,
                sparse_mode=3,
                actual_seq_lengths=self.forward_metadata.seq_lens_cpu_list_cumsum,
                actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_list_cumsum,
                num_heads=layer.tp_q_head_num,
                input_layout="TND",
                scale=layer.scaling,
                next_tokens=0,
            )
            attn_output = attn_output.reshape(-1, layer.tp_q_head_num, layer.v_head_dim)
            if num_token_padding != forward_batch.num_token_non_padded_cpu:
                attn_output = torch.cat(
                    [
                        attn_output,
                        attn_output.new_zeros(
                            num_token_padding - attn_output.shape[0],
                            *attn_output.shape[1:],
                        ),
                    ],
                    dim=0,
                )
            return attn_output

    def forward_mtp(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
    ):
        if save_kv_cache:
            if self.use_mla:
                cache_kv = k.view(-1, layer.tp_k_head_num, layer.head_dim)
                k = cache_kv[:, :, : layer.v_head_dim]
                v = cache_kv[:, :, layer.v_head_dim :]
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
        # When kv separation, head_dim needs to be split into 512 and 64.
        head_dim = layer.v_head_dim
        v_head_dim = layer.head_dim - layer.v_head_dim
        c_kv, k_rope = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_rope_cache = k_rope.view(-1, layer.tp_k_head_num, self.page_size, v_head_dim)
        c_kv_cache = c_kv.view(-1, layer.tp_v_head_num, self.page_size, head_dim)
        q = q.contiguous()
        q_nope = q[:, :, : layer.v_head_dim].contiguous()
        q_rope = q[:, :, layer.v_head_dim :].contiguous()
        if not self.graph_mode:
            num_token_padding = q.shape[0]
            q_nope = q_nope[: forward_batch.num_token_non_padded_cpu]
            q_rope = q_rope[: forward_batch.num_token_non_padded_cpu]
        if self.forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
        else:
            actual_seq_lengths_kv = (
                self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
            )
        if forward_batch.forward_mode.is_target_verify():
            actual_seq_lengths = np.arange(
                self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens + q_nope.shape[0],
                self.speculative_num_draft_tokens,
            )
        else:
            actual_seq_lengths = (
                np.array(forward_batch.extend_seq_lens_cpu).cumsum().tolist()
            )
        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            q_nope,
            c_kv_cache,
            c_kv_cache,
            query_rope=q_rope,
            key_rope=k_rope_cache,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            scale=layer.scaling,
            antiquant_mode=0,
            antiquant_scale=None,
            block_table=self.forward_metadata.block_tables,
            block_size=self.page_size,
            sparse_mode=3,
            atten_mask=self.mtp_mask,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
        )
        attn_output = torch.zeros_like(q_nope, dtype=q.dtype, device=q.device)
        softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)
        torch_npu.npu_fused_infer_attention_score.out(
            q_nope,
            c_kv_cache,
            c_kv_cache,
            query_rope=q_rope,
            key_rope=k_rope_cache,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            scale=layer.scaling,
            antiquant_mode=0,
            antiquant_scale=None,
            block_table=self.forward_metadata.block_tables,
            block_size=self.page_size,
            sparse_mode=3,
            atten_mask=self.mtp_mask,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            workspace=workspace,
            out=[attn_output, softmax_lse],
        )
        attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        if (
            not self.graph_mode
            and forward_batch.num_token_non_padded_cpu != num_token_padding
        ):
            attn_output = torch.cat(
                [
                    attn_output,
                    attn_output.new_zeros(
                        num_token_padding - attn_output.shape[0], *attn_output.shape[1:]
                    ),
                ],
                dim=0,
            )
        return attn_output

    def forward_decode_graph(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        if not self.use_mla:
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                layer.layer_id
            ).view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                layer.layer_id
            ).view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
            query = q.view(-1, 1, layer.tp_q_head_num * layer.qk_head_dim)
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                o = torch.empty_like(q)

            use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

            q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

            causal = True
            if (
                layer.is_cross_attention
                or layer.attn_type == AttentionType.ENCODER_ONLY
            ):
                causal = False

            self.native_attn._run_sdpa_forward_extend(
                q_,
                o_,
                k_cache.view(
                    -1, layer.tp_k_head_num, (self.kv_lora_rank + self.qk_rope_head_dim)
                ),
                v_cache.view(-1, layer.tp_v_head_num, self.kv_lora_rank),
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.extend_prefix_lens,
                forward_batch.extend_seq_lens,
                scaling=layer.scaling,
                enable_gqa=use_gqa,
                causal=causal,
            )
            return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
        if not self.use_mla:
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            query = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            num_tokens = query.shape[0]
            output = torch.empty(
                (num_tokens, layer.tp_q_head_num, layer.v_head_dim),
                dtype=query.dtype,
                device=query.device,
            )

            torch_npu._npu_paged_attention(
                query=query,
                key_cache=k_cache,
                value_cache=v_cache,
                num_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                scale_value=layer.scaling,
                block_table=self.forward_metadata.block_tables,
                context_lens=self.forward_metadata.seq_lens_cpu_int,
                out=output,
            )
            return output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            num_tokens = query.shape[0]
            kv_c_and_k_pe_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                layer.layer_id
            )
            kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
                -1,
                self.page_size,
                layer.tp_k_head_num,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )

            attn_output = torch.empty(
                [num_tokens, layer.tp_q_head_num, self.kv_lora_rank],
                dtype=q.dtype,
                device=q.device,
            )
            torch_npu._npu_paged_attention_mla(
                query=query,
                key_cache=kv_c_and_k_pe_cache,
                num_kv_heads=layer.tp_k_head_num,
                num_heads=layer.tp_q_head_num,
                scale_value=layer.scaling,
                block_table=self.forward_metadata.block_tables,
                context_lens=self.forward_metadata.seq_lens_cpu_int,
                mla_vheadsize=self.kv_lora_rank,
                out=attn_output,
            )
            return attn_output.view(num_tokens, layer.tp_q_head_num * self.kv_lora_rank)


class AscendAttnMultiStepDraftBackend:
    """
    Wrap multiple Ascend attention backends as one for multiple consecutive
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
            self.attn_backends.append(AscendAttnBackend(model_runner))

    def common_template(self, forward_batch: ForwardBatch, call_fn: int):
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
