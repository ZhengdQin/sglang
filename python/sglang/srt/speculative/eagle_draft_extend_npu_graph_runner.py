# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run the model with npu graph and torch.compile"""

from __future__ import annotations

import bisect
import inspect
import os
import threading
from typing import TYPE_CHECKING

import torch
import tqdm

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.dp_attention import DPPaddingMode
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.npu_graph_runner import NPUGraphRunner
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_utils import EagleDraftInput
from sglang.srt.utils import fast_topk, get_available_gpu_memory, get_compiler_backend

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker import EAGLEWorker


class EAGLEDraftExtendNpuGraphRunner(EAGLEDraftExtendCudaGraphRunner):
    def __init__(self, eagle_worker: EAGLEWorker):
        self.enable_cache = False
        self.pp_size = 1
        self.forward_batch = {}
        super().__init__(eagle_worker)

    def _create_graph(self):
        return torch.npu.NPUGraph()

    def _capture_init(self, run_once_fn):
        for _ in range(2):
            torch.npu.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.npu.graph(
            graph, pool=pool, stream=stream, auto_dispatch_capture=True
        ):
            out = run_once_fn()
        return out

    def replay(self, forward_batch: ForwardBatch):
        if self.enable_torch_compile:
            # pad, similar with front part in replay of parent class
            raw_bs = forward_batch.batch_size
            num_tokens = forward_batch.input_ids.shape[0]
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            if self.require_mlp_tp_gather:
                max_batch_size = (
                    max_num_tokens // self.num_tokens_per_bs
                    if self.model_runner.spec_algorithm.is_eagle()
                    else max_num_tokens
                )
                index = bisect.bisect_left(self.capture_bs, max_batch_size)
            else:
                index = bisect.bisect_left(self.capture_bs, raw_bs)

            bs = self.capture_bs[index]
            if bs * self.num_tokens_per_bs != num_tokens:
                self.forward_batch[bs].seq_lens.fill_(self.seq_len_fill_value)
                self.forward_batch[bs].extend_seq_lens.fill_(1)
                self.forward_batch[bs].out_cache_loc.zero_()
                self.forward_batch[bs].spec_info.accept_length.fill_(1)

            # Common inputs
            self.forward_batch[bs].input_ids[:num_tokens].copy_(forward_batch.input_ids)
            self.forward_batch[bs].seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
            if forward_batch.extend_seq_lens is not None:
                self.forward_batch[bs].extend_seq_lens[:raw_bs].copy_(
                    forward_batch.extend_seq_lens
                )
            self.forward_batch[bs].out_cache_loc[:num_tokens].copy_(
                forward_batch.out_cache_loc
            )
            self.forward_batch[bs].positions[:num_tokens].copy_(forward_batch.positions)
            self.forward_batch[bs].spec_info.hidden_states[:num_tokens].copy_(
                forward_batch.spec_info.hidden_states
            )
            if forward_batch.spec_info.accept_length is not None:
                self.forward_batch[bs].spec_info.accept_length[:raw_bs].copy_(
                    forward_batch.spec_info.accept_length
                )
            self.forward_batch[bs].req_pool_indices[:raw_bs].copy_(
                forward_batch.req_pool_indices
            )

            # TODO(ch-wan): support num_token_non_padded
            if self.require_gathered_buffer:
                self.forward_batch[bs].global_num_tokens_gpu.fill_(
                    bs * self.num_tokens_per_bs
                )
                self.forward_batch[bs].global_num_tokens_for_logprob_gpu.fill_(bs)

            if forward_batch.seq_lens_cpu is not None:
                if bs != raw_bs:
                    self.forward_batch[bs].seq_lens_cpu.fill_(self.seq_len_fill_value)
                self.forward_batch[bs].seq_lens_cpu[:raw_bs].copy_(
                    forward_batch.seq_lens_cpu
                )

            if bs != raw_bs:
                self.forward_batch[bs].spec_info.positions = forward_batch.positions

            self.eagle_worker.draft_extend_attn_backend.init_forward_metadata_replay_cuda_graph(
                self.forward_batch[bs]
                # bs=bs,
                # req_pool_indices=self.forward_batch[bs].req_pool_indices,
                # seq_lens=self.forward_batch[bs].seq_lens,
                # seq_lens_sum=forward_batch.seq_lens_sum
                # + (bs - raw_bs) * self.seq_len_fill_value,
                # encoder_lens=None,
                # forward_mode=ForwardMode.DRAFT_EXTEND,
                # spec_info=self.forward_batch[bs].spec_info,
                # seq_lens_cpu=self.forward_batch[bs].seq_lens_cpu,
            )
            self.raw_bs = raw_bs
            self.bs = bs

            # replay
            compile_method_name = f"compile_forward_{forward_batch.input_ids.size(0)}bs"
            compile_forward = (
                getattr(self.model_runner.model, compile_method_name)
                if self.enable_cache
                else self.model_runner.model.compile_forward
            )

            with torch.no_grad():
                out = compile_forward(
                    self.forward_batch[bs].input_ids,
                    self.forward_batch[bs].positions,
                    self.forward_batch[bs],
                )
                probs = torch.softmax(out.next_token_logits, dim=-1)
                out.topk_p, out.topk_index = fast_topk(probs, self.topk, dim=-1)
            # npu need start
            if bs != raw_bs:
                forward_batch.spec_info.accept_length = self.forward_batch[
                    bs
                ].spec_info.accept_length[:raw_bs]
                out_copy = out
                out = LogitsProcessorOutput(
                    next_token_logits=out.next_token_logits[:raw_bs],
                    hidden_states=out.hidden_states[:raw_bs],
                )
                out.topk_p = out_copy.topk_p[:raw_bs]
                out.topk_index = out_copy.topk_index[:raw_bs]
            # npu need end
            return out
        else:
            return super().replay(forward_batch)

    def prepare_forward_batch(self, bs: int, num_tokens: int) -> ForwardBatch:
        # Graph inputs
        with torch.device(self.model_runner.device):
            input_ids = torch.zeros((num_tokens,), dtype=torch.int64)
            req_pool_indices = torch.zeros((bs,), dtype=torch.int64)
            seq_lens = torch.full((bs,), self.seq_len_fill_value, dtype=torch.int64)
            out_cache_loc = torch.zeros((num_tokens,), dtype=torch.int64)
            positions = torch.zeros((num_tokens,), dtype=torch.int64)
            extend_seq_lens = torch.ones((bs,), dtype=torch.int32)
            accept_length = torch.full((bs,), self.num_tokens_per_bs, dtype=torch.int32)
            if self.eagle_worker.speculative_algorithm.is_eagle3():
                hidden_states = torch.zeros(
                    (
                        num_tokens,
                        (
                            self.model_runner.model_config.hf_config.target_hidden_size
                            * 3
                            if hasattr(
                                self.model_runner.model_config.hf_config,
                                "target_hidden_size",
                            )
                            else self.model_runner.model_config.hidden_size * 3
                        ),
                    ),
                    dtype=self.model_runner.dtype,
                )
            else:
                hidden_states = torch.zeros(
                    (num_tokens, self.model_runner.model_config.hidden_size),
                    dtype=self.model_runner.dtype,
                )

            gathered_buffer = None
            if self.require_gathered_buffer:
                gathered_buffer = torch.zeros(
                    (
                        bs * self.dp_size if self.dp_size > 1 else num_tokens,
                        self.model_runner.model_config.hidden_size,
                    ),
                    dtype=self.model_runner.dtype,
                )
            next_token_logits_buffer = torch.zeros(
                (bs, self.model_runner.model_config.vocab_size),
                dtype=torch.float,
            )

        if self.require_mlp_tp_gather:
            global_num_tokens_gpu = torch.tensor(
                [num_tokens] * self.dp_size,
                dtype=torch.int64,
                device=input_ids.device,
            )
        elif self.require_attn_tp_gather:
            global_num_tokens_gpu = torch.tensor(
                [num_tokens], dtype=torch.int64, device=input_ids.device
            )
        else:
            global_num_tokens_gpu = None
            gathered_buffer = None

        spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            accept_length=accept_length,
        )
        spec_info.positions = torch.zeros((num_tokens,), dtype=torch.int64)

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens.cpu().tolist(),
            next_token_logits_buffer=next_token_logits_buffer,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            global_num_tokens_cpu=global_num_tokens_gpu.cpu().tolist(),
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=self.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DPPaddingMode.get_default_mode_in_cuda_graph(),
            gathered_buffer=gathered_buffer,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            attn_backend=self.eagle_worker.draft_extend_attn_backend,
            padded_static_len=self.padded_static_len,
            can_run_graph=True,
        )
        self.eagle_worker.draft_extend_attn_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=None,
            forward_mode=ForwardMode.DRAFT_EXTEND,
            spec_info=spec_info,
            forward_batch=forward_batch,
        )
        return forward_batch

    def capture(self):
        if self.enable_torch_compile:
            self.model_runner.model.compile_forward = torch.compile(
                torch.no_grad()(self.eagle_worker.draft_model_runner.model.forward),
                fullgraph=True,
                dynamic=True,
                backend=get_compiler_backend(),
            )

            compile_range = (
                tqdm.tqdm(list(reversed(self.compile_bs)))
                if get_tensor_model_parallel_rank() == 0
                else reversed(self.compile_bs)
            )
            for bs in compile_range:
                if get_tensor_model_parallel_rank() == 0:
                    avail_mem = get_available_gpu_memory(
                        self.model_runner.device,
                        self.model_runner.gpu_id,
                        empty_cache=False,
                    )
                    compile_range.set_description(
                        f"Capturing batches ({bs=}, {avail_mem=:.2f} GB)"
                    )
                num_tokens = bs * self.num_tokens_per_bs
                self.forward_batch[bs] = self.prepare_forward_batch(bs, num_tokens)

                # Run and capture
                def run_once():
                    # Clean intermediate result cache for DP attention
                    self.forward_batch[bs].dp_local_start_pos = self.forward_batch[
                        bs
                    ].dp_local_num_tokens = None

                    kwargs = {}
                    if (
                        self.pp_size > 1
                        and "pp_proxy_tensors"
                        in inspect.signature(
                            self.model_runner.model.compile_forward
                        ).parameters
                    ):
                        kwargs["pp_proxy_tensors"] = self.forward_batch[
                            bs
                        ].pp_proxy_tensors
                    self.mark_static(
                        self.forward_batch[bs], kwargs.get("pp_proxy_tensors")
                    )

                    compile_forward = self.model_runner.model.compile_forward

                    with torch.no_grad():
                        logits_output_or_pp_proxy_tensors = compile_forward(
                            self.forward_batch[bs].input_ids,
                            self.forward_batch[bs].positions,
                            self.forward_batch[bs],
                            **kwargs,
                        )
                        return logits_output_or_pp_proxy_tensors

                for _ in range(2):
                    torch.npu.synchronize()
                    self.model_runner.tp_group.barrier()
                    run_once()
        elif not self.model_runner.server_args.disable_cuda_graph:
            super().capture()

    def replay_update(self, seq_lens):
        self.graphs[self.bs].update(
            cpu_update_input=[{"actual_seq_lengths_kv": seq_lens}]
        )

    def _update_and_replay(self, forward_batch: ForwardBatch):
        seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (self.bs - self.raw_bs)
        thread = threading.Thread(target=self.replay_update, args=(seq_lens,))
        thread.start()
        self.graphs[self.bs].replay()
        thread.join()

    def mark_static(self, forward_batch, pp_proxy_tensors: PPProxyTensors):
        NPUGraphRunner.mark_static(self, forward_batch, pp_proxy_tensors)

    def can_run(self, forward_batch: ForwardBatch):
        if self.enable_torch_compile:
            if self.require_mlp_tp_gather:
                cuda_graph_bs = (
                    max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                    if self.model_runner.spec_algorithm.is_eagle()
                    else max(forward_batch.global_num_tokens_cpu)
                )
            else:
                cuda_graph_bs = forward_batch.seq_lens.numel()

            is_bs_supported = (
                cuda_graph_bs in self.compile_bs
                if self.disable_padding
                else cuda_graph_bs <= self.max_bs
            )

            # if self.require_mlp_sync:
            #     is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

            # is_bs_supported = (
            #     is_bs_supported
            #     and forward_batch.global_num_tokens_cpu[0]
            #     - (forward_batch.seq_lens_sum - forward_batch.seq_lens_cpu[-1])
            #     <= 16  # TND QS of each batch computed by actual_seq_lengths should be in range [0, 16]
            # )
            return is_bs_supported
        else:
            return super().can_run(forward_batch)
