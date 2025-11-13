# Copyright 2024-2025 SGLang Team
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
"""Run the model with npu graph and torch.compile."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch
import tqdm
import bisect

from sglang.srt.configs.model_config import is_deepseek_nsa
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.model_executor.cuda_graph_runner import LogitsProcessorOutput
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.layers.dp_attention import (
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.utils import get_available_gpu_memory, get_compiler_backend
from sglang.srt.speculative.spec_utils import fast_topk

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker import EAGLEWorker


class EAGLEDraftExtendNpuGraphRunner(EAGLEDraftExtendCudaGraphRunner):
    def __init__(self, eagle_worker: EAGLEWorker):
        super().__init__(eagle_worker)
        self.compile_bs = [self.model_runner.server_args.max_running_requests // self.dp_size]
        self.forward_batch = {}
        if self.enable_torch_compile:
            self.warm_up()

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

    def _replay_update(self, seq_lens):
        self.graphs[self.bs].update(
            cpu_update_input=[{"actual_seq_lengths_kv": seq_lens}]
        )

    def _replay(self, forward_batch: ForwardBatch):
        if not is_deepseek_nsa(self.model_runner.model_config.hf_config):
            seq_lens = forward_batch.seq_lens_cpu.tolist() + [0] * (
                self.bs - self.raw_bs
            )
            thread = threading.Thread(target=self._replay_update, args=(seq_lens,))
            thread.start()
            self.graphs[self.bs].replay()
            thread.join()
        else:
            self.graphs[self.bs].replay()

    def replay(self, forward_batch: ForwardBatch):
        if not self.enable_torch_compile:
            return super().replay(forward_batch)

        # pad, similar with front part in replay of parent class
        raw_bs = forward_batch.batch_size
        num_tokens = forward_batch.input_ids.shape[0]
        max_num_tokens = max(forward_batch.global_num_tokens_cpu)

        bs = self.compile_bs[0]
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

        if forward_batch.extend_seq_lens_cpu is not None:
            self.forward_batch[bs].extend_seq_lens_cpu[:raw_bs] = forward_batch.extend_seq_lens_cpu

        if bs != raw_bs:
            self.forward_batch[bs].spec_info.positions = forward_batch.positions

        self.eagle_worker.draft_extend_attn_backend.init_forward_metadata(self.forward_batch[bs], can_run_graph=True)
        self.raw_bs = raw_bs
        self.bs = bs

        # replay
        # compile_method_name = f"compile_forward_{forward_batch.input_ids.size(0)}bs"
        # compile_forward = (
        #     getattr(self.model_runner.model, compile_method_name)
        #     if self.enable_cache
        #     else self.model_runner.model.compile_forward
        # )

        with torch.no_grad():
            out = self.model_runner.model.compile_forward(
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



    def prepare_forward_batch(self, bs: int, num_seqs: int) -> ForwardBatch:
        num_tokens = bs * self.num_tokens_per_bs

        # Graph inputs
        input_ids = self.input_ids[:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        seq_lens_cpu = self.seq_lens_cpu[:bs]
        extend_seq_lens = self.extend_seq_lens[:bs]
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:bs]
        accept_length = self.accept_length[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        mrope_positions = self.mrope_positions[:, :num_tokens]
        hidden_states = self.hidden_states[:num_tokens]
        next_token_logits_buffer = self.next_token_logits_buffer[
            : bs if self.forward_mode == ForwardMode.DRAFT_EXTEND else num_tokens
        ]

        if self.require_mlp_tp_gather:
            global_num_tokens = torch.tensor(
                    [
                    num_tokens // self.dp_size + (i < (num_tokens % self.dp_size))
                    for i in range(self.dp_size)
                    ],
                    dtype=torch.int64,
                    device=self.input_ids.device,
                )
            self.global_num_tokens_for_logprob_gpu = torch.tensor(
                    [bs] * self.dp_size,
                    dtype=torch.int64,
                    device=self.input_ids.device,
            )
            global_dp_buffer_len = num_tokens * self.dp_size
        elif self.require_attn_tp_gather:
            global_num_tokens = torch.tensor(
                    [num_tokens],
                    dtype=torch.int64,
                    device=self.input_ids.device,
                )
            self.global_num_tokens_for_logprob_gpu = torch.tensor(
                    [bs],
                    dtype=torch.int64,
                    device=self.input_ids.device,
                )
            global_dp_buffer_len = num_tokens
        else:
            global_num_tokens = None
            global_dp_buffer_len = None

        spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            accept_length=accept_length,
        )
        spec_info.positions = None

        self.deepep_adapter.capture(is_extend_in_batch=True)

        # Forward batch
        forward_batch = ForwardBatch(
            forward_mode=self.forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=global_num_tokens,
            global_num_tokens_for_logprob_gpu=self.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            attn_backend=self.eagle_worker.draft_extend_attn_backend,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            padded_static_len=self.padded_static_len,
        )
        return forward_batch

    def warm_up(self):
        @torch.compile(dynamic=True, backend=get_compiler_backend())
        def run_for_init(input):
            return input + 1

        input = torch.zeros([1]).to(self.model_runner.device)
        run_for_init(input)

        backend = get_compiler_backend()
        self.model_runner.model.compile_forward = torch.compile(
            torch.no_grad()(self.model_runner.model.forward),
            fullgraph=True,
            dynamic=False,
            backend=backend,
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
                    f"Capturing mtp1 batches ({bs=} {avail_mem=:.2f} GB)"
                )
            num_tokens = bs * self.num_tokens_per_bs
            self.forward_batch[bs] = self.prepare_forward_batch(bs, num_tokens)
            set_dp_buffer_len(self.forward_batch[bs].global_dp_buffer_len, num_tokens)
            set_is_extend_in_batch(False)
            self.forward_batch[bs].attn_backend.init_forward_metadata(self.forward_batch[bs], can_run_graph=True)

            # Run and compile
            def run_once():
                # Clean intermediate result cache for DP attention
                self.forward_batch[bs].dp_local_start_pos = self.forward_batch[bs].dp_local_num_tokens = None

                # Backup two fields, which will be modified in-place in `draft_forward`.
                output_cache_loc_backup = self.forward_batch[bs].out_cache_loc
                hidden_states_backup = self.forward_batch[bs].spec_info.hidden_states

                with torch.no_grad():
                    ret = self.model_runner.model.compile_forward(
                        self.forward_batch[bs].input_ids,
                        self.forward_batch[bs].positions,
                        self.forward_batch[bs],
                    )
                probs = torch.softmax(ret.next_token_logits, dim=-1)
                ret.topk_p, ret.topk_index = fast_topk(probs, self.topk, dim=-1)

                self.forward_batch[bs].out_cache_loc = output_cache_loc_backup
                self.forward_batch[bs].spec_info.hidden_states = hidden_states_backup
                return ret

            torch.npu.synchronize()
            self.model_runner.tp_group.barrier()
            run_once()

    def can_run(self, forward_batch: ForwardBatch):
        if not self.enable_torch_compile:
            return super().can_run(forward_batch)

        return (
            not forward_batch.is_extend_in_batch
            # and forward_batch.batch_size in self.compile_bs
        )