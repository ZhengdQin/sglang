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

import inspect
import logging
import threading
from typing import TYPE_CHECKING

import torch
import tqdm

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.dp_attention import DPPaddingMode
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.npu_graph_runner import NPUGraphRunner
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_utils import EagleDraftInput
from sglang.srt.utils import get_available_gpu_memory, get_compiler_backend, is_npu

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

logger = logging.getLogger(__name__)

if is_npu():
    torch.cuda.CUDAGraph = torch.npu.NPUGraph
    torch.cuda.synchronize = torch.npu.synchronize
    torch.cuda.graph = torch.npu.graph
    torch.cuda.stream = torch.npu.stream
    torch.cuda.Stream = torch.npu.Stream
    torch.cuda.current_stream = torch.npu.current_stream


class EAGLEDraftNpuGraphRunner(EAGLEDraftCudaGraphRunner):
    def __init__(self, eagle_worker: EAGLEWorker):
        self.enable_cache = False
        self.pp_size = 1
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

    def replay_update(self, seq_lens):
        self.graphs[self.bs].update(
            cpu_update_input=[{"actual_seq_lengths_kv": seq_lens}]
        )

    def prepare_forward_batch(self, bs: int, num_tokens: int) -> ForwardBatch:
        # Graph inputs, it is similar with capture_one_batch_size in parent class
        with torch.device(self.model_runner.device):
            input_ids = torch.zeros((num_tokens,), dtype=torch.int64)
            req_pool_indices = torch.zeros((num_tokens,), dtype=torch.int64)
            seq_lens = torch.full(
                (num_tokens,), self.seq_len_fill_value, dtype=torch.int64
            )
            out_cache_loc = torch.zeros(
                (num_tokens * self.speculative_num_steps,), dtype=torch.int32
            )
            positions = torch.zeros((num_tokens,), dtype=torch.int64)
            topk_p = torch.zeros((num_tokens, self.topk), dtype=torch.float32)
            topk_index = torch.zeros((num_tokens, self.topk), dtype=torch.int64)
            hidden_states = torch.zeros(
                (num_tokens, self.model_runner.model_config.hidden_size),
                dtype=self.model_runner.dtype,
            )
            num_token_non_padded = torch.tensor(num_tokens, dtype=torch.int32)

        num_tokens = bs * self.num_tokens_per_bs

        if self.require_mlp_tp_gather:
            global_num_tokens = torch.tensor(
                [num_tokens] * self.dp_size,
                dtype=torch.int32,
                device=self.input_ids.device,
            )
            global_num_tokens_for_logprob = torch.tensor(
                [num_tokens] * self.dp_size,
                dtype=torch.int32,
                device=self.input_ids.device,
            )
            gathered_buffer = torch.zeros(
                (
                    num_tokens * self.dp_size,
                    self.model_runner.model_config.hidden_size,
                ),
                dtype=self.model_runner.dtype,
            )
        elif self.require_attn_tp_gather:
            global_num_tokens = torch.tensor(
                [num_tokens],
                dtype=torch.int32,
                device=self.input_ids.device,
            )
            global_num_tokens_for_logprob = torch.tensor(
                [num_tokens],
                dtype=torch.int32,
                device=self.input_ids.device,
            )
            gathered_buffer = torch.zeros(
                (
                    num_tokens,
                    self.model_runner.model_config.hidden_size,
                ),
                dtype=self.model_runner.dtype,
            )
        else:
            global_num_tokens = None
            gathered_buffer = None
            global_num_tokens_for_logprob = None

        spec_info = EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden_states,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

        # Forward batch, npu needs more params
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=num_tokens,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.eagle_worker.draft_attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=global_num_tokens,
            gathered_buffer=gathered_buffer,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=(
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            ),
            num_token_non_padded=num_token_non_padded,
            global_forward_mode=None,
            mm_inputs=[None] * bs,
            lora_ids=[None] * bs,
            global_num_tokens_cpu=[num_tokens],
            global_num_tokens_for_logprob_cpu=[num_tokens],
            global_num_tokens_for_logprob_gpu=global_num_tokens.clone(),
            can_run_graph=True,
        )

        # Attention backend
        self.eagle_worker.draft_attn_backend.init_forward_metadata_capture_cuda_graph(
            forward_batch
        )
        return forward_batch

    def capture(self):
        return
        if self.enable_torch_compile:
            self.model_runner.model.compile_forward = torch.compile(
                torch.no_grad()(self.eagle_worker.draft_forward),
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
                forward_batch = self.prepare_forward_batch(bs, num_tokens)

                # Run and capture
                def run_once():
                    # Clean intermediate result cache for DP attention
                    forward_batch.dp_local_start_pos = (
                        forward_batch.dp_local_num_tokens
                    ) = None

                    kwargs = {}
                    if (
                        self.pp_size > 1
                        and "pp_proxy_tensors"
                        in inspect.signature(
                            self.model_runner.model.compile_forward
                        ).parameters
                    ):
                        kwargs["pp_proxy_tensors"] = forward_batch.pp_proxy_tensors
                    self.mark_static(forward_batch, kwargs.get("pp_proxy_tensors"))

                    compile_forward = self.model_runner.model.compile_forward

                    with torch.no_grad():
                        logits_output_or_pp_proxy_tensors = compile_forward(
                            forward_batch.input_ids,
                            forward_batch.positions,
                            forward_batch,
                            **kwargs,
                        )
                        return logits_output_or_pp_proxy_tensors

                for _ in range(2):
                    torch.npu.synchronize()
                    self.model_runner.tp_group.barrier()
                    run_once()
        elif not self.model_runner.server_args.disable_cuda_graph:
            super().capture()

    def _update_and_replay(self, forward_batch: ForwardBatch):
        if self.enable_torch_compile:
            bs = self.bs
            raw_bs = forward_batch.batch_size
            num_tokens = bs * self.num_tokens_per_bs
            forward_batch.input_ids = self.input_ids[:bs]
            compile_method_name = f"compile_forward_{forward_batch.input_ids.size(0)}bs"
            compile_forward = self.model_runner.model.compile_forward

            with torch.no_grad():
                return compile_forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                )
        else:
            seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (
                self.bs - self.raw_bs
            )
            thread = threading.Thread(target=self.replay_update, args=(seq_lens,))
            thread.start()
            self.graphs[self.bs].replay()
            thread.join()

    def mark_static(self, forward_batch, pp_proxy_tensors: PPProxyTensors):
        NPUGraphRunner.mark_static(self, forward_batch, pp_proxy_tensors)

    def can_run(self, forward_batch: ForwardBatch):
        return False
