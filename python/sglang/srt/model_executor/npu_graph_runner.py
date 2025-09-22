# Copyright 2023-2024 SGLang Team
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

import bisect
import inspect
import logging
import os
import threading
import types
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional, Union

import torch
import tqdm

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.dp_attention import DPPaddingMode
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    PPProxyTensors,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.graph_runner import GraphRunner
from sglang.srt.utils import get_available_gpu_memory, get_compiler_backend

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class NPUGraphRunner(GraphRunner):
    """A NPUGraphRunner runs the forward pass of a model with npu graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        if model_runner.is_draft_worker:
            pass
        else:
            if self.enable_torch_compile:
                self.enable_cache = os.getenv("ENABLE_TORCH_COMPILE_CACHE", "0") == "1"
                self.warm_up()
            elif (
                not model_runner.server_args.disable_cuda_graph
            ):  # todo (wangq) support aclgraph in main model
                super().warm_up()

    def _create_device_graph(self):
        return torch.npu.NPUGraph()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.npu.graph(
            graph,
            pool=pool,
            stream=stream,
            auto_dispatch_capture=True,
        ):
            out = run_once_fn()
        return out

    def _update_inputs(self, seq_lens):
        self.graphs[self.bs].update(
            cpu_update_input=[{"actual_seq_lengths_kv": seq_lens}]
        )

    def _cache_loc_dtype(self):
        return torch.int32

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        if self.enable_torch_compile:
            # pad, similar with replay_prepare
            raw_bs = forward_batch.batch_size
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            if self.require_mlp_tp_gather:
                max_batch_size = (
                    max_num_tokens / self.num_tokens_per_bs
                    if self.model_runner.spec_algorithm.is_eagle()
                    else max_num_tokens
                )
                index = bisect.bisect_left(self.capture_bs, max_batch_size)
            else:
                index = bisect.bisect_left(self.capture_bs, raw_bs)
            bs = self.capture_bs[index]
            if bs != raw_bs:
                forward_batch.seq_lens = forward_batch._pad_tensor_to_size(
                    forward_batch.seq_lens, bs, value=self.seq_len_fill_value
                )
                forward_batch.seq_lens_cpu = forward_batch._pad_tensor_to_size(
                    forward_batch.seq_lens_cpu, bs, value=self.seq_len_fill_value
                )
                if forward_batch.extend_seq_lens is not None:
                    forward_batch.extend_seq_lens = forward_batch._pad_tensor_to_size(
                        forward_batch.extend_seq_lens, bs, value=self.seq_len_fill_value
                    )
            num_token = bs * self.num_tokens_per_bs
            if num_token != max_num_tokens:
                forward_batch.input_ids = forward_batch._pad_tensor_to_size(
                    forward_batch.input_ids, num_token
                )
                forward_batch.out_cache_loc = forward_batch._pad_tensor_to_size(
                    forward_batch.out_cache_loc, num_token
                )
                forward_batch.positions = forward_batch._pad_tensor_to_size(
                    forward_batch.positions, num_token
                )
                forward_batch.req_pool_indices = forward_batch._pad_tensor_to_size(
                    forward_batch.req_pool_indices, bs
                )
                forward_batch.gathered_buffer = torch.zeros(
                    (num_token, self.model_runner.model_config.hidden_size),
                    dtype=self.model_runner.dtype,
                    device=self.model_runner.device,
                )
            # NPU need start
            if (
                forward_batch.extend_seq_lens_cpu is not None
                and len(forward_batch.extend_seq_lens_cpu) != bs
            ):
                forward_batch.extend_seq_lens_cpu += [self.seq_len_fill_value] * (
                    bs - len(forward_batch.extend_seq_lens_cpu)
                )
            if forward_batch.extend_logprob_start_lens_cpu is not None:
                l = len(forward_batch.extend_logprob_start_lens_cpu)
                if l < bs:
                    forward_batch.extend_logprob_start_lens_cpu += [
                        self.seq_len_fill_value
                    ] * (bs - l)
                elif l > bs:
                    forward_batch.extend_logprob_start_lens_cpu = (
                        forward_batch.extend_logprob_start_lens_cpu[:bs]
                    )

            forward_batch.attn_backend.init_forward_metadata(forward_batch)
            # NPU need end

            kwargs = {}
            if pp_proxy_tensors is not None:
                kwargs["pp_proxy_tensors"] = pp_proxy_tensors

            compile_method_name = f"compile_forward_{bs}bs"
            compile_forward = (
                getattr(self.model_runner.model, compile_method_name)
                if self.enable_cache
                else self.model_runner.model.compile_forward
            )
            with torch.no_grad():
                return compile_forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )

        else:
            if not skip_attn_backend_init:
                self.replay_prepare(forward_batch, pp_proxy_tensors)
            else:
                # In speculative decoding, these two fields are still needed.
                self.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
                self.positions[: self.raw_num_token].copy_(forward_batch.positions)

            # Replay
            seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (
                self.bs - self.raw_bs
            )
            if self.capture_forward_mode.is_target_verify():
                if self.bs != len(forward_batch.extend_seq_lens_cpu):
                    forward_batch.extend_seq_lens_cpu += [0] * (
                        self.bs - len(forward_batch.extend_seq_lens_cpu)
                    )
                if self.bs != len(forward_batch.extend_logprob_start_lens_cpu):
                    forward_batch.extend_logprob_start_lens_cpu += [0] * (
                        self.bs - len(forward_batch.extend_logprob_start_lens_cpu)
                    )

            thread = threading.Thread(target=self._update_inputs, args=(seq_lens,))
            thread.start()
            self.graphs[self.bs].replay()
            thread.join()

            output = self.output_buffers[self.bs]
            if isinstance(output, LogitsProcessorOutput):
                return LogitsProcessorOutput(
                    next_token_logits=output.next_token_logits[: self.raw_num_token],
                    hidden_states=(
                        output.hidden_states[: self.raw_num_token]
                        if output.hidden_states is not None
                        else None
                    ),
                )
            else:
                assert isinstance(output, PPProxyTensors)
                return PPProxyTensors(
                    {k: v[: self.bs] for k, v in output.tensors.items()}
                )

    def prepare_forward_batch(self, bs: int, num_tokens: int) -> ForwardBatch:
        # Graph inputs
        if self.dp_size > 1:
            global_num_tokens = num_tokens * self.dp_size
            dp_padding_mode = DPPaddingMode.MAX_LEN
        else:
            global_num_tokens = num_tokens
            dp_padding_mode = None
        with torch.device(self.model_runner.device):
            input_ids = torch.zeros((num_tokens,), dtype=torch.int64)
            req_pool_indices = torch.zeros((bs,), dtype=torch.int64)
            seq_lens = torch.full((bs,), self.seq_len_fill_value, dtype=torch.int64)
            out_cache_loc = torch.zeros((num_tokens,), dtype=torch.int64)
            positions = torch.zeros((num_tokens,), dtype=torch.int64)
            gathered_buffer = None
            if self.require_gathered_buffer:
                gathered_buffer = torch.zeros(
                    (
                        global_num_tokens,
                        self.model_runner.model_config.hidden_size,
                    ),
                    dtype=self.model_runner.dtype,
                )
                if self.tp_size == self.dp_size:
                    # Here tp_size and dp_size are from augments, so tp_size == dp_size means attention tp size is 1
                    scattered_buffer = torch.zeros(
                        (
                            num_tokens,
                            self.model_runner.model_config.hidden_size,
                        ),
                        dtype=self.model_runner.dtype,
                    )
                else:
                    scattered_buffer = None
            num_token_non_padded = torch.tensor(num_tokens, dtype=torch.int32)
            if self.capture_forward_mode.is_target_verify():
                extend_seq_lens = torch.full(
                    (bs,), self.seq_len_fill_value, dtype=torch.int32
                )
                extend_seq_lens_cpu = extend_seq_lens.cpu().tolist()
                extend_logprob_start_lens_cpu = [0] * bs
            else:
                extend_seq_lens = None
                extend_seq_lens_cpu = None
                extend_logprob_start_lens_cpu = None

        if self.is_encoder_decoder:
            encoder_lens = self.encoder_lens[:bs]
        else:
            encoder_lens = None
        mrope_positions = None

        # pipeline parallelism
        if self.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in self.pp_proxy_tensors.items()}
            )

        if self.require_mlp_tp_gather:
            global_num_tokens = torch.tensor(
                [num_tokens for i in range(self.dp_size)],
                dtype=torch.int64,
                device=input_ids.device,
            )
            global_num_tokens_cpu = [num_tokens for i in range(self.dp_size)]
            global_num_tokens_for_logprob_cpu = [
                num_tokens for i in range(self.dp_size)
            ]
        elif self.require_attn_tp_gather:
            global_num_tokens = torch.tensor(
                [num_tokens], dtype=torch.int64, device=input_ids.device
            )
            global_num_tokens_cpu = [num_tokens]
            global_num_tokens_for_logprob_cpu = [num_tokens]
        else:
            global_num_tokens = None
            gathered_buffer = None

        spec_info = self.get_spec_info(num_tokens)
        if self.capture_hidden_mode != CaptureHiddenMode.FULL:
            self.capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=global_num_tokens,
            gathered_buffer=gathered_buffer,
            scattered_buffer=scattered_buffer,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            num_token_non_padded=num_token_non_padded,
            global_forward_mode=None,
            mm_inputs=[None] * bs,
            lora_ids=[None] * bs,
            global_num_tokens_cpu=global_num_tokens_cpu,
            global_num_tokens_for_logprob_cpu=global_num_tokens_for_logprob_cpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens.clone(),
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=extend_logprob_start_lens_cpu,
            can_run_graph=True,
            dp_padding_mode=dp_padding_mode,
        )
        # todo
        return forward_batch

    def warm_up(self):
        backend = get_compiler_backend()
        if not self.enable_cache:
            self.model_runner.model.compile_forward = torch.compile(
                torch.no_grad()(self.model_runner.model.forward),
                fullgraph=True,
                dynamic=True,
                backend=backend,
            )

        compile_range = (
            tqdm.tqdm(list(reversed(self.compile_bs)))
            if get_tensor_model_parallel_rank() == 0
            else reversed(self.compile_bs)
        )

        def build_method(method_name):
            method_code = f"""
def {method_name}(self, input_ids, positions, forward_batch, **kwargs):
    return self.forward(input_ids, positions, forward_batch, **kwargs)
"""
            exec(method_code)
            return locals()[method_name]

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
            forward_batch.attn_backend.init_forward_metadata(forward_batch)

            if self.enable_cache:
                import torchair

                method_name = f"forward_{bs}bs"
                compile_method_name = f"compile_forward_{bs}bs"
                setattr(
                    self.model_runner.model,
                    method_name,
                    types.MethodType(
                        build_method(method_name), self.model_runner.model
                    ),
                )
                setattr(
                    self.model_runner.model,
                    compile_method_name,
                    torchair.inference.cache_compile(
                        getattr(self.model_runner.model, method_name), backend=backend
                    ),
                )

            # Run and capture
            def run_once():
                # Clean intermediate result cache for DP attention
                forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = (
                    None
                )

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

                compile_forward = (
                    getattr(self.model_runner.model, compile_method_name)
                    if self.enable_cache
                    else self.model_runner.model.compile_forward
                )

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

    def mark_static(
        self, forward_batch: ForwardBatch, pp_proxy_tensors: PPProxyTensors = None
    ):
        def mark_tensor_static(model_input, is_cache=False):
            if model_input is not None:
                if isinstance(model_input, torch.Tensor):
                    torch._dynamo.mark_static(model_input)
                elif is_cache:
                    for buffer_per_layer in model_input:
                        torch._dynamo.mark_static(buffer_per_layer)
                elif isinstance(model_input, PPProxyTensors):
                    for pp_out in model_input.tensors.items():
                        torch._dynamo.mark_static(pp_out)
                elif isinstance(model_input, tuple):
                    for value in model_input:
                        torch._dynamo.mark_static(value)
                else:
                    raise ValueError(
                        f"Unsupported type with mark static: {type(model_input)}"
                    )

        mark_tensor_static(pp_proxy_tensors)
        mark_tensor_static(forward_batch.input_ids)
        mark_tensor_static(forward_batch.positions)
        mark_tensor_static(forward_batch.input_embeds)
        mark_tensor_static(forward_batch.out_cache_loc)
        mark_tensor_static(forward_batch.gathered_buffer)
        mark_tensor_static(forward_batch.input_embeds)
        mark_tensor_static(forward_batch.out_cache_loc)
        mark_tensor_static(forward_batch.next_token_logits_buffer)
        mark_tensor_static(forward_batch.extend_seq_lens)
        if hasattr(forward_batch.attn_backend, "forward_metadata"):
            mark_tensor_static(
                forward_batch.attn_backend.forward_metadata.block_kv_indices
            )
            mark_tensor_static(forward_batch.attn_backend.forward_metadata.mc2_mask)
        if hasattr(forward_batch.spec_info, "hidden_status"):
            mark_tensor_static(forward_batch.spec_info.hidden_status)
        try:
            mark_tensor_static(forward_batch.token_to_kv_pool.k_buffer, is_cache=True)
            mark_tensor_static(forward_batch.token_to_kv_pool.v_buffer, is_cache=True)
        except AttributeError as e:
            mark_tensor_static(forward_batch.token_to_kv_pool.kv_buffer, is_cache=True)

    @contextmanager
    def get_runner_context(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
    ) -> Generator[Callable[[bool, PPProxyTensors | None], Any], Any, None]:
        def runner_fn(pp_proxy_tensors: Optional[PPProxyTensors]):

            return self.replay(forward_batch, skip_attn_backend_init, pp_proxy_tensors)

        # if not skip_attn_backend_init:
        #     forward_batch.attn_backend.init_forward_metadata(forward_batch)
        yield runner_fn

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        return bool(
            self.enable_torch_compile
            and forward_batch.forward_mode.is_cuda_graph()
            and forward_batch.batch_size <= max(self.compile_bs)
            and len(set(forward_batch.global_num_tokens_for_logprob_cpu)) == 1
        )
