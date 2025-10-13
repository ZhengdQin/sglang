from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_rank,
    get_moe_tensor_parallel_world_size,
    get_moe_ep_group,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.layers.quantization.w8a8_int8 import NPU_W8A8MoEMethod
from sglang.srt.layers.quantization.base_config import QuantizationConfig

from typing import Optional

import torch
import torch_npu
import torch.distributed as dist


class UnquantizedFusedMoEGMMMethod(UnquantizedFusedMoEMethod):

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            expert_tokens: torch.Tensor,
            group_list_type: int,
            **kargs,
    ) -> torch.Tensor:
        mm1_mm3 = torch_npu.npu_grouped_matmul([x], [layer.w13_weight],
            group_list=expert_tokens, group_type=0, group_list_type=group_list_type, split_item=3)[0]
        mm1_mm3 = torch_npu.npu_swiglu(mm1_mm3)
        out = torch_npu.npu_grouped_matmul(
            [mm1_mm3], [layer.w2_weight],
            group_list=expert_tokens, group_type=0, group_list_type=group_list_type, split_item=3)[0]
        return out

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()

        # w13_ptr = layer.w13_weight.data.data_ptr()
        layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 29)  # 29: format nz
        # torch_npu.npu.caching_allocator_delete(w13_ptr)

        # w2_ptr = layer.w2_weight.data.data_ptr()
        layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 29)  # 29: format nz
        # torch_npu.npu.caching_allocator_delete(w2_ptr)


class W8A8Int8MoEGMMMethod(NPU_W8A8MoEMethod):

    def create_weights(self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,):
        super().create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs)
        scale_dtype = torch.float32 if params_dtype == torch.float16 else torch.bfloat16
        smooth_scale_1 = torch.nn.Parameter(torch.ones((num_experts, hidden_size), dtype=scale_dtype), requires_grad=False)
        smooth_scale_2 = torch.nn.Parameter(torch.ones((num_experts, intermediate_size_per_partition),
                                            dtype=scale_dtype), requires_grad=False)
        layer.register_parameter("smooth_scale_1", smooth_scale_1)
        layer.register_parameter("smooth_scale_2", smooth_scale_2)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              expert_tokens: torch.Tensor,
              group_list_type: int,
              pertoken_scale: torch.Tensor = None,
              final_output_dtype: torch.dtype = torch.bfloat16,):
        hidden_size = x.size(-1)
        if pertoken_scale is None:
            x, pertoken_scale = torch_npu.npu_dynamic_quant(x)

        if pertoken_scale.dim() > 1:
            pertoken_scale = pertoken_scale.reshape(-1)
            x = x.view(-1, hidden_size)

        mm1_mm3 = torch_npu.npu_grouped_matmul([x], [layer.w13_weight],
                                                group_list=expert_tokens, split_item=3,
                                                output_dtype=torch.int32, group_type=0,
                                                group_list_type=group_list_type,
                                                tuning_config=[0]
                                                )[0]

        intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            mm1_mm3, weight_scale=layer.w13_weight_scale,
            quant_scale=layer.smooth_scale_2,
            group_index=expert_tokens,
            activate_left=True,
            quant_mode=1,
            activation_scale=pertoken_scale
            )

        out_hidden = torch_npu.npu_grouped_matmul(
            [intermediate_h], [layer.w2_weight], bias=None,
            scale=[layer.w2_weight_scale], per_token_scale=[pertoken_scale],
            group_list=expert_tokens, split_item=3,
            output_dtype=final_output_dtype, group_type=0,
            group_list_type=group_list_type,
            tuning_config=[0]
        )[0]

        return out_hidden

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight.data.transpose(1, 2).contiguous(), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight.data.transpose(1, 2).contiguous(), requires_grad=False
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.data.squeeze(-1).contiguous().to(torch.float32), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.data.squeeze(-1).contiguous(), requires_grad=False
        )
        layer.w13_weight_offset = torch.nn.Parameter(
            layer.w13_weight_offset.data.squeeze(-1).contiguous(), requires_grad=False
        )
        layer.w2_weight_offset = torch.nn.Parameter(
            layer.w2_weight_offset.data.squeeze(-1).contiguous(), requires_grad=False
        )
        layer.smooth_scale_1 = torch.nn.Parameter(
            layer.smooth_scale_1.data.to(torch.float32), requires_grad=False
        )
        moe_ep_size = get_moe_expert_parallel_world_size()
        moe_ep_group = get_moe_ep_group().device_group
        if moe_ep_size > 1:
            all_experts_smooth_scale = layer.smooth_scale_1.data.new_empty(
                layer.smooth_scale_1.data.shape[0] * moe_ep_size, layer.smooth_scale_1.data.shape[1])
            dist.all_gather_into_tensor(all_experts_smooth_scale, layer.smooth_scale_1.data, group=moe_ep_group)
            layer.smooth_scale_1.data = all_experts_smooth_scale


class FusedMoEGMM(FusedMoE):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
    ):
        torch.nn.Module.__init__(self)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.layer_id = layer_id
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_fused_shared_experts = num_fused_shared_experts

        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.moe_ep_rank = get_moe_expert_parallel_rank()
        self.moe_tp_size = get_moe_tensor_parallel_world_size()
        self.moe_tp_rank = get_moe_tensor_parallel_rank()
        assert num_experts % self.moe_ep_size == 0
        self.num_local_experts = num_experts // self.moe_ep_size

        assert intermediate_size % self.moe_tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // self.moe_tp_size


        self.quant_config = quant_config
        self.hidden_size = hidden_size

        if quant_config is None:
            self.quant_method = UnquantizedFusedMoEGMMMethod()
        else:
            self.quant_method = W8A8Int8MoEGMMMethod(self)

        self.use_triton_kernels = False
        self.use_presharded_weights = False

        self.quant_method.create_weights(
            layer=self,
            num_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
            intermediate_size_full=intermediate_size,
            top_k=top_k,
            with_bias=False,
        )


    def forward(self, x: torch.Tensor,
                expert_tokens: torch.Tensor,
                group_list_type: int = 0,
                pertoken_scale: torch.Tensor = None,
                final_output_dtype: torch.dtype = torch.bfloat16,
                ):
        if self.quant_method is None:
            raise RuntimeError("self.quant_method must not be None")

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=x,
            expert_tokens=expert_tokens,
            group_list_type=group_list_type,
            pertoken_scale=pertoken_scale,
            final_output_dtype=final_output_dtype,
        )

        return final_hidden_states
