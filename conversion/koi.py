from __future__ import annotations

import torch

from .base import ModelBase, TextModel, gguf


@ModelBase.register("KoiForCausalLM")
class KoiModel(TextModel):
    model_arch = gguf.MODEL_ARCH.KOI

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        h = self.hparams
        if h["hidden_act"] != "relu2":
            raise ValueError("Koi requires relu2 activation")
        if self.fuse_qkv:
            raise ValueError("Koi requires separate Q/K/V projections")

        super().set_gguf_parameters()
        w = self.gguf_writer
        w.add_key_length(h["head_dim"])
        w.add_value_length(h["head_dim"])
        w.add_input_layer_count(h["n_input_layers"])
        w.add_recurrent_layer_count(h["n_recurrent_layers"])
        w.add_output_layer_count(h["n_output_layers"])
        w.add_num_loops(h["n_recurrent_loops"])
        patterns = {"linear_attention": "M", "full_attention": "*", "mlp": "-"}
        w.add_layer_pattern("".join(patterns[t] for t in h["layer_types"]))
        w.add_recurrent_kv_cache_mode(h.get("recurrent_kv_cache_mode", "per_loop"))
        w.add_loop_conditioning_hidden_size(h["loop_conditioning_hidden_size"])
        w.add_loop_conditioning_max_loops(h["loop_conditioning_max_loops"])
        w.add_loop_adapter_rank(h["loop_adapter_rank"])
        w.add_loop_input_mixing(h["loop_input_mixing"])
        w.add_ssm_conv_kernel(h["linear_conv_kernel_dim"])
        w.add_ssm_inner_size(h["linear_num_value_heads"] * h["linear_value_head_dim"])
        w.add_ssm_state_size(h["linear_key_head_dim"])
        w.add_ssm_time_step_rank(h["linear_num_value_heads"])
        w.add_ssm_group_count(h["linear_num_key_heads"])
        w.add_ssm_householder_count(h["linear_num_householder"])

    def modify_tensors(self, data_torch, name, bid):
        mapped = self.map_tensor_name(name)
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch.float())
        elif name.endswith(".dt_bias"):
            mapped += ".bias"
        elif not mapped.endswith((".weight", ".bias")):
            mapped += ".weight"
        if name.endswith(".conv1d.weight"):
            data_torch = data_torch.squeeze(1)
        yield mapped, data_torch
