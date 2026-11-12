#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################

import argparse
import math
from pathlib import Path

import onnx
import torch
from torch import nn
from transformers import AutoModelForCausalLM


class TinyQwenDecoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers
        self.norm = model.model.norm
        self.lm_head = model.lm_head

        config = model.config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        inv_freq = 1.0 / (config.rope_theta**(torch.arange(
            0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def apply_rotary(self, x: torch.Tensor, cos: torch.Tensor,
                     sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (self.rotate_half(x) * sin)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                position_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        freqs = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq
        embeddings = torch.cat((freqs, freqs), dim=-1)
        cos = embeddings.cos().unsqueeze(1)
        sin = embeddings.sin().unsqueeze(1)

        for layer in self.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            query = layer.self_attn.q_proj(hidden_states)
            key = layer.self_attn.k_proj(hidden_states)
            value = layer.self_attn.v_proj(hidden_states)

            query = query.reshape(1, -1, self.num_heads,
                                  self.head_dim).transpose(1, 2)
            key = key.reshape(1, -1, self.num_key_value_heads,
                              self.head_dim).transpose(1, 2)
            value = value.reshape(1, -1, self.num_key_value_heads,
                                  self.head_dim).transpose(1, 2)

            query = self.apply_rotary(query, cos, sin)
            key = self.apply_rotary(key, cos, sin)

            groups = self.num_heads // self.num_key_value_heads
            heads = []
            for head in range(self.num_heads):
                kv_head = head // groups
                head_query = query[:, head:head + 1]
                head_key = key[:, kv_head:kv_head + 1]
                head_value = value[:, kv_head:kv_head + 1]
                weights = torch.matmul(head_query, head_key.transpose(
                    -2, -1)) * self.scale
                weights = torch.softmax(weights + attention_mask, dim=-1)
                heads.append(torch.matmul(weights, head_value))
            attention = torch.cat(heads, dim=1)
            attention = attention.transpose(1,
                                            2).reshape(1, -1, self.hidden_size)
            hidden_states = residual + layer.self_attn.o_proj(attention)

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp.down_proj(
                layer.mlp.act_fn(layer.mlp.gate_proj(hidden_states)) *
                layer.mlp.up_proj(hidden_states))
            hidden_states = residual + hidden_states

        return self.lm_head(self.norm(hidden_states))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=
        "Download and export a tiny cache-free Qwen2 model for dynamic execution."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/qwen"),
        help="Directory in which to export model.onnx.",
    )
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        "onnx-internal-testing/tiny-random-Qwen2ForCausalLM")
    decoder = TinyQwenDecoder(model).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.onnx"
    sequence_length = 4
    torch.onnx.export(
        decoder,
        (
            torch.zeros((1, sequence_length), dtype=torch.int64),
            torch.zeros(
                (1, 1, sequence_length, sequence_length), dtype=torch.float32),
            torch.arange(sequence_length, dtype=torch.int64).unsqueeze(0),
        ),
        model_path,
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {
                1: "sequence_length"
            },
            "attention_mask": {
                2: "sequence_length",
                3: "sequence_length"
            },
            "position_ids": {
                1: "sequence_length"
            },
            "logits": {
                1: "sequence_length"
            },
        },
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(model_path)
    print(model_path)


if __name__ == "__main__":
    main()
