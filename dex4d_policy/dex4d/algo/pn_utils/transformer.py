# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import math
import numpy as np

import torch
import torch.nn as nn

# a BERT-style transformer block
class Transformer_Block(nn.Module):
    def __init__(self, latent_dim, num_head, dropout_rate) -> None:
        super().__init__()
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.ln_1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout_rate, batch_first=True)
        self.ln_2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x):
        x = self.ln_1(x)
        x = x + self.attn(x, x, x, need_weights=False)[0]
        x = self.ln_2(x)
        x = x + self.mlp(x)
        
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, output_token_idx, latent_dim=128, num_head=4, num_layer=4, dropout_rate=0.1, use_input_layer=True, use_positional_encoding=True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_token_idx = output_token_idx
        self.latent_dim = latent_dim
        self.num_head = num_head
        self.num_layer = num_layer
        self.use_input_layer = use_input_layer
        self.use_positional_encoding = use_positional_encoding
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )
        self.attention_blocks = nn.Sequential(
            *[Transformer_Block(latent_dim, num_head, dropout_rate) for _ in range(num_layer)],
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, output_dim),
        )
        self.positional_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def _add_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positional_encoding = self._sinusoidal_positional_encoding(seq_len, x.device, x.dtype)
        return self.positional_dropout(x + positional_encoding)

    def _sinusoidal_positional_encoding(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        base_dtype = torch.float32
        position = torch.arange(seq_len, device=device, dtype=base_dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.latent_dim, 2, device=device, dtype=base_dtype)
            * (-math.log(10000.0) / self.latent_dim)
        )
        pe = torch.zeros(seq_len, self.latent_dim, device=device, dtype=base_dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).to(dtype=dtype)
    
    def forward(self, x):
        # x.shape: (B, T, input_dim)
        if self.use_input_layer:
            x = self.input_layer(x)
        if self.use_positional_encoding:
            x = self._add_positional_encoding(x)
        x = self.attention_blocks(x)

        latent_tokens = x.clone()
        x = x[:, self.output_token_idx, :]
        x = self.output_layer(x)

        return x, latent_tokens
    
if __name__ == "__main__":
    # simple test
    batch_size = 2
    seq_len = 16 # 16 tokens
    input_dim = 3 # token dimension
    output_dim = 22 # output dimension
    latent_dim = 128 # latent_dim
    num_head = 4
    num_layer = 4
    dropout_rate = 0.1

    model = Transformer(input_dim, output_dim, latent_dim, num_head, num_layer, dropout_rate)
    print(model)
    x = torch.randn(batch_size, seq_len, input_dim)
    y, latent = model(x, output_token_idx=0)
    print(y.shape, latent.shape)  # should be (2, 22) and (2, seq_len, latent_dim)