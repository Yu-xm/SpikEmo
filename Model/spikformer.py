import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__)
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based import neuron
import numpy as np

# tau = 10.0 # beta = 1 - 1/tau
backend = "torch"
detach_reset=True
# common_thr = 1.0
# attn_thr = common_thr / 4

class spiking_self_attention(nn.Module):
    def __init__(self, tau, common_thr, dim, heads=8, qkv_bias=False, qk_scale=0.25):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.heads = heads
        self.qk_scale = qk_scale

        self.q_m = nn.Linear(dim, dim)
        self.q_ln = nn.LayerNorm(dim)
        self.q_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

        self.k_m = nn.Linear(dim, dim)
        self.k_ln = nn.LayerNorm(dim)
        self.k_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

        self.v_m = nn.Linear(dim, dim)
        self.v_ln = nn.LayerNorm(dim)
        self.v_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

        self.attn_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr/2, backend=backend)

        self.last_m = nn.Linear(dim, dim)
        self.last_ln = nn.LayerNorm(dim)
        self.last_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

    def forward(self, x): # B T L D
        # print('<---------------------------------->')
        # print()
        # print('x: ', x.shape)
        x = x.transpose(0, 1) # T B L D
        # print('x: ', x.shape)
        
        T, B, L, D = x.shape # 32, 16, 23, 256
        
        x_for_qkv = x.flatten(0, 1) # TB L D
        # print('x_for_qkv: ', x_for_qkv.shape)
        
        q_m_out = self.q_m(x_for_qkv) # TB L D
        # print('q_m_out: ', q_m_out.shape)
        
        q_m_out = self.q_ln(q_m_out).reshape(T, B, L, D).contiguous()
        # print('q_m_out: ', q_m_out.shape)
        # print()
        # print('<---------------------------------->')
        q_m_out = self.q_lif(q_m_out)
        q = q_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        k_m_out = self.k_m(x_for_qkv)
        k_m_out = self.k_ln(k_m_out).reshape(T, B, L, D).contiguous()
        k_m_out = self.k_lif(k_m_out)
        k = k_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        v_m_out = self.v_m(x_for_qkv)
        v_m_out = self.v_ln(v_m_out).reshape(T, B, L, D).contiguous()
        v_m_out = self.v_lif(v_m_out)
        v = v_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()
        
        # print('<---------------------------------->')

        attn = (q @ k.transpose(-2, -1))
        # print(attn.shape)
        x = (attn @ v) * self.qk_scale  # x_shape: T * B * heads * L * //heads
        # print(x.shape)

        x = x.transpose(2, 3).reshape(T, B, L, D).contiguous()
        # print(x.shape)
        x = self.attn_lif(x)
        
        x = x.flatten(0, 1)
        # print(x.shape)
        x = self.last_m(x)
        x = self.last_ln(x)
        x = self.last_lif(x.reshape(T, B, L, D).contiguous())

        x = x.transpose(0, 1) # B T L D
        return x


class mlp(nn.Module):
    def __init__(self, tau, common_thr, in_features, hidden_features=None, out_features=None, ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        self.lif1 = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)
        self.lif2 = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

    def forward(self, x):
        # B T L D
        x = x.transpose(0, 1) # T B L D
        T, B, L, D = x.shape
        x = x.flatten(0, 1)
        x = self.lif1(self.ln1(self.fc1(x)).reshape(T, B, L, -1).contiguous())
        x = x.flatten(0, 1)
        x = self.lif2(self.ln2(self.fc2(x)).reshape(T, B, L, -1).contiguous())
        x = x.transpose(0, 1) # B T L D
        return x


class block(nn.Module):
    def __init__(self, tau, common_thr, dim, heads=8, qkv_bias=False, qk_scale=0.125):
        super().__init__()
        self.attn = spiking_self_attention(tau=tau, common_thr=common_thr, dim=dim, heads=heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.mlp = mlp(tau=tau, common_thr=common_thr, in_features=dim, hidden_features=dim*4, out_features=dim)

    def forward(self, x):
        # B T L D
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class transform(nn.Module):
    def __init__(self, dim):
        super(transform, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        return x


class Spikformer(nn.Module):
    def __init__(self, depths, tau, common_thr, dim, T, heads, qkv_bias=False, qk_scale=0.125):
        super().__init__()
        self.atan = surrogate.ATan()
        self.T = T
        self.blocks = nn.ModuleList([block(
            tau=tau, common_thr=common_thr, dim=dim, heads=heads, qkv_bias=qkv_bias, qk_scale=qk_scale
        ) for _ in range(depths)])
        self.last_ln = nn.LayerNorm(dim)

        self.transforms = nn.ModuleList([
            transform(dim) for _ in range(depths)
        ])
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


    def forward(self, x):
        
        # print('<------------------------>In spikformer<------------------------>')
        # print('<------------------------>')
        # print("x : ", x.shape)
        # print('<------------------------>')
        # L B D
        # print(x.shape)
        x = x.transpose(0, 1) # B L D
        x = x.repeat(tuple([self.T] + torch.ones(len(x.size()), dtype=int).tolist())) # T B L D
        x = x.transpose(0, 1) # B T L D
        x = self.atan(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x) # B T L D

        # B T L D
        x = self.last_ln(x)
        # B T L D
        x = x.mean(1)
        # x: B L D
        
        x = x.transpose(0, 1) # x: L B D
        
#         print('<------------------------>')
#         print("x : ", x.shape)
#         print('<------------------------>')
#         print('<------------------------>In spikformer<------------------------>')
        return x