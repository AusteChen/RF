from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
# from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torch.distributed as dist

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
# flex_attention = torch.compile(
#     flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


# ========================================
# 新增：课程式 Mask 调度器
# ========================================

class CausalTransitionScheduler:
    """
    课程式 Mask 调度器：从全双向逐步过渡到全因果

    训练初期：允许模型看到更多未来帧（Bidirectional）
    训练后期：逐步减少直到完全因果（Causal）

    这是一种课程式学习策略，让模型从易到难学习时序建模。
    """

    def __init__(
        self,
        total_steps: int,
        num_frames: int,
        start_step: int = 0,
        decay_mode: str = 'linear'
    ):
        """
        Args:
            total_steps (int): 总训练步数
            num_frames (int): 视频的总帧数
            start_step (int): 从多少步开始衰减（前几步保持全双向作为Warmup）
            decay_mode (str): 'linear' 或 'cosine'
        """
        self.total_steps = total_steps
        self.num_frames = num_frames
        self.start_step = start_step
        self.decay_mode = decay_mode

    def get_lookahead_window(self, current_step: int) -> int:
        """
        计算当前允许看未来的帧数 k
        k = num_frames -> 全双向 (Bidirectional)
        k = 0 -> 全因果 (Causal)
        """
        if current_step < self.start_step:
            return self.num_frames  # Warmup期保持全双向

        denominator = self.total_steps - self.start_step
        if denominator <= 0:
            progress = 1.0
        else:
            progress = (current_step - self.start_step) / denominator
        progress = max(0.0, min(1.0, progress))

        if self.decay_mode == 'linear':
            k = int(self.num_frames * (1 - progress))
        elif self.decay_mode == 'cosine':
            k = int(self.num_frames * 0.5 * (1 + math.cos(progress * math.pi)))
        else:
            raise ValueError(f"Unknown decay mode: {self.decay_mode}")

        return max(0, k)


# ========================================
# 新增：Gumbel-Softmax 门控路由器
# ========================================

class LayerRouter(nn.Module):
    """
    Gumbel-Softmax 门控路由器 (单层独立版)

    推荐用法：在 CausalAttention 或 TransformerBlock 的 __init__ 中实例化本类。
    每一层独立维护自己的可学习路由概率。
    """

    def __init__(self, init_global_prob: float = 0.8):
        super().__init__()
        # 使用两个标量参数代表 [local_logit, global_logit]
        # 初始时给予偏向全局层的概率
        init_local = math.log(1.0 - init_global_prob + 1e-8)
        init_global = math.log(init_global_prob + 1e-8)

        # 将其注册为当前层的可学习参数
        self.routing_logits = nn.Parameter(torch.tensor([init_local, init_global]))

    def reset_parameters(self):
        """重置路由 logits 为初始值，确保与 PyTorch FSDP 兼容"""
        init_local = math.log(1.0 - 0.8 + 1e-8)  # 默认 0.8 全局概率
        init_global = math.log(0.8 + 1e-8)
        with torch.no_grad():
            self.routing_logits.copy_(torch.tensor([init_local, init_global]))

    def forward(
        self,
        temperature: float = 1.0,
        step: int = 0,
        total_steps: int = 100000,
        warmup_steps: int = 5000,
    ) -> torch.Tensor:
        """
        前向传播：直接输出这一层是否为全局层的确切决策（标量）。

        Args:
            temperature: Gumbel-Softmax 退火温度。
            step: 当前训练步数（用于自动温度退火）。
            total_steps: 总训练步数（用于温度退火进度计算）。
            warmup_steps: 预热步数（前 warmup_steps 步内温度保持初始值 tau_start）。
        Returns:
            is_global: 标量张量 (1.0 代表全局层，0.0 代表局部层)
        """
        # 自动温度退火：warmup 结束后从 tau_start 线性降到 tau_end
        if step < warmup_steps:
            tau = temperature  # 使用传入的初始温度
        else:
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            tau_start, tau_end = 5.0, 0.5
            tau = tau_start + (tau_end - tau_start) * progress

        if self.training:
            # 官方底层的 Gumbel Softmax。自带噪声与 STE (hard=True)
            # 输出类似 [0.0, 1.0] 的 one-hot 向量
            routing_weights = F.gumbel_softmax(
                self.routing_logits,
                tau=tau,
                hard=True,
                dim=0
            )
        else:
            # 推理阶段：直接选 logits 最大的那个作为 one-hot 向量
            routing_weights = F.one_hot(torch.argmax(self.routing_logits, dim=0), num_classes=2).float()

        # 索引 1 代表全局层
        is_global = routing_weights[1]

        return is_global

# ========================================
# 新增：双通道历史信息提取头（无状态核心）
# ========================================

class DualChannelExtractionHead(nn.Module):
    """
    双通道历史信息提取头 (流式生成专用)

    通道一：基于余弦相似度检测场景切换，动态更新 1-2 帧局部锚点。
    通道二：将滑出窗口的废弃特征 (evicted) 进行空间池化降维，拼接到定长队列中。

    训练支持：主训练链路走 kv-cache 展开时，通道二会通过 eviction 事件接收梯度。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        memory_size: int = 100,       # 固定长度 FIFO（按 frame token 计）
        anchor_frames: int = 2,       # 动态 sink 保留的帧 token 数
        scene_change_tau: float = 0.6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.memory_size = memory_size
        self.anchor_frames = anchor_frames
        self.scene_change_tau = scene_change_tau

        # 动态状态（不使用 Parameter，因为形状受 Batch 影响且随时间增长）
        # 在 forward 中动态初始化
        self.memory_k = None
        self.memory_v = None
        self.sink_k = None
        self.sink_v = None
        self.last_frame_key = None

        # 这几组轻量投影让双路头具备可训练能力。
        self.sink_key_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.sink_value_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.memory_key_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.memory_value_proj = nn.Linear(head_dim, head_dim, bias=False)

    def reset_parameters(self):
        """重置动态状态，确保与 PyTorch FSDP 兼容"""
        self.memory_k = None
        self.memory_v = None
        self.sink_k = None
        self.sink_v = None
        self.last_frame_key = None

    def _pool_frame_tokens(self, x: torch.Tensor, spatial_shape: tuple) -> torch.Tensor:
        """
        将单帧的空间 token 压成 1 个 frame token，输出 [B, T, N, D]。
        这是功能 3 的固定长度 FIFO 版本：先按 frame 压缩，再做 FIFO。
        """
        b, s, n, d = x.shape
        h, w = spatial_shape
        tokens_per_frame = max(1, h * w)

        if s % tokens_per_frame != 0:
            target_t = max(1, round(s / tokens_per_frame))
            pooled = F.adaptive_avg_pool1d(
                x.permute(0, 2, 3, 1).reshape(b * n, d, s),
                target_t,
            )
            return pooled.reshape(b, n, d, target_t).permute(0, 3, 1, 2)

        t = s // tokens_per_frame
        return x.reshape(b, t, tokens_per_frame, n, d).mean(dim=2)

    def _apply_proj(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        b, t, n, d = x.shape
        return proj(x.reshape(b * t * n, d)).reshape(b, t, n, d)

    def _compute_sink(
        self,
        current_k: torch.Tensor,
        current_v: torch.Tensor,
        last_frame_key: torch.Tensor,
    ) -> tuple:
        """
        通道一：动态 attention sink。
        所有层都能访问这一路；当检测到场景变化时，用当前块的前若干 frame token 刷新 sink。
        """
        current_frame_repr = current_k.mean(dim=2).mean(dim=1)

        if last_frame_key is not None:
            current_norm = F.normalize(current_frame_repr, dim=-1)
            last_norm = F.normalize(last_frame_key, dim=-1)
            similarity = (current_norm * last_norm).sum(dim=-1).mean().item()
            is_scene_change = similarity < self.scene_change_tau
        else:
            is_scene_change = True

        sink_source_k = current_k[:, :self.anchor_frames]
        sink_source_v = current_v[:, :self.anchor_frames]
        new_sink_k = self._apply_proj(sink_source_k, self.sink_key_proj)
        new_sink_v = self._apply_proj(sink_source_v, self.sink_value_proj)

        return new_sink_k, new_sink_v, current_frame_repr.detach(), is_scene_change

    def _compute_fifo_memory(
        self,
        evicted_k: torch.Tensor,
        evicted_v: torch.Tensor,
        memory_k: torch.Tensor,
        memory_v: torch.Tensor,
        spatial_shape: tuple,
    ) -> tuple:
        """
        通道二：固定长度 FIFO 历史压缩。
        先按 frame 压缩成少量 token，再追加到固定长度 memory queue。
        """
        pooled_k = self._pool_frame_tokens(evicted_k, spatial_shape)
        pooled_v = self._pool_frame_tokens(evicted_v, spatial_shape)
        compressed_k = self._apply_proj(pooled_k, self.memory_key_proj)
        compressed_v = self._apply_proj(pooled_v, self.memory_value_proj)

        if memory_k is None:
            new_memory_k = compressed_k
            new_memory_v = compressed_v
        else:
            new_memory_k = torch.cat([memory_k, compressed_k], dim=1)
            new_memory_v = torch.cat([memory_v, compressed_v], dim=1)

        if new_memory_k.shape[1] > self.memory_size:
            new_memory_k = new_memory_k[:, -self.memory_size:, :, :]
            new_memory_v = new_memory_v[:, -self.memory_size:, :, :]

        return new_memory_k, new_memory_v, new_memory_k, new_memory_v

    def forward(
        self,
        current_k: torch.Tensor,
        current_v: torch.Tensor,
        evicted_k: torch.Tensor,
        evicted_v: torch.Tensor,
        is_global_layer: bool,
        spatial_shape: tuple,
        last_frame_key: torch.Tensor = None,
        sink_k: torch.Tensor = None,
        sink_v: torch.Tensor = None,
        memory_k: torch.Tensor = None,
        memory_v: torch.Tensor = None,
        update_state: bool = True,
    ) -> dict:
        output = {
            'sink_k': None,
            'sink_v': None,
            'memory_k': None,
            'memory_v': None,
            'is_scene_change': False,
            'last_frame_key_out': last_frame_key,
            'sink_k_out': sink_k,
            'sink_v_out': sink_v,
            'memory_k_out': memory_k,
            'memory_v_out': memory_v,
        }

        current_frame_k = self._pool_frame_tokens(current_k, spatial_shape)
        current_frame_v = self._pool_frame_tokens(current_v, spatial_shape)

        current_sink_k, current_sink_v, last_frame_key_out, is_scene_change = self._compute_sink(
            current_frame_k,
            current_frame_v,
            last_frame_key,
        )
        output['is_scene_change'] = is_scene_change

        if update_state:
            if is_scene_change or self.sink_k is None:
                self.sink_k = current_sink_k.detach()
                self.sink_v = current_sink_v.detach()
            output['sink_k'] = current_sink_k if (is_scene_change or sink_k is None) else self.sink_k
            output['sink_v'] = current_sink_v if (is_scene_change or sink_v is None) else self.sink_v
            self.last_frame_key = last_frame_key_out
        else:
            sink_k_out = current_sink_k if (is_scene_change or sink_k is None) else sink_k
            sink_v_out = current_sink_v if (is_scene_change or sink_v is None) else sink_v
            output['sink_k'] = sink_k_out
            output['sink_v'] = sink_v_out
            output['sink_k_out'] = sink_k_out.detach() if sink_k_out is not None else None
            output['sink_v_out'] = sink_v_out.detach() if sink_v_out is not None else None
            output['last_frame_key_out'] = last_frame_key_out

        if evicted_k is not None and evicted_k.shape[1] > 0:
            new_memory_k, new_memory_v, memory_k_out, memory_v_out = self._compute_fifo_memory(
                evicted_k,
                evicted_v,
                memory_k if not update_state else self.memory_k,
                memory_v if not update_state else self.memory_v,
                spatial_shape,
            )

            if update_state:
                self.memory_k = memory_k_out.detach()
                self.memory_v = memory_v_out.detach()
                output['memory_k'] = new_memory_k
                output['memory_v'] = new_memory_v
            else:
                output['memory_k'] = new_memory_k
                output['memory_v'] = new_memory_v
                output['memory_k_out'] = memory_k_out.detach()
                output['memory_v_out'] = memory_v_out.detach()
        elif update_state:
            output['memory_k'] = self.memory_k
            output['memory_v'] = self.memory_v
        else:
            output['memory_k'] = memory_k
            output['memory_v'] = memory_v

        return output

    def reset(self):
        """重置动态状态"""
        self.memory_k = None
        self.memory_v = None
        self.sink_k = None
        self.sink_v = None
        self.last_frame_key = None


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=1,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.frame_length = 1560
        self.max_attention_size = 21 * self.frame_length
        self.block_length = 3 * self.frame_length
        self.sink_size = max(1, sink_size)

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        updating_cache=False,
        sink_k=None,    # [B, S_sink, N, D]  动态 attention sink，所有层可见
        sink_v=None,
        memory_k=None,  # [B, S_mem, N, D]   FIFO 压缩历史，仅全局层可见
        memory_v=None,
        global_gate=None,  # 标量 gate，1 表示全局层，0 表示局部层
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
            sink_k/sink_v: 动态 attention sink，局部层和全局层都可访问。
            memory_k/memory_v: FIFO 压缩历史，仅全局层访问。
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d) # [B, L, 12, 128]
            k = self.norm_k(self.k(x)).view(b, s, n, d) # [B, L, 12, 128]
            v = self.v(x).view(b, s, n, d)              # [B, L, 12, 128]
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)   # [B, L, 12, 128]
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)   # [B, L, 12, 128]
            
            grid_sizes_one_block = grid_sizes.clone()
            grid_sizes_one_block[:,0] = 3

            # only caching the first block
            cache_end = cache_start + self.block_length
            num_new_tokens = cache_end - kv_cache["global_end_index"].item()
            kv_cache_size = kv_cache["k"].shape[1]

            sink_tokens = self.sink_size * self.block_length

            if (num_new_tokens > 0) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

                # 收集被驱逐的 tokens，供 DualChannelExtractionHead 的通道二使用
                evicted_tokens_k = kv_cache["k"][:, sink_tokens:sink_tokens + num_evicted_tokens].clone()
                evicted_tokens_v = kv_cache["v"][:, sink_tokens:sink_tokens + num_evicted_tokens].clone()

                if "evicted_k" in kv_cache and "evicted_v" in kv_cache:
                    evicted_k_data = kv_cache["evicted_k"]
                    evicted_v_data = kv_cache["evicted_v"]
                    new_evicted_k = torch.cat([evicted_k_data, evicted_tokens_k], dim=1)
                    new_evicted_v = torch.cat([evicted_v_data, evicted_tokens_v], dim=1)
                    # 限制 evicted queue 的大小（最多保留最近 2 * memory_size 个 token）
                    max_evicted_size = 2 * 1560 * 24
                    if new_evicted_k.shape[1] > max_evicted_size:
                        new_evicted_k = new_evicted_k[:, -max_evicted_size:, :, :]
                        new_evicted_v = new_evicted_v[:, -max_evicted_size:, :, :]
                    kv_cache["evicted_k"] = new_evicted_k
                    kv_cache["evicted_v"] = new_evicted_v

                local_end_index = kv_cache["local_end_index"].item() + cache_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - self.block_length
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key[:, :self.block_length]
                kv_cache["v"][:, local_start_index:local_end_index] = v[:, :self.block_length]
            else:
                local_end_index = kv_cache["local_end_index"].item() + cache_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - self.block_length
                if local_start_index == 0: # first block is not roped in the cache
                    kv_cache["k"][:, local_start_index:local_end_index] = k[:, :self.block_length]
                else:
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_key[:, :self.block_length]

                kv_cache["v"][:, local_start_index:local_end_index] = v[:, :self.block_length]

            if num_new_tokens > 0: # prevent updating when caching clean frame
                kv_cache["global_end_index"].fill_(cache_end)
                kv_cache["local_end_index"].fill_(local_end_index)

            if local_start_index == 0:
                # no kv attn with cache
                x = attention(
                    roped_query,
                    roped_key,
                    v)
            else:
                if updating_cache: # updating working cache with clean frame
                    extract_cache_end = local_end_index
                    extract_cache_start = max(0, local_end_index-self.max_attention_size)
                    working_cache_key = kv_cache["k"][:, extract_cache_start:extract_cache_end].clone()
                    working_cache_v = kv_cache["v"][:, extract_cache_start:extract_cache_end]

                    if extract_cache_start == 0: # rope the global first block in working cache
                        working_cache_key[:,:self.block_length] = causal_rope_apply(
                            working_cache_key[:,:self.block_length], grid_sizes_one_block, freqs, start_frame=0).type_as(v)

                    x = attention(
                        roped_query,
                        working_cache_key,
                        working_cache_v
                    )

                else:
                    # 1. extract working cache
                    extract_cache_end = local_start_index
                    if global_gate is None:
                        query_length = roped_query.shape[1]
                        working_cache_max_length = self.max_attention_size - query_length - self.block_length
                    elif self.local_attn_size != -1:
                        working_cache_max_length = self.local_attn_size * self.frame_length
                    else:
                        # 当启用 router 时，默认把主 working cache 控制在最近 1 个 block；
                        # 更远历史统一交给 FIFO 压缩记忆。
                        working_cache_max_length = self.block_length
                    extract_cache_start = max(
                        self.block_length,
                        local_start_index - working_cache_max_length,
                    )
                    working_cache_key = kv_cache["k"][:, extract_cache_start:extract_cache_end]
                    working_cache_v = kv_cache["v"][:, extract_cache_start:extract_cache_end]

                    # 2. 保留原始 anchor 作为 route A 的兜底。
                    working_cache_frame_length = working_cache_key.shape[1] // self.frame_length
                    rope_start_frame = current_start_frame - working_cache_frame_length - 3
                    anchor_cache_key = causal_rope_apply(
                        kv_cache["k"][:, :self.block_length],
                        grid_sizes_one_block,
                        freqs,
                        start_frame=rope_start_frame,
                    ).type_as(v)
                    anchor_cache_v = kv_cache["v"][:, :self.block_length]

                    local_key_parts = []
                    local_v_parts = []
                    if sink_k is not None and sink_v is not None:
                        local_key_parts.append(sink_k)
                        local_v_parts.append(sink_v)
                    else:
                        local_key_parts.append(anchor_cache_key)
                        local_v_parts.append(anchor_cache_v)
                    local_key_parts += [working_cache_key, roped_key]
                    local_v_parts += [working_cache_v, v]

                    local_out = attention(
                        roped_query,
                        torch.cat(local_key_parts, dim=1),
                        torch.cat(local_v_parts, dim=1),
                    )

                    if memory_k is not None and memory_v is not None:
                        global_key_parts = local_key_parts[:1] + [memory_k] + local_key_parts[1:]
                        global_v_parts = local_v_parts[:1] + [memory_v] + local_v_parts[1:]
                        global_out = attention(
                            roped_query,
                            torch.cat(global_key_parts, dim=1),
                            torch.cat(global_v_parts, dim=1),
                        )

                        if global_gate is None:
                            x = global_out
                        elif self.training:
                            gate = global_gate.to(dtype=local_out.dtype, device=local_out.device).view(1, 1, 1, 1)
                            x = local_out * (1 - gate) + global_out * gate
                        else:
                            x = global_out if global_gate.detach().item() >= 0.5 else local_out
                    else:
                        x = local_out
                 

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 # 新增参数
                 is_global_layer=None,
                 use_dual_channel_head=False,
                 memory_size=100,
                 anchor_frames=2,
                 scene_change_tau=0.6,
                 ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # 新增参数
        self.is_global_layer = is_global_layer
        self.use_dual_channel_head = use_dual_channel_head

        # 双通道历史信息提取头
        if use_dual_channel_head:
            head_dim = dim // num_heads
            self.dual_channel_head = DualChannelExtractionHead(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                memory_size=memory_size,
                anchor_frames=anchor_frames,
                scene_change_tau=scene_change_tau,
            )
        else:
            self.dual_channel_head = None

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def reset_parameters(self):
        """重置所有可学习参数，确保与 PyTorch FSDP 兼容"""
        # 重置 modulation 参数
        nn.init.normal_(self.modulation, std=1.0 / self.dim**0.5)
        # 递归重置所有子模块
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        updating_cache=False,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        layer_router=None,   # 可选的 LayerRouter 实例（由 CausalWanModel 的 block loop 传入）
        router_step=0,
        router_total_steps=100000,
        router_warmup_steps=5000,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            layer_router: 可选 LayerRouter；若传入则用动态路由决策覆盖 is_global_layer。
            router_step: 当前训练步数（供路由器温度退火）。
            router_total_steps: 总训练步数。
            router_warmup_steps: 路由器温度退火预热步数。
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # -------------------------------------------------------
        # 1. 决定本层是否为全局层
        #    优先级：layer_router（动态）> is_global_layer（硬编码）> 默认全局
        # -------------------------------------------------------
        if layer_router is not None:
            global_gate = layer_router(
                step=router_step,
                total_steps=router_total_steps,
                warmup_steps=router_warmup_steps,
            )
        elif self.dual_channel_head is not None and self.is_global_layer is not None:
            global_gate = x.new_tensor(float(self.is_global_layer))
        else:
            global_gate = None
        is_global_bool = True if global_gate is None else (global_gate.detach().item() > 0.5)

        # -------------------------------------------------------
        # 2. Dual-channel history extraction
        #    - Inference path  (kv_cache not None, not updating_cache): eviction-based
        #    - Training path    (kv_cache is None): full-sequence auxiliary path
        #      The head still participates in the forward pass so its parameters
        #      receive gradients; output is discarded (no KV cache to extend).
        # -------------------------------------------------------
        sink_k, sink_v = None, None
        memory_k, memory_v = None, None
        if self.dual_channel_head is not None:
            if kv_cache is not None and not updating_cache:
                # --- Inference path (unchanged) ---
                b, s_total = x.shape[:2]
                n, d = self.num_heads, self.dim // self.num_heads
                normed_x = (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                            * (1 + e[1]) + e[0]).flatten(1, 2)
                curr_k = self.self_attn.norm_k(self.self_attn.k(normed_x)).view(b, s_total, n, d)
                curr_v = self.self_attn.v(normed_x).view(b, s_total, n, d)

                evicted_k = kv_cache.get("evicted_k", None) if isinstance(kv_cache, dict) else None
                evicted_v = kv_cache.get("evicted_v", None) if isinstance(kv_cache, dict) else None

                H = grid_sizes[0][1].item()
                W = grid_sizes[0][2].item()
                head_out = self.dual_channel_head(
                    current_k=curr_k,
                    current_v=curr_v,
                    evicted_k=evicted_k,
                    evicted_v=evicted_v,
                    is_global_layer=is_global_bool,
                    spatial_shape=(H, W),
                    update_state=True,  # 推理路径：更新内部状态
                )
                sink_k = head_out["sink_k"]
                sink_v = head_out["sink_v"]
                memory_k = head_out["memory_k"]
                memory_v = head_out["memory_v"]
                # evicted queue 只应被消费一次；否则同一批历史会被重复压入 FIFO memory。
                if isinstance(kv_cache, dict) and evicted_k is not None and evicted_k.shape[1] > 0:
                    kv_cache["evicted_k"] = evicted_k[:, :0]
                    kv_cache["evicted_v"] = evicted_v[:, :0]

            else:
                # score distillation 主训练路径走 kv-cache 展开，双路头会在那条路径里收到梯度。
                # 这里保留 no-op，避免普通全序列训练路径引入额外的 mask 复杂度。
                pass

        # -------------------------------------------------------
        # 3. Self-attention（传入 sink/memory + router gate）
        # -------------------------------------------------------
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start,
            updating_cache=updating_cache,
            sink_k=sink_k,
            sink_v=sink_v,
            memory_k=memory_k,
            memory_v=memory_v,
            global_gate=global_gate,
        )

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def reset_parameters(self):
        """重置所有可学习参数，确保与 PyTorch FSDP 兼容"""
        # 重置线性层
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
        # 重置 modulation 参数
        nn.init.normal_(self.modulation, std=1.0 / self.dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 # 新增参数
                 use_dual_channel_head=False,
                 use_gumbel_router=False,
                 memory_size=100,
                 anchor_frames=2,
                 scene_change_tau=0.6,
                 global_layer_indices=None,  # None表示所有层为全局层
                 use_curriculum_mask=False,
                 curriculum_total_steps=100000,
                 curriculum_start_step=5000,
                 curriculum_decay_mode='linear',
                 ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            use_dual_channel_head (`bool`, *optional*, defaults to False):
                是否使用双通道历史信息提取头
            use_gumbel_router (`bool`, *optional*, defaults to False):
                是否使用 Gumbel-Softmax 门控路由器
            memory_size (`int`, *optional*, defaults to 100):
                FIFO 压缩历史队列的最大长度（按 frame token 计）
            anchor_frames (`int`, *optional*, defaults to 2):
                动态 attention sink 保留的 frame token 数
            scene_change_tau (`float`, *optional*, defaults to 0.6):
                动态 sink 的场景切换阈值
            global_layer_indices (`list`, *optional*, defaults to None):
                特定的全局层索引列表（None表示所有层为全局层）
            use_curriculum_mask (`bool`, *optional*, defaults to False):
                是否使用课程式 mask（从全双向逐步过渡到全因果）
            curriculum_total_steps (`int`, *optional*, defaults to 100000):
                课程式学习的总步数
            curriculum_start_step (`int`, *optional*, defaults to 5000):
                课程式学习开始衰减的步数
            curriculum_decay_mode (`str`, *optional*, defaults to 'linear'):
                课程式学习衰减模式：'linear' 或 'cosine'
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # 新增参数
        self.use_dual_channel_head = use_dual_channel_head
        self.use_gumbel_router = use_gumbel_router

        # 处理全局层索引
        # 逻辑：
        # 1. use_gumbel_router=true → 路由器自行判断，不预先设置 global_layer_indices
        # 2. use_gumbel_router=false + global_layer_indices=[...] → 按 global_layer_indices 硬编码
        # 3. use_gumbel_router=false + global_layer_indices=None → 所有层为全局层
        if use_gumbel_router:
            # 路由器模式：不预设 global_layer_indices，让路由器自行判断
            self.global_layer_indices = None
        elif global_layer_indices is not None:
            # 硬编码模式：按 global_layer_indices 设置
            self.global_layer_indices = sorted(list(set(global_layer_indices)))
        else:
            # 默认模式：所有层都为全局层
            self.global_layer_indices = list(range(num_layers))

        # 课程式 Mask 调度器
        self.use_curriculum_mask = use_curriculum_mask
        self.curriculum_scheduler = None
        if use_curriculum_mask:
            self.curriculum_total_steps = curriculum_total_steps
            self.curriculum_start_step = curriculum_start_step
            self.curriculum_decay_mode = curriculum_decay_mode
            self.curriculum_num_frames = None  # 在 forward 时设置

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'

        # 如果使用 Gumbel Router 或双通道头，需要为每个 block 添加这些模块
        self.blocks = nn.ModuleList([])
        self.layer_routers = nn.ModuleList([]) if use_gumbel_router else None

        for layer_idx in range(num_layers):
            # 确定该层是否为全局层：
            # - use_gumbel_router=true: 由路由器自行判断，is_global_layer=None
            # - use_gumbel_router=false: 使用预设的 global_layer_indices
            if use_gumbel_router:
                is_global_layer = None  # 路由器自行判断
            else:
                is_global_layer = layer_idx in self.global_layer_indices

            block = CausalWanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                local_attn_size, sink_size, qk_norm, cross_attn_norm, eps,
                is_global_layer=is_global_layer,
                use_dual_channel_head=use_dual_channel_head,
                memory_size=memory_size,
                anchor_frames=anchor_frames,
                scene_change_tau=scene_change_tau,
            )
            self.blocks.append(block)

            # 为每一层添加 Gumbel Router
            if use_gumbel_router:
                router = LayerRouter(init_global_prob=0.8)
                self.layer_routers.append(router)

        # 课程式 mask 调度器初始化（会在 forward 时根据帧数设置）
        if use_curriculum_mask:
            self.curriculum_scheduler = None  # 会在 forward 时初始化

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False
        # 当前训练步数（供 LayerRouter 的温度退火使用）
        self.current_step = 0
        self.router_total_steps = curriculum_total_steps if use_curriculum_mask else 100000
        self.router_warmup_steps = curriculum_start_step if use_curriculum_mask else 5000

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def step(self):
        """递增训练步数，供 LayerRouter 的温度退火使用。"""
        self.current_step += 1

    def get_step(self):
        """获取当前训练步数。"""
        return self.current_step

    def reset_stream_state(self):
        """重置流式推理相关的动态状态（双路头 memory/sink 等）。"""
        for block in self.blocks:
            if getattr(block, "dual_channel_head", None) is not None:
                block.dual_channel_head.reset()

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ):
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ):
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ):
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_with_lookahead(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block: int = 1,
        lookahead_blocks: int = 0, local_attn_size: int = -1
    ):
        """
        课程式 lookahead mask：在标准块因果 mask 基础上，允许每个 query block
        额外向前看 lookahead_blocks 个块的内容。

        lookahead_blocks == 0  → 等同于纯因果 mask（同 _prepare_blockwise_causal_attn_mask）
        lookahead_blocks >= num_blocks → 完全双向 mask

        实现思路：
          对于 block i（其 query token 范围是 [i*B, (i+1)*B)），
          标准因果 mask 允许它 attend 到 kv_idx < (i+1)*B 的所有 token。
          加了 lookahead_blocks=k 之后，允许它额外 attend 到
          kv_idx < (i+1+k)*B 的 token（但不超过 total_length）。
        """
        total_length = num_frames * frame_seqlen
        block_len = frame_seqlen * num_frame_per_block
        num_blocks = math.ceil(total_length / block_len)

        padded_length = math.ceil(total_length / 128) * 128 - total_length

        # causal_end[i] = block i 在纯因果下能看到的最大 kv 位置（exclusive）
        # lookahead_end[i] = block i 加了 lookahead 后能看到的最大 kv 位置（exclusive）
        causal_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        lookahead_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        for block_i in range(num_blocks):
            q_start = block_i * block_len
            q_end = min((block_i + 1) * block_len, total_length)
            c_end = min((block_i + 1) * block_len, total_length)
            la_end = min((block_i + 1 + lookahead_blocks) * block_len, total_length)
            causal_ends[q_start:q_end] = c_end
            lookahead_ends[q_start:q_end] = la_end

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < lookahead_ends[q_idx]) | (q_idx == kv_idx)
            else:
                # 局部窗口从因果边界往前数，不受 lookahead 影响（lookahead 额外开放）
                local_ok = (kv_idx < causal_ends[q_idx]) & \
                            (kv_idx >= (causal_ends[q_idx] - local_attn_size * frame_seqlen))
                lookahead_ok = (kv_idx >= causal_ends[q_idx]) & (kv_idx < lookahead_ends[q_idx])
                return local_ok | lookahead_ok | (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask, B=None, H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False, device=device
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f" [Curriculum] lookahead mask: num_frames={num_frames}, "
                  f"num_frame_per_block={num_frame_per_block}, lookahead_blocks={lookahead_blocks}")

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        updating_cache=False,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        lookahead_blocks: int = 0,
        force_update_mask: bool = False,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            updating_cache=updating_cache,
        )

        def create_custom_forward(module, block_kwargs):
            def custom_forward(x):
                return module(x, **block_kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            # 获取本层对应的 router（如果启用了 Gumbel Router）
            router = self.layer_routers[block_index] if self.layer_routers is not None else None

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                block_kwargs = dict(
                    e=e0,
                    seq_lens=seq_lens,
                    grid_sizes=grid_sizes,
                    freqs=self.freqs,
                    context=context,
                    context_lens=context_lens,
                    block_mask=self.block_mask,
                    updating_cache=updating_cache,
                    kv_cache=kv_cache[block_index],
                    current_start=current_start,
                    cache_start=cache_start,
                    layer_router=router,
                    router_step=self.current_step,
                    router_total_steps=self.router_total_steps,
                    router_warmup_steps=self.router_warmup_steps,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, block_kwargs),
                    x,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "layer_router": router,
                        "router_step": self.current_step,
                        "router_total_steps": self.router_total_steps,
                        "router_warmup_steps": self.router_warmup_steps,
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
        lookahead_blocks: int = 0,
        force_update_mask: bool = False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        # 课程式 lookahead：当 force_update_mask=True 时，每步都重新生成 mask（lookahead 在变化）。
        # lookahead_blocks==0 且 block_mask 已有缓存时，直接复用缓存（纯因果阶段）。
        frame_seqlen_val = x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2])
        num_frames_val = x.shape[2]
        need_rebuild = (self.block_mask is None) or \
                       (force_update_mask and lookahead_blocks != getattr(self, '_cached_lookahead', -1))

        if need_rebuild:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=num_frames_val,
                        frame_seqlen=frame_seqlen_val,
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=num_frames_val,
                        frame_seqlen=frame_seqlen_val,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                elif lookahead_blocks > 0:
                    # 课程式 mask：允许向前看 lookahead_blocks 个块
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_with_lookahead(
                        device, num_frames=num_frames_val,
                        frame_seqlen=frame_seqlen_val,
                        num_frame_per_block=self.num_frame_per_block,
                        lookahead_blocks=lookahead_blocks,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=num_frames_val,
                        frame_seqlen=frame_seqlen_val,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
            # 缓存当前 lookahead 值，避免 lookahead 没变时重复构建
            self._cached_lookahead = lookahead_blocks

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            router = self.layer_routers[block_index] if self.layer_routers is not None else None

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **{
                        **kwargs,
                        "layer_router": router,
                        "router_step": self.current_step,
                        "router_total_steps": self.router_total_steps,
                        "router_warmup_steps": self.router_warmup_steps,
                    },
                    use_reentrant=False,
                )
            else:
                x = block(x, **{
                    **kwargs,
                    "layer_router": router,
                    "router_step": self.current_step,
                    "router_total_steps": self.router_total_steps,
                    "router_warmup_steps": self.router_warmup_steps,
                })

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        # 训练时的 step 递增由训练管线控制（见 trainer/distillation.py），
        # 此处仅供推理时不会误增。
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
