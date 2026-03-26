"""
Rolling Forcing 帧间注意力可视化（修复版）
==========================================
直接 hook 进真实推理流程，无需修改模型源码。

修复内容：
  1. window=0 及早期 cache 未满时（local_start_index==0），
     单独捕获"纯窗口内自注意力"并正确绘图（横轴只有 Window 段）
  2. n_query_frames 改用 grid_sizes[0][0] 取值，避免整除误差

用法：
    python visualize_attention_real.py \
        --config_path configs/rolling_forcing_dmd.yaml \
        --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
        --prompt "a cat walking in the garden" \
        --output_dir attention_vis \
        --window_index 5      # 可视化第5个滚动窗口（-1=全部）
"""

import argparse
import os
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict
from omegaconf import OmegaConf


# ─────────────────────────────────────────────
# 1. AttentionVisualizer
# ─────────────────────────────────────────────

class AttentionVisualizer:
    """
    Monkey-patch CausalWanSelfAttention.forward，捕获每层真实注意力权重。

    分支处理：
      - local_start_index == 0  →  "窗口内自注意力"（window=0 等早期帧）
                                    横轴只有 Window 段，无 Anchor/History
      - local_start_index  > 0  →  "完整三段注意力"
                                    横轴 = [Anchor | History | Window]
    """

    def __init__(self, frame_seqlen: int = 1560, num_layers: int = 30,
                 frames_per_block: int = 3):
        self.frame_seqlen     = frame_seqlen
        self.num_layers       = num_layers
        self._frames_per_block = frames_per_block
        self._captures            = []
        self._original_forwards   = {}
        self._installed           = False

    # ── install / uninstall ─────────────────────────────────────────────

    def install(self, model):
        from wan.modules.causal_model import CausalWanSelfAttention
        for block_index, block in enumerate(model.blocks):
            attn = block.self_attn
            assert isinstance(attn, CausalWanSelfAttention)
            self._original_forwards[block_index] = attn.forward
            attn.forward = self._make_hooked_forward(attn, block_index)
        self._installed = True
        print(f"[AttentionVisualizer] Installed hooks on {len(model.blocks)} blocks.")

    def uninstall(self, model):
        for block_index, block in enumerate(model.blocks):
            if block_index in self._original_forwards:
                block.self_attn.forward = self._original_forwards[block_index]
        self._original_forwards.clear()
        self._installed = False
        print("[AttentionVisualizer] Hooks removed.")

    def reset(self):
        self._captures.clear()

    # ── hook 核心 ───────────────────────────────────────────────────────

    def _make_hooked_forward(self, attn_module, block_index):
        original_forward = attn_module.forward
        vis = self

        def hooked_forward(
            x, seq_lens, grid_sizes, freqs,
            block_mask, kv_cache=None, current_start=0,
            cache_start=None, updating_cache=False
        ):
            # 先执行原始 forward，保证 kv_cache 正常更新
            output = original_forward(
                x, seq_lens, grid_sizes, freqs, block_mask,
                kv_cache=kv_cache, current_start=current_start,
                cache_start=cache_start, updating_cache=updating_cache
            )

            # updating_cache（clean cache 更新阶段）和无 cache（训练）跳过
            if updating_cache or kv_cache is None:
                return output

            try:
                b, s = x.shape[0], x.shape[1]
                n, d = attn_module.num_heads, attn_module.head_dim

                frame_seqlen        = math.prod(grid_sizes[0][1:]).item()
                current_start_frame = current_start // frame_seqlen

                # n_query_frames 以 grid_sizes[0][0] 为准（最权威，避免整除误差）
                n_query_frames = int(grid_sizes[0][0].item())

                with torch.no_grad():
                    q = attn_module.norm_q(attn_module.q(x)).view(b, s, n, d)
                    k = attn_module.norm_k(attn_module.k(x)).view(b, s, n, d)

                    from wan.modules.causal_model import causal_rope_apply
                    roped_query = causal_rope_apply(
                        q, grid_sizes, freqs,
                        start_frame=current_start_frame
                    ).type_as(x)
                    roped_key = causal_rope_apply(
                        k, grid_sizes, freqs,
                        start_frame=current_start_frame
                    ).type_as(x)

                    block_length       = attn_module.block_length
                    max_attention_size = attn_module.max_attention_size

                    local_end_index   = kv_cache["local_end_index"].item()
                    local_start_index = local_end_index - block_length

                    # ── 分支 1：local_start_index <= 0
                    #    第一个窗口（cache 尚无历史），只有窗口内自注意力
                    if local_start_index <= 0:
                        attn_map = vis._compute_self_attn(
                            roped_query, roped_key,
                            frame_seqlen, n_query_frames
                        )
                        fpb = vis._frames_per_block
                        vis._captures.append({
                            "layer_index":          block_index,
                            "current_start_frame":  current_start_frame,
                            "start_block":          current_start_frame // frame_seqlen // fpb,
                            "frames_per_block":     fpb,
                            "history_start_frame":  0,
                            "mode":                 "window_only",
                            "attn_map":             attn_map,
                            "n_query_frames":       n_query_frames,
                            "n_anchor_frames":      0,
                            "n_history_frames":     0,
                            "n_window_frames":      n_query_frames,
                            "frame_seqlen":         frame_seqlen,
                        })
                        return output

                    # ── 分支 2：有 Anchor + History + Window 三段 ──
                    query_length             = roped_query.shape[1]
                    working_cache_max_length = max_attention_size - query_length - block_length
                    extract_cache_end        = local_start_index
                    extract_cache_start      = max(
                        block_length,
                        local_start_index - working_cache_max_length
                    )

                    anchor_len  = block_length
                    history_len = extract_cache_end - extract_cache_start
                    window_len  = roped_key.shape[1]

                    # history 段在全局 token cache 中的起始帧编号
                    history_start_frame = extract_cache_start // frame_seqlen

                    attn_map = vis._compute_full_attn(
                        roped_query, roped_key, kv_cache,
                        grid_sizes, freqs,
                        current_start_frame, frame_seqlen,
                        anchor_len, history_len, window_len,
                        extract_cache_start, extract_cache_end,
                        local_end_index, block_length,
                        n_query_frames
                    )

                    vis._captures.append({
                        "layer_index":          block_index,
                        "current_start_frame":  current_start_frame,
                        "start_block":          current_start_frame // frame_seqlen // vis._frames_per_block,
                        "frames_per_block":     vis._frames_per_block,
                        "history_start_frame":  history_start_frame,
                        "mode":                 "full",
                        "attn_map":             attn_map,
                        "n_query_frames":       n_query_frames,
                        "n_anchor_frames":      anchor_len  // frame_seqlen,
                        "n_history_frames":     history_len // frame_seqlen,
                        "n_window_frames":      window_len  // frame_seqlen,
                        "frame_seqlen":         frame_seqlen,
                    })

            except Exception as e:
                print(f"[AttentionVisualizer] Layer {block_index} capture failed: {e}")

            return output

        return hooked_forward

    # ── 注意力计算（内存高效版）──────────────────────────────────────────
    #
    # 原始做法：softmax(Q @ K^T) 一次性实例化完整矩阵
    #   稳定阶段 Lq=14040, Lk=32760, 12 heads → ~20GB → OOM
    #
    # 新做法：每次只取 1 个 query 帧的 1560 个 token，
    #   对全部 K 做点积 + softmax，再在 key 帧维度 mean-pool。
    #   峰值内存：1560 × Lk × 12 × 4B ≈ 最大 ~1.5GB，可接受。
    #   计算全部在 CPU 上进行，不占显存。

    @torch.no_grad()
    def _compute_self_attn(self, roped_query, roped_key, frame_seqlen, n_query_frames):
        """分支1：窗口内自注意力（无历史 cache）。输出 [n_q_blk, n_w_blk]。"""
        q = roped_query[0].float().cpu()  # [Lq, H, D]
        k = roped_key[0].float().cpu()    # [Lk, H, D]
        fpb = self._frames_per_block
        return self._blockwise_frame_attn(q, k, frame_seqlen, n_query_frames, n_query_frames, fpb)

    @torch.no_grad()
    def _compute_full_attn(
        self, roped_query, roped_key, kv_cache,
        grid_sizes, freqs,
        current_start_frame, frame_seqlen,
        anchor_len, history_len, window_len,
        extract_cache_start, extract_cache_end,
        local_end_index, block_length,
        n_query_frames
    ):
        """分支2：完整三段注意力 [Anchor|History|Window]。输出 [n_q, n_k]。"""
        from wan.modules.causal_model import causal_rope_apply

        grid_sizes_one_block = grid_sizes.clone()
        grid_sizes_one_block[:, 0] = 3

        working_cache_key          = kv_cache["k"][:, extract_cache_start:extract_cache_end].clone()
        working_cache_frame_length = working_cache_key.shape[1] // frame_seqlen
        rope_start_frame           = current_start_frame - working_cache_frame_length - 3

        anchor_cache_key = causal_rope_apply(
            kv_cache["k"][:, :block_length],
            grid_sizes_one_block, freqs,
            start_frame=rope_start_frame
        ).type_as(roped_query)

        # 拼接后立即搬到 CPU，释放显存
        input_key = torch.cat([anchor_cache_key, working_cache_key, roped_key], dim=1)
        k = input_key[0].float().cpu()    # [Lk, H, D]
        q = roped_query[0].float().cpu()  # [Lq, H, D]
        del input_key, anchor_cache_key, working_cache_key

        n_k = (anchor_len + history_len + window_len) // frame_seqlen
        fpb = self._frames_per_block
        return self._blockwise_frame_attn(q, k, frame_seqlen, n_query_frames, n_k, fpb)

    @staticmethod
    def _blockwise_frame_attn(
        q: torch.Tensor, k: torch.Tensor,
        frame_seqlen: int, n_q_frames: int, n_k_frames: int,
        frames_per_block: int = 1
    ) -> np.ndarray:
        """
        逐 query 帧分块计算帧级注意力矩阵，输出 [n_q_frames, n_k_frames]。

        流程（每次处理 1 个 query 帧，避免 OOM）：
          1. q_frame @ k^T → [H, frame_seqlen, Lk]
          2. softmax over Lk
          3. head + query-token 维度均值 → [Lk]
          4. 按 key 帧 mean-pool → [n_k_frames]

        Args:
            q:               [Lq, H, D]  CPU float32
            k:               [Lk, H, D]  CPU float32
            n_q_frames:      query 帧数
            n_k_frames:      key 帧数
            frames_per_block: 保留参数，不再用于 pool（帧级输出）
        Returns:
            frame_attn: np.ndarray [n_q_frames, n_k_frames]
        """
        scale  = math.sqrt(q.shape[-1])
        result = np.zeros((n_q_frames, n_k_frames), dtype=np.float32)

        # 预转置 k：[H, D, Lk]
        k_t = k.permute(1, 2, 0)

        for qi in range(n_q_frames):
            q_i          = q[qi * frame_seqlen:(qi + 1) * frame_seqlen].permute(1, 0, 2)
            scores       = torch.matmul(q_i, k_t) / scale          # [H, fsl, Lk]
            weights_mean = torch.softmax(scores, dim=-1).mean(dim=(0, 1))  # [Lk]
            for ki in range(n_k_frames):
                result[qi, ki] = float(
                    weights_mean[ki * frame_seqlen:(ki + 1) * frame_seqlen].mean()
                )
        return result

    # ── 可视化入口 ──────────────────────────────────────────────────────

    def plot_all(self, output_dir: str, window_index: int, start_block: int,
                 current_start_frame: int = None):
        """
        文件名格式：window{window_index:03d}_startblk{start_block:03d}_layer{layer:02d}.png
        window_index 保证唯一，start_block 体现语义。
        """
        os.makedirs(output_dir, exist_ok=True)

        captures = self._captures
        if current_start_frame is not None:
            captures = [c for c in captures
                        if c["current_start_frame"] == current_start_frame]

        if not captures:
            print(f"[AttentionVisualizer] window {window_index}: no captures to plot.")
            return

        captures = sorted(captures, key=lambda c: c["layer_index"])
        all_maps = []
        prefix   = f"window{window_index:03d}_startblk{start_block:03d}"

        for cap in captures:
            fig   = self._plot_single_layer(cap, window_index, start_block)
            fname = os.path.join(
                output_dir,
                f"{prefix}_layer{cap['layer_index']:02d}.png"
            )
            fig.savefig(fname, dpi=120, bbox_inches="tight", facecolor="#0F1117")
            plt.close(fig)
            all_maps.append(cap["attn_map"])
            print(f"  Saved: {fname}")

        summary_map = np.stack(all_maps).mean(0)
        fig   = self._plot_summary(summary_map, captures[0], window_index, start_block)
        fname = os.path.join(output_dir, f"{prefix}_summary.png")
        fig.savefig(fname, dpi=120, bbox_inches="tight", facecolor="#0F1117")
        plt.close(fig)
        print(f"  Saved summary: {fname}")

    # ── 构造横纵轴标签 ───────────────────────────────────────────────────

    @staticmethod
    def _make_axis_labels(cap: dict):
        """
        纵轴（Query）：当前窗口内每一帧的全局帧索引。
        横轴（Key）：每一帧一个 tick，带段前缀标注来源。
          - Anchor 段：A:f0, A:f1, A:f2, ...
          - History 段：H:f3, H:f4, ...
          - Window 段：W:f12, W:f13, ...
        返回 (y_labels, x_labels)
        """
        start_frame  = cap["current_start_frame"]   # 全局起始帧
        n_q          = cap["n_query_frames"]

        # y labels: 每帧一个，全局帧编号
        y_labels = [f"f{start_frame + i}" for i in range(n_q)]

        mode = cap["mode"]
        if mode == "window_only":
            x_labels = [f"W:f{start_frame + i}" for i in range(n_q)]
        else:
            n_a = cap["n_anchor_frames"]
            n_h = cap["n_history_frames"]
            n_w = cap["n_window_frames"]

            # Anchor: 全局帧 0 开始
            x_a = [f"A:f{i}" for i in range(n_a)]

            # History: 全局帧索引从 history_start_frame 开始
            h_start_f = cap["history_start_frame"]
            x_h = [f"H:f{h_start_f + i}" for i in range(n_h)]

            # Window: 全局帧索引从 start_frame 开始
            x_w = [f"W:f{start_frame + i}" for i in range(n_w)]

            x_labels = x_a + x_h + x_w

        return y_labels, x_labels

    # ── 单层图 ──────────────────────────────────────────────────────────

    def _plot_single_layer(self, cap: dict, window_index: int, start_block: int):
        mode   = cap["mode"]
        attn   = cap["attn_map"]           # [n_q_frames, n_k_frames]
        n_q    = attn.shape[0]
        n_k    = attn.shape[1]
        n_a    = cap["n_anchor_frames"]  if mode == "full" else 0
        n_h    = cap["n_history_frames"] if mode == "full" else 0
        n_w    = cap["n_window_frames"]

        y_labels, x_labels = self._make_axis_labels(cap)

        fig, axes = plt.subplots(
            1, 2, figsize=(max(12, n_k * 0.7 + 4), max(4, n_q * 0.6 + 2)),
            gridspec_kw={"width_ratios": [3, 1]},
            facecolor="#0F1117"
        )
        ax_heat, ax_bar = axes

        im = ax_heat.imshow(attn, aspect="auto", cmap="magma",
                            vmin=0, vmax=attn.max() or 1e-6)
        plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)

        if mode == "full":
            if n_a > 0:
                ax_heat.axvline(x=n_a - 0.5,
                                color="#FF6B35", linewidth=1.5, linestyle="--", alpha=0.8)
            if n_h > 0:
                ax_heat.axvline(x=n_a + n_h - 0.5,
                                color="#4ECDC4", linewidth=1.5, linestyle="--", alpha=0.8)
            self._shade_columns(ax_heat, 0,     n_a,     "#FF6B35", alpha=0.08)
            self._shade_columns(ax_heat, n_a,   n_a+n_h, "#4ECDC4", alpha=0.08)
            self._shade_columns(ax_heat, n_a+n_h, n_k,   "#45B7D1", alpha=0.08)
            mode_tag = "[Anchor | History | Window]"
        else:
            self._shade_columns(ax_heat, 0, n_k, "#45B7D1", alpha=0.08)
            mode_tag = "Window-only (no history yet)"

        ax_heat.set_xticks(range(n_k))
        ax_heat.set_xticklabels(x_labels, rotation=45, ha="right",
                                fontsize=7, color="white")
        ax_heat.set_yticks(range(n_q))
        ax_heat.set_yticklabels(y_labels, fontsize=8, color="white")
        ax_heat.set_xlabel(f"Key blocks  ({mode_tag})", color="white", fontsize=9)
        ax_heat.set_ylabel("Query blocks (current window)", color="white", fontsize=9)
        ax_heat.set_title(
            f"window={window_index}  start_block={start_block}  |  Transformer layer {cap['layer_index']:02d}",
            color="white", fontsize=10, pad=8
        )
        ax_heat.tick_params(colors="white")
        for sp in ax_heat.spines.values():
            sp.set_edgecolor("#444")

        # ── 右侧条形图（归一化为百分比）──
        if mode == "full":
            seg_names  = ["Anchor", "History", "Window"]
            seg_colors = ["#FF6B35", "#4ECDC4", "#45B7D1"]
            raw = [
                float(attn[:, :n_a].mean())          if n_a > 0 else 0.0,
                float(attn[:, n_a:n_a+n_h].mean())   if n_h > 0 else 0.0,
                float(attn[:, n_a+n_h:].mean())       if n_w > 0 else 0.0,
            ]
        else:
            seg_names  = ["Window"]
            seg_colors = ["#45B7D1"]
            raw = [float(attn.mean())]

        total     = sum(raw) or 1.0
        normed    = [v / total * 100 for v in raw]  # 转为百分比

        bars = ax_bar.barh(seg_names, normed, color=seg_colors,
                           edgecolor="none", height=0.5)
        for bar, pct, raw_v in zip(bars, normed, raw):
            ax_bar.text(
                min(pct + 1.5, 95),
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%  ({raw_v:.2e})",
                va="center", ha="left", color="white", fontsize=7
            )
        ax_bar.set_xlim(0, 110)
        ax_bar.set_xlabel("Proportion of total attention (%)", color="white", fontsize=8)
        ax_bar.set_title("Segment weights (normalized)", color="white", fontsize=9)
        ax_bar.tick_params(colors="white")
        ax_bar.set_facecolor("#0F1117")
        for sp in ax_bar.spines.values():
            sp.set_edgecolor("#444")

        fig.patch.set_facecolor("#0F1117")
        fig.tight_layout(pad=1.5)
        return fig

    # ── 汇总图 ──────────────────────────────────────────────────────────

    def _plot_summary(self, summary_map: np.ndarray, cap0: dict, window_index: int, start_block: int):
        mode   = cap0["mode"]
        n_q    = summary_map.shape[0]
        n_k    = summary_map.shape[1]
        n_a    = cap0["n_anchor_frames"]  if mode == "full" else 0
        n_h    = cap0["n_history_frames"] if mode == "full" else 0
        n_w    = cap0["n_window_frames"]

        y_labels, x_labels = self._make_axis_labels(cap0)

        fig, ax = plt.subplots(
            figsize=(max(10, n_k * 0.7 + 3), max(4, n_q * 0.6 + 2)),
            facecolor="#0F1117"
        )
        im = ax.imshow(summary_map, aspect="auto", cmap="magma",
                       vmin=0, vmax=summary_map.max() or 1e-6)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

        if mode == "full":
            if n_a > 0:
                ax.axvline(x=n_a - 0.5,
                           color="#FF6B35", linewidth=1.5, linestyle="--", alpha=0.9)
            if n_h > 0:
                ax.axvline(x=n_a + n_h - 0.5,
                           color="#4ECDC4", linewidth=1.5, linestyle="--", alpha=0.9)
            self._shade_columns(ax, 0,      n_a,     "#FF6B35", alpha=0.10)
            self._shade_columns(ax, n_a,    n_a+n_h, "#4ECDC4", alpha=0.10)
            self._shade_columns(ax, n_a+n_h, n_k,    "#45B7D1", alpha=0.10)
            mode_tag = "[Anchor | History | Window]"
            fpb = cap0["frames_per_block"]
            h_f = cap0["history_start_frame"]
            legend_patches = [
                mpatches.Patch(color="#FF6B35",
                               label=f"Anchor  ({n_a} blks / {n_a*fpb} frames: f0~f{n_a*fpb-1})"),
                mpatches.Patch(color="#4ECDC4",
                               label=f"History ({n_h} blks / {n_h*fpb} frames: f{h_f}~f{h_f+n_h*fpb-1})"),
                mpatches.Patch(color="#45B7D1",
                               label=f"Window  ({n_w} blks / {n_w*fpb} frames: f{start_block*fpb}~f{start_block*fpb+n_w*fpb-1})"),
            ]
        else:
            self._shade_columns(ax, 0, n_k, "#45B7D1", alpha=0.10)
            mode_tag = "(window-only, no history yet)"
            fpb = cap0["frames_per_block"]
            legend_patches = [
                mpatches.Patch(color="#45B7D1",
                               label=f"Window ({n_w} blks / {n_w*fpb} frames: f{start_block*fpb}~f{start_block*fpb+n_w*fpb-1}) — no history"),
            ]

        ax.set_xticks(range(n_k))
        ax.set_xticklabels(x_labels, rotation=45, ha="right",
                           fontsize=7, color="white")
        ax.set_yticks(range(n_q))
        ax.set_yticklabels(y_labels, fontsize=8, color="white")
        ax.set_title(
            f"window={window_index}  start_block={start_block}  |  All-layer average  {mode_tag}",
            color="white", fontsize=11, pad=8
        )
        ax.set_xlabel(f"Key blocks  {mode_tag}", color="white", fontsize=9)
        ax.set_ylabel("Query blocks", color="white", fontsize=9)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
                  facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

        fig.patch.set_facecolor("#0F1117")
        fig.tight_layout(pad=1.5)
        return fig

    @staticmethod
    def _shade_columns(ax, x_start, x_end, color, alpha=0.1):
        if x_end > x_start:
            ax.axvspan(x_start - 0.5, x_end - 0.5,
                       alpha=alpha, color=color, linewidth=0)


# ─────────────────────────────────────────────
# 2. 主推理循环（含可视化）
# ─────────────────────────────────────────────

def run(args):
    from pipeline import CausalInferencePipeline
    from utils.misc import set_seed

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    config         = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config         = OmegaConf.merge(default_config, config)

    pipeline = CausalInferencePipeline(config, device=device)

    if args.checkpoint_path:
        print(f"Loading checkpoint: {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        key = "generator_ema" if args.use_ema else "generator"
        sd  = state_dict[key]
        new_sd = OrderedDict()
        for k, v in sd.items():
            new_sd[k.replace("_fsdp_wrapped_module.", "")] = v
        pipeline.generator.load_state_dict(new_sd)
        print("Checkpoint loaded.")

    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)

    vis = AttentionVisualizer(
        frame_seqlen=pipeline.frame_seq_length,
        num_layers=pipeline.num_transformer_blocks,
        frames_per_block=pipeline.num_frame_per_block
    )
    vis.install(pipeline.generator.model)

    conditional_dict = pipeline.text_encoder(text_prompts=[args.prompt])

    num_frames    = args.num_output_frames
    sampled_noise = torch.randn(
        [1, num_frames, 16, 60, 104],
        device=device, dtype=torch.bfloat16
    )

    batch_size = 1
    pipeline._initialize_kv_cache(batch_size, torch.bfloat16, device)
    pipeline._initialize_crossattn_cache(batch_size, torch.bfloat16, device)

    num_frame_per_block          = pipeline.num_frame_per_block
    assert num_frames % num_frame_per_block == 0
    num_blocks                   = num_frames // num_frame_per_block
    num_denoising_steps          = len(pipeline.denoising_step_list)
    rolling_window_length_blocks = num_denoising_steps
    window_num                   = num_blocks + rolling_window_length_blocks - 1

    window_start_blocks, window_end_blocks = [], []
    for wi in range(window_num):
        window_start_blocks.append(max(0, wi - rolling_window_length_blocks + 1))
        window_end_blocks.append(min(num_blocks - 1, wi))

    shared_timestep = torch.ones(
        [batch_size, rolling_window_length_blocks * num_frame_per_block],
        device=device, dtype=torch.float32
    )
    for idx, ts in enumerate(reversed(pipeline.denoising_step_list)):
        shared_timestep[
            :, idx * num_frame_per_block:(idx + 1) * num_frame_per_block
        ] *= ts

    noisy_cache = torch.zeros_like(sampled_noise)
    output      = torch.zeros(
        [batch_size, num_frames, 16, 60, 104],
        device=device, dtype=torch.bfloat16
    )
    os.makedirs(args.output_dir, exist_ok=True)

    for window_index in range(window_num):
        print(f"\n=== Window {window_index}/{window_num-1} ===")

        start_block         = window_start_blocks[window_index]
        end_block           = window_end_blocks[window_index]
        current_start_frame = start_block * num_frame_per_block
        current_end_frame   = (end_block + 1) * num_frame_per_block
        current_num_frames  = current_end_frame - current_start_frame

        if (current_num_frames == rolling_window_length_blocks * num_frame_per_block
                or current_start_frame == 0):
            noisy_input = torch.cat([
                noisy_cache[
                    :, current_start_frame:current_end_frame - num_frame_per_block
                ],
                sampled_noise[
                    :, current_end_frame - num_frame_per_block:current_end_frame
                ]
            ], dim=1)
        else:
            noisy_input = noisy_cache[:, current_start_frame:current_end_frame]

        if current_num_frames == rolling_window_length_blocks * num_frame_per_block:
            current_timestep = shared_timestep
        elif current_start_frame == 0:
            current_timestep = shared_timestep[:, -current_num_frames:]
        else:
            current_timestep = shared_timestep[:, :current_num_frames]

        # ── 正向推理（hook 在此触发） ──
        vis.reset()
        _, denoised_pred = pipeline.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=current_timestep,
            kv_cache=pipeline.kv_cache_clean,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length,
        )
        output[:, current_start_frame:current_end_frame] = denoised_pred

        # ── 可视化 ──
        should_plot = (args.window_index < 0) or (window_index == args.window_index)
        if should_plot and vis._captures:
            mode = vis._captures[0]["mode"]
            print(f"[Vis] Plotting start_block={start_block} "
                  f"({len(vis._captures)} layers, mode={mode})...")
            vis.plot_all(
                output_dir=args.output_dir,
                window_index=window_index,
                start_block=start_block,
                current_start_frame=current_start_frame,
            )

        # ── 更新 noisy_cache ──
        with torch.no_grad():
            for block_idx in range(start_block, end_block + 1):
                bt = current_timestep[
                    :,
                    (block_idx - start_block) * num_frame_per_block:
                    (block_idx - start_block + 1) * num_frame_per_block
                ].mean().item()
                matches = torch.abs(pipeline.denoising_step_list - bt) < 1e-4
                bidx    = torch.nonzero(matches, as_tuple=True)[0]
                if bidx == len(pipeline.denoising_step_list) - 1:
                    continue
                next_ts = pipeline.denoising_step_list[bidx + 1].to(device)
                noisy_cache[
                    :,
                    block_idx * num_frame_per_block:
                    (block_idx + 1) * num_frame_per_block
                ] = pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_ts * torch.ones(
                        [batch_size * current_num_frames],
                        device=device, dtype=torch.long
                    )
                ).unflatten(0, denoised_pred.shape[:2])[
                    :,
                    (block_idx - start_block) * num_frame_per_block:
                    (block_idx - start_block + 1) * num_frame_per_block
                ]

        # ── 更新 clean cache（updating_cache=True，hook 自动跳过） ──
        with torch.no_grad():
            ctx_ts = torch.ones_like(current_timestep) * pipeline.args.context_noise
            pipeline.generator(
                noisy_image_or_video=denoised_pred[:, :num_frame_per_block],
                conditional_dict=conditional_dict,
                timestep=ctx_ts[:, :num_frame_per_block],
                kv_cache=pipeline.kv_cache_clean,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
                updating_cache=True,
            )

    vis.uninstall(pipeline.generator.model)

    # ── VAE decode + 保存视频 ──
    print("\nDecoding latents to video...")
    with torch.no_grad():
        video = pipeline.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        pipeline.vae.model.clear_cache()

    from einops import rearrange
    from torchvision.io import write_video
    video_uint8 = (rearrange(video, 'b t c h w -> b t h w c') * 255.0).byte()
    # 取 prompt 前 60 个字符作为文件名，去掉非法字符
    safe_prompt = "".join(c for c in args.prompt[:60] if c.isalnum() or c in " _-").strip()
    video_path  = os.path.join(args.output_dir, f"{safe_prompt}.mp4")
    write_video(video_path, video_uint8[0].cpu(), fps=16)
    print(f"Video saved: {video_path}")

    print(f"Attention maps saved to: {args.output_dir}")


# ─────────────────────────────────────────────
# 3. CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling Forcing attention visualizer")
    parser.add_argument("--config_path",       type=str, required=True)
    parser.add_argument("--checkpoint_path",   type=str, default="")
    parser.add_argument("--prompt",            type=str,
                        default="a cat walking in the garden")
    parser.add_argument("--num_output_frames", type=int, default=21)
    parser.add_argument("--output_dir",        type=str, default="attention_vis")
    parser.add_argument("--window_index",      type=int, default=-1,
                        help="Which window to visualize (-1 = all)")
    parser.add_argument("--use_ema",           action="store_true")
    parser.add_argument("--seed",              type=int, default=42)
    args = parser.parse_args()

    run(args)