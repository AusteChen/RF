import argparse
import torch
import os
import math
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
import re 

# --- Import ---
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# --- 导入项目模块 ---
from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

# ！！！导入 Causal 模块 ！！！
import wan.modules.causal_model as causal_module
from wan.modules.causal_model import CausalWanSelfAttention

# =========================================================================
#  核心工具：WanAttentionVisualizer (V6 - 修复版)
# =========================================================================
class WanAttentionVisualizer:
    def __init__(self, model):
        # 存储全局注意力图: {layer_id: Global_Matrix}
        self.global_data = {}  
        self.model = model
        self.frame_len = 1560 # WanModel 固定参数
        
        # 线程局部变量
        self._current_frame_offset = 0 
        self._layer_counter = 0
        
        # 自动探测层数
        try:
            if hasattr(model, 'generator'):
                self.num_layers = len(model.generator.model.blocks)
            elif hasattr(model, 'model'):
                self.num_layers = len(model.model.blocks)
            elif hasattr(model, 'blocks'):
                self.num_layers = len(model.blocks)
            else:
                self.num_layers = 32
        except:
            self.num_layers = 32
            
        print(f"[Visualizer] 初始化: 检测到 {self.num_layers} 层 Transformer。")

        # 保存原始函数
        self.original_attn_forward = CausalWanSelfAttention.forward
        self.original_attention_calc = causal_module.attention

    def __enter__(self):
        # 安装双重 Hook
        # 注意：这里的 self._hooked_layer_forward 是一个 bound method
        # 当它被赋值给类属性时，调用时会自动传入 instance 作为第一个参数（即 self_layer），
        # 但因为它是 bound method，它已经携带了 visualizer 实例作为 self。
        CausalWanSelfAttention.forward = self._hooked_layer_forward
        causal_module.attention = self._hooked_attention_calc
        print(f"[Visualizer] 全局拼接模式 Hook 已启动。")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 卸载 Hook
        CausalWanSelfAttention.forward = self.original_attn_forward
        causal_module.attention = self.original_attention_calc
        print(f"[Visualizer] Hook 已移除。")

    # --- Hook 1: 拦截层的前向传播，获取位置信息 ---
    # 修复点：添加了 'self' 作为第一个参数
    def _hooked_layer_forward(self, self_layer, x, seq_lens, grid_sizes, freqs, block_mask, 
                              kv_cache=None, current_start=0, cache_start=None, updating_cache=False):
        
        # 记录当前的起始帧号
        self._current_frame_offset = current_start // self.frame_len
        
        # 继续原始计算
        return self.original_attn_forward(
            self_layer, x, seq_lens, grid_sizes, freqs, block_mask, 
            kv_cache, current_start, cache_start, updating_cache
        )

    # --- Hook 2: 拦截注意力计算，获取权重并拼图 ---
    def _hooked_attention_calc(self, q, k, v, *args, **kwargs):
        try:
            self._compute_and_stitch(q, k)
        except Exception as e:
            # print(f"[Viz Error] {e}")
            pass

        return self.original_attention_calc(q, k, v, *args, **kwargs)

    def _compute_and_stitch(self, q, k):
        with torch.no_grad():
            # 1. 预处理维度
            if q.ndim == 3: q = q.unsqueeze(1)
            if k.ndim == 3: k = k.unsqueeze(1)
            if q.shape[1] != k.shape[1] and q.shape[2] == k.shape[2]: 
                 q = q.permute(0, 2, 1, 3)
                 k = k.permute(0, 2, 1, 3)

            # 只取第一个 Head
            head_dim = q.shape[-1]
            scale = 1.0 / math.sqrt(head_dim)
            q_vec = q[0, 0, :, :].float()
            k_vec = k[0, 0, :, :].float()

            # 2. 计算局部注意力矩阵
            attn_scores = torch.matmul(q_vec, k_vec.transpose(0, 1)) * scale
            attn_probs = F.softmax(attn_scores, dim=-1)

            # 3. 聚合：Token -> Frame
            num_local_frames = math.ceil(attn_probs.shape[0] / self.frame_len)
            num_history_frames = math.ceil(attn_probs.shape[1] / self.frame_len)
            
            local_frame_matrix = torch.zeros((num_local_frames, num_history_frames))
            
            for i in range(num_local_frames):
                s_q = i * self.frame_len
                e_q = min((i + 1) * self.frame_len, attn_probs.shape[0])
                if s_q >= e_q: continue
                
                sub_attn = attn_probs[s_q:e_q, :]
                
                for j in range(num_history_frames):
                    s_k = j * self.frame_len
                    e_k = min((j + 1) * self.frame_len, attn_probs.shape[1])
                    if s_k >= e_k: continue
                    
                    block_val = sub_attn[:, s_k:e_k].sum(dim=1).mean()
                    local_frame_matrix[i, j] = block_val.cpu()

            # 4. 全局拼接
            layer_id = self._layer_counter % self.num_layers
            self._layer_counter += 1
            
            if layer_id not in self.global_data:
                self.global_data[layer_id] = torch.zeros((100, 100))
            
            global_map = self.global_data[layer_id]
            
            start_row = self._current_frame_offset
            num_rows = local_frame_matrix.shape[0]
            num_cols = local_frame_matrix.shape[1]
            
            # 动态扩容
            max_needed = max(start_row + num_rows, num_cols)
            if max_needed > global_map.shape[0]:
                new_size = max(max_needed, global_map.shape[0] * 2)
                new_map = torch.zeros((new_size, new_size))
                new_map[:global_map.shape[0], :global_map.shape[1]] = global_map
                self.global_data[layer_id] = new_map
                global_map = self.global_data[layer_id]

            # 写入
            global_map[start_row : start_row + num_rows, :num_cols] = local_frame_matrix

    def save_all_layers(self, save_dir):
        if not self.global_data:
            print(f"[Visualizer] 无数据。")
            return

        print(f"[Visualizer] 保存中...")
        layers = sorted(self.global_data.keys())

        for layer_id in layers:
            matrix = self.global_data[layer_id]
            
            # 裁剪
            non_zero_rows = torch.nonzero(matrix.sum(dim=1))
            non_zero_cols = torch.nonzero(matrix.sum(dim=0))
            
            if len(non_zero_rows) > 0 and len(non_zero_cols) > 0:
                max_row = non_zero_rows.max().item() + 1
                max_col = non_zero_cols.max().item() + 1
                matrix = matrix[:max_row, :max_col]
            else:
                continue 

            plt.figure(figsize=(10, 8))
            sns.heatmap(matrix.numpy(), cmap="Reds", square=True, vmin=0, vmax=1)
            plt.title(f"Layer {layer_id} Global Attention Map")
            plt.xlabel("Key Frame (Global Index)")
            plt.ylabel("Query Frame (Global Index)")
            
            filename = f"layer_{layer_id:02d}.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=100)
            plt.close()

# =========================================================================
#  主程序
# =========================================================================

def sanitize_filename(name):
    clean_name = re.sub(r'[^\w\-_\. ]', '_', name)
    return clean_name[:50].strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
    parser.add_argument("--output_folder", type=str, help="Output folder")
    parser.add_argument("--num_output_frames", type=int, default=21)
    parser.add_argument("--i2v", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--save_with_index", action="store_true")
    args = parser.parse_args()

    # Init
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        set_seed(args.seed + local_rank)
    else:
        device = torch.device("cuda")
        local_rank = 0
        set_seed(args.seed)

    torch.set_grad_enabled(False)

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    if hasattr(config, 'denoising_step_list'):
        pipeline = CausalInferencePipeline(config, device=device)
    else:
        pipeline = CausalDiffusionInferencePipeline(config, device=device)

    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        if args.use_ema:
            state_dict_to_load = state_dict['generator_ema']
            new_state_dict = OrderedDict()
            for key, value in state_dict_to_load.items():
                if "_fsdp_wrapped_module." in key:
                    new_state_dict[key.replace("_fsdp_wrapped_module.", "")] = value
                else:
                    new_state_dict[key] = value
            state_dict_to_load = new_state_dict
        else:
            state_dict_to_load = state_dict['generator']
        pipeline.generator.load_state_dict(state_dict_to_load)

    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)

    if args.i2v:
        transform = transforms.Compose([
            transforms.Resize((480, 832)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = TextImagePairDataset(args.data_path, transform=transform)
    else:
        dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
    
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)
        
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

    if local_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
        idx = batch_data['idx'].item()
        if isinstance(batch_data, dict):
            batch = batch_data
        elif isinstance(batch_data, list):
            batch = batch_data[0]

        all_video = []
        prompt_str = batch['prompts'][0]
        
        if args.i2v:
            prompts = [prompt_str] * args.num_samples
            image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
            initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
            )
        else:
            extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
            prompts = [extended_prompt] * args.num_samples if extended_prompt else [prompt_str] * args.num_samples
            initial_latent = None
            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
            )

        with WanAttentionVisualizer(pipeline) as viz:
            video, latents = pipeline.inference_rolling_forcing(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
                initial_latent=initial_latent,
            )
        
        if local_rank == 0:
            safe_name = sanitize_filename(prompt_str)
            sub_folder_name = f"{idx:03d}_{safe_name}"
            prompt_save_dir = os.path.join(args.output_folder, sub_folder_name)
            os.makedirs(prompt_save_dir, exist_ok=True)
            viz.save_all_layers(save_dir=prompt_save_dir)

        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)
        video = 255.0 * torch.cat(all_video, dim=1)
        pipeline.vae.model.clear_cache()

        if idx < len(dataset):
            for seed_idx in range(args.num_samples):
                output_path = os.path.join(prompt_save_dir, f"video_seed{seed_idx}.mp4")
                write_video(output_path, video[seed_idx], fps=16)

if __name__ == "__main__":
    main()