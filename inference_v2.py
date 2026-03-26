"""
V2 Rolling Forcing 推理脚本

支持 V2 架构参数：
- use_dual_channel_head: 双通道历史信息提取头
- use_gumbel_router: Gumbel-Softmax 门控路由器
- compression_ratio: 历史信息压缩比
- global_layer_indices: 硬编码的全局层索引列表（None表示所有层为全局层）
"""

import argparse
import torch
import os
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
import imageio
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

parser = argparse.ArgumentParser(description="V2 Rolling Forcing Inference")
# 基础参数
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")

# 推理参数
parser.add_argument("--num_output_frames", type=int, default=81,
                    help="Number of output frames")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")

# ========================================
# V2 架构参数（可选，覆盖配置文件）
# ========================================
parser.add_argument("--use_dual_channel_head", type=lambda x: x.lower() == "true", default=None,
                    help="Enable dual channel history extraction head")
parser.add_argument("--use_gumbel_router", type=lambda x: x.lower() == "true", default=None,
                    help="Enable Gumbel-Softmax router")
parser.add_argument("--compression_ratio", type=int, default=None,
                    help="History compression ratio")
parser.add_argument("--global_layer_indices", type=str, default=None,
                    help="Comma-separated global layer indices, e.g., '0,5,10,15'")
parser.add_argument("--reset_cache", action="store_true", default=False,
                    help="Reset KV cache between samples")

args = parser.parse_args()


def merge_args_to_config(config, args):
    """将命令行参数合并到配置中"""
    # 确保 model_kwargs 存在
    if not hasattr(config, 'model_kwargs'):
        config.model_kwargs = OmegaConf.create({})

    # 覆盖 V2 架构参数
    if args.use_dual_channel_head is not None:
        config.model_kwargs.use_dual_channel_head = args.use_dual_channel_head
    if args.use_gumbel_router is not None:
        config.model_kwargs.use_gumbel_router = args.use_gumbel_router
    if args.compression_ratio is not None:
        config.model_kwargs.compression_ratio = args.compression_ratio
    if args.global_layer_indices is not None:
        indices = [int(x.strip()) for x in args.global_layer_indices.split(',')]
        config.model_kwargs.global_layer_indices = indices

    return config


# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

torch.set_grad_enabled(False)

# Load config
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Merge command line args to config
config = merge_args_to_config(config, args)

# Print V2 architecture info
print("=" * 60)
print("V2 Rolling Forcing Inference Configuration")
print("=" * 60)
model_kwargs = getattr(config, 'model_kwargs', {})
print(f"  use_dual_channel_head: {model_kwargs.get('use_dual_channel_head', False)}")
print(f"  use_gumbel_router:     {model_kwargs.get('use_gumbel_router', False)}")
print(f"  compression_ratio:     {model_kwargs.get('compression_ratio', 4)}")
print(f"  global_layer_indices:   {model_kwargs.get('global_layer_indices', None)}")
if model_kwargs.get('global_layer_indices') is None and not model_kwargs.get('use_gumbel_router', False):
    print("  (Default: all layers are global layers)")
print("=" * 60)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    if args.use_ema:
        state_dict_to_load = state_dict['generator_ema']
        def remove_fsdp_prefix(state_dict):
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if "_fsdp_wrapped_module." in key:
                    new_key = key.replace("_fsdp_wrapped_module.", "")
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            return new_state_dict
        state_dict_to_load = remove_fsdp_prefix(state_dict_to_load)
    else:
        state_dict_to_load = state_dict['generator']
    pipeline.generator.load_state_dict(state_dict_to_load)

pipeline = pipeline.to(device=device, dtype=torch.bfloat16)

# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)

num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)

dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]
    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]

    all_video = []
    num_generated_frames = 0

    # Reset cache between samples if requested
    if args.reset_cache and i > 0:
        pipeline.kv_cache_clean = None
        pipeline.crossattn_cache = None

    if args.i2v:
        prompt = batch['prompts'][0]
        prompts = [prompt] * args.num_samples

        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    # Generate video using Rolling Forcing
    video, latents = pipeline.inference_rolling_forcing(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)
