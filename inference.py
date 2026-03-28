import argparse
import os
from collections import OrderedDict

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.io import write_video
from tqdm import tqdm

from pipeline import CausalDiffusionInferencePipeline, CausalInferencePipeline
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed


def build_parser():
    parser = argparse.ArgumentParser(description="Rolling Forcing inference")
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
    parser.add_argument("--output_folder", type=str, help="Output folder")
    parser.add_argument("--num_output_frames", type=int, default=21, help="Number of output frames")
    parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
    parser.add_argument("--save_with_index", action="store_true",
                        help="Whether to save the video using the index or prompt as the filename")
    parser.add_argument("--reset_cache", action="store_true", default=False,
                        help="Reset KV cache between samples")

    # Causal architecture overrides
    parser.add_argument("--local_attn_size", type=int, default=None,
                        help="Recent visible history for local layers, measured in frames")
    parser.add_argument("--sink_size", type=int, default=None,
                        help="Number of sink frames kept in KV cache")
    parser.add_argument("--use_dynamic_anchor", type=lambda x: x.lower() == "true", default=None,
                        help="Enable dynamic anchor")
    parser.add_argument("--use_layer_specialization", type=lambda x: x.lower() == "true", default=None,
                        help="Enable global/local layer specialization")
    parser.add_argument("--use_auto_layer_classification", type=lambda x: x.lower() == "true", default=None,
                        help="Enable automatic layer classification")
    parser.add_argument("--global_layer_indices", type=str, default=None,
                        help="Comma-separated global layer indices, e.g. '0,5,10,15'")
    parser.add_argument("--local_history_blocks", type=int, default=None,
                        help="Number of history blocks visible to local layers")
    parser.add_argument("--anchor_blocks", type=int, default=None,
                        help="Number of block-level anchor tokens kept in the dynamic anchor")
    parser.add_argument("--scene_change_tau", type=float, default=None,
                        help="Scene-change threshold for refreshing the dynamic anchor")
    return parser


def merge_args_to_config(config, args):
    if not hasattr(config, "model_kwargs"):
        config.model_kwargs = OmegaConf.create({})

    if args.local_attn_size is not None:
        config.model_kwargs.local_attn_size = args.local_attn_size
    if args.sink_size is not None:
        config.model_kwargs.sink_size = args.sink_size
    if args.use_dynamic_anchor is not None:
        config.model_kwargs.use_dynamic_anchor = args.use_dynamic_anchor
    if args.use_layer_specialization is not None:
        config.model_kwargs.use_layer_specialization = args.use_layer_specialization
    if args.use_auto_layer_classification is not None:
        config.model_kwargs.use_auto_layer_classification = args.use_auto_layer_classification
    if args.global_layer_indices is not None:
        config.model_kwargs.global_layer_indices = [int(x.strip()) for x in args.global_layer_indices.split(",")]
    if args.local_history_blocks is not None:
        config.model_kwargs.local_history_blocks = args.local_history_blocks
    if args.anchor_blocks is not None:
        config.model_kwargs.anchor_blocks = args.anchor_blocks
    if args.scene_change_tau is not None:
        config.model_kwargs.scene_change_tau = args.scene_change_tau

    return config


def init_distributed(seed):
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        set_seed(seed + local_rank)
    else:
        local_rank = 0
        device = torch.device("cuda")
        set_seed(seed)
    return device, local_rank


def load_pipeline(config, device, checkpoint_path=None, use_ema=False):
    if hasattr(config, "denoising_step_list"):
        pipeline = CausalInferencePipeline(config, device=device)
    else:
        pipeline = CausalDiffusionInferencePipeline(config, device=device)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if use_ema:
            state_dict_to_load = state_dict["generator_ema"]

            def remove_fsdp_prefix(sd):
                new_sd = OrderedDict()
                for key, value in sd.items():
                    if "_fsdp_wrapped_module." in key:
                        new_sd[key.replace("_fsdp_wrapped_module.", "")] = value
                    else:
                        new_sd[key] = value
                return new_sd

            state_dict_to_load = remove_fsdp_prefix(state_dict_to_load)
        else:
            state_dict_to_load = state_dict["generator"]
        pipeline.generator.load_state_dict(state_dict_to_load)

    return pipeline.to(device=device, dtype=torch.bfloat16)


def build_dataset(args):
    if args.i2v:
        transform = transforms.Compose([
            transforms.Resize((480, 832)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return TextImagePairDataset(args.data_path, transform=transform)
    return TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)


def main():
    parser = build_parser()
    args = parser.parse_args()

    device, local_rank = init_distributed(args.seed)
    torch.set_grad_enabled(False)

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config = merge_args_to_config(config, args)

    model_kwargs = getattr(config, "model_kwargs", {})
    print("=" * 60)
    print("Rolling Forcing Inference Configuration")
    print("=" * 60)
    print(f"  use_dynamic_anchor:          {model_kwargs.get('use_dynamic_anchor', False)}")
    print(f"  use_layer_specialization:    {model_kwargs.get('use_layer_specialization', False)}")
    print(f"  use_auto_layer_classification: {model_kwargs.get('use_auto_layer_classification', False)}")
    print(f"  global_layer_indices:        {model_kwargs.get('global_layer_indices', None)}")
    print(f"  local_history_blocks:        {model_kwargs.get('local_history_blocks', 2)}")
    print(f"  anchor_blocks:               {model_kwargs.get('anchor_blocks', 1)}")
    print(f"  scene_change_tau:            {model_kwargs.get('scene_change_tau', 0.6)}")
    print("=" * 60)

    pipeline = load_pipeline(config, device, args.checkpoint_path, args.use_ema)
    dataset = build_dataset(args)
    num_prompts = len(dataset)
    print(f"Number of prompts: {num_prompts}")

    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

    if local_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    for sample_idx, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
        idx = batch_data["idx"].item()
        batch = batch_data if isinstance(batch_data, dict) else batch_data[0]

        if args.reset_cache and sample_idx > 0:
            pipeline.kv_cache_clean = None
            pipeline.crossattn_cache = None
            if hasattr(pipeline.generator.model, "reset_stream_state"):
                pipeline.generator.model.reset_stream_state()

        if args.i2v:
            prompt = batch["prompts"][0]
            prompts = [prompt] * args.num_samples
            image = batch["image"].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
            initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames - 1, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16,
            )
        else:
            prompt = batch["prompts"][0]
            extended_prompt = batch["extended_prompts"][0] if "extended_prompts" in batch else None
            prompts = [extended_prompt if extended_prompt is not None else prompt] * args.num_samples
            initial_latent = None
            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16,
            )

        video, latents = pipeline.inference_rolling_forcing(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
        )
        del latents
        video = 255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()

        pipeline.vae.model.clear_cache()

        if idx < num_prompts:
            model_name = "ema" if args.use_ema else "regular"
            for seed_idx in range(args.num_samples):
                if args.save_with_index:
                    output_path = os.path.join(args.output_folder, f"{idx}-{seed_idx}_{model_name}.mp4")
                else:
                    output_path = os.path.join(args.output_folder, f"{prompt[:100]}-{seed_idx}.mp4")
                write_video(output_path, video[seed_idx], fps=16)


if __name__ == "__main__":
    main()
