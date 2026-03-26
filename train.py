import argparse
import os
from omegaconf import OmegaConf
from trainer import DiffusionTrainer, GANTrainer, ODETrainer, ScoreDistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="",
                        help="日志目录（TensorBoard、WandB）")
    parser.add_argument("--model-save-dir", type=str, default="",
                        help="模型保存目录（checkpoint_last.pt 和历史快照）")
    parser.add_argument("--wandb-save-dir", type=str, default="")
    parser.add_argument("--disable-wandb", default=False, action="store_true")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint_last.pt to resume from")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config.config_name = os.path.basename(args.config_path).split(".")[0]
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir if args.wandb_save_dir else args.logdir
    config.disable_wandb = args.disable_wandb
    # model_save_dir 未传时回退到 logdir（向后兼容旧命令行）
    config.model_save_dir = args.model_save_dir if args.model_save_dir else args.logdir

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "gan":
        trainer = GANTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config, resume_path=args.resume)
    else:
        raise ValueError(f"Unknown trainer: {config.trainer}")

    trainer.train()


if __name__ == "__main__":
    main()