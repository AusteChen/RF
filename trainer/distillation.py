import gc
import logging
import random
import numpy as np
import wandb

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import set_seed, merge_dict_list
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD
from wan.modules.causal_model import CausalTransitionScheduler
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import time
import os


# ---------------------------------------------------------------------------
# RNG 状态保存/恢复（用于断点续训精确复现）
# ---------------------------------------------------------------------------

def _get_rng_state():
    return {
        "cpu":    torch.get_rng_state(),
        "cuda":   torch.cuda.get_rng_state_all(),
        "numpy":  np.random.get_state(),
        "random": random.getstate(),
    }


def _set_rng_state(state):
    try:
        torch.set_rng_state(state["cpu"])
        torch.cuda.set_rng_state_all(state["cuda"])
        np.random.set_state(state["numpy"])
        random.setstate(state["random"])
    except Exception as e:
        print(f"[WARN] Failed to restore RNG state: {e}")


# ---------------------------------------------------------------------------
# WandB 可视化工具（仿照 CausVid prepare_for_saving）
# ---------------------------------------------------------------------------

def _prepare_for_wandb(tensor, fps=16, caption=None):
    """
    将 VAE 解码后的像素张量（范围 [-1,1]）转成 wandb 可接受的格式。
    - 4D [B, C, H, W]    -> wandb.Image（网格拼图）
    - 5D [B, T, C, H, W] -> wandb.Video
      注意：wandb.Video 期望 [B, T, C, H, W]，channels 在空间维度前面，不能做 permute

    仿照 CausVid/causvid/util.py 中的 prepare_for_saving 函数实现
    """
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1).detach()
    if tensor.ndim == 4:
        # [B, C, H, W] -> 拼图 grid 后转为 wandb.Image
        tensor = make_grid(tensor, 4, padding=0, normalize=False)
        return wandb.Image((tensor * 255).cpu().numpy().astype(np.uint8), caption=caption)
    elif tensor.ndim == 5:
        # [B, T, C, H, W] -> wandb.Video，直接传，不做 permute
        return wandb.Video((tensor * 255).cpu().numpy().astype(np.uint8), fps=fps, format="webm", caption=caption)
    else:
        raise ValueError("Unsupported tensor shape for saving.")


class Trainer:
    def __init__(self, config, resume_path=None):
        self.config = config
        self.resume_path = resume_path

        # ------------------------------------------------------------------
        # Step 1: 分布式环境
        # ------------------------------------------------------------------
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal

        # ------------------------------------------------------------------
        # 断点续训：预读取 checkpoint，获取 step 和 wandb_id
        # ------------------------------------------------------------------
        self._resume_ckpt = None
        self._resume_ema_state = None
        self._resume_ckpt_step = None
        self._resume_wandb_id = None

        if self.resume_path:
            if self.is_main_process:
                print(f"[Resume] Loading checkpoint: {self.resume_path}")
            ckpt = torch.load(self.resume_path, map_location="cpu", weights_only=False)
            self._resume_ckpt = ckpt
            self._resume_ema_state = ckpt.get("generator_ema", None)
            self._resume_ckpt_step = ckpt.get("step", 0)
            self._resume_wandb_id = ckpt.get("wandb_id", None)
            self.step = self._resume_ckpt_step + 1
            if self.is_main_process:
                print(f"[Resume] step={self.step}, ckpt_step={self._resume_ckpt_step}, "
                      f"wandb_id={self._resume_wandb_id}")
        else:
            self.step = 0


        # 随机种子
        if config.seed == 0:
            random_seed = torch.randint(0, 10_000_000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
        set_seed(config.seed + global_rank)

        # ------------------------------------------------------------------
        # 日志（TensorBoard + WandB）
        # ------------------------------------------------------------------
        if self.is_main_process:
            os.makedirs(config.logdir, exist_ok=True)
            self.writer = SummaryWriter(
                log_dir=os.path.join(config.logdir, "tensorboard"),
                flush_secs=10,
            )

            if not getattr(config, "disable_wandb", False):
                wandb_key = getattr(config, "wandb_key", None)
                if wandb_key:
                    wandb.login(key=wandb_key)

                run_name, run_id, resume_mode = self._resolve_wandb_resume()
                wandb_kwargs = dict(
                    project=getattr(config, "wandb_project", "rolling_forcing"),
                    entity=getattr(config, "wandb_entity", None),
                    name=run_name,
                    dir=getattr(config, "wandb_save_dir", config.logdir),
                    config={k: v for k, v in config.items()
                            if not callable(v) and not k.startswith("_")},
                )
                if run_id is not None:
                    wandb_kwargs["id"] = run_id
                if resume_mode is not None:
                    wandb_kwargs["resume"] = resume_mode

                wandb.init(**wandb_kwargs)
                print(f"[WandB] project={wandb_kwargs['project']}, "
                      f"run_name={run_name}, run_id={run_id}, resume_mode={resume_mode}")


        # log 目录：TensorBoard、WandB 日志
        self.log_path = config.logdir
        # 模型保存目录：checkpoint_last.pt 和历史快照
        # 优先使用 config.model_save_dir，未配置时回退到 logdir（向后兼容）
        self.output_path = getattr(config, "model_save_dir", config.logdir)
        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            print(f"[Path] Log dir  : {self.log_path}")
            print(f"[Path] Model dir: {self.output_path}")

        # ------------------------------------------------------------------
        # 课程式学习调度器
        # ------------------------------------------------------------------
        self.use_curriculum = getattr(config, "use_curriculum", False)
        self.transition_scheduler = None
        if self.use_curriculum:
            num_frames = config.image_or_video_shape[1]
            num_frame_per_block = getattr(config, "num_frame_per_block", 1)
            total_blocks = num_frames // num_frame_per_block
            transition_steps = getattr(config, "transition_steps", 50000)
            decay_mode = getattr(config, "decay_mode", "linear")
            self.transition_scheduler = CausalTransitionScheduler(
                total_steps=transition_steps,
                num_frames=total_blocks,
                start_step=getattr(config, "curriculum_start_step", 0),
                decay_mode=decay_mode,
            )
            if self.is_main_process:
                print(f"[Curriculum] total_blocks={total_blocks}, "
                      f"transition_steps={transition_steps}, decay_mode={decay_mode}")

        # ------------------------------------------------------------------
        # Step 2: 模型
        # ------------------------------------------------------------------
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError(f"Invalid distribution_loss: {config.distribution_loss}")

        # 加载预训练 generator（断点续训时跳过，resume ckpt 里已含完整权重）
        if getattr(config, "generator_ckpt", False) and self.resume_path is None:
            print(f"[Init] Loading generator ckpt: {config.generator_ckpt}")
            sd = torch.load(config.generator_ckpt, map_location="cpu", weights_only=False)
            if "generator" in sd:
                sd = sd["generator"]
            elif "model" in sd:
                sd = sd["model"]
            sd = {"model." + k: v for k, v in sd.items()}
            missing, unexpected = self.model.generator.load_state_dict(sd, strict=False)
            if missing:
                print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            if not missing and not unexpected:
                print("[Init] Generator loaded successfully.")

        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        # ------------------------------------------------------------------
        # Step 3: FSDP wrap
        # ------------------------------------------------------------------
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
        )
        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
        )
        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
        )
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device,
                dtype=torch.bfloat16 if config.mixed_precision else torch.float32,
            )

        # ------------------------------------------------------------------
        # Step 4: 优化器
        # ------------------------------------------------------------------
        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.model.generator.parameters() if p.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
        self.critic_optimizer = torch.optim.AdamW(
            [p for p in self.model.fake_score.parameters() if p.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay,
        )

        # 断点续训：恢复权重 + 优化器 + RNG（在 FSDP wrap 之后）
        if self._resume_ckpt is not None:
            self._load_checkpoint_state(self._resume_ckpt)
            del self._resume_ckpt
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Step 5: 数据加载器
        # ------------------------------------------------------------------
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size,
            sampler=sampler, num_workers=8,
        )
        if dist.get_rank() == 0:
            print(f"[Data] Dataset size: {len(dataset)}")
        self.dataloader = cycle(dataloader, sampler=sampler, start_step=self.step)

        # ------------------------------------------------------------------
        # Step 6: EMA
        # ------------------------------------------------------------------
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"[EMA] decay={ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)
            if self._resume_ema_state is not None:
                self.generator_ema.load_state_dict(self._resume_ema_state)
                if self.is_main_process:
                    print("[Resume] EMA state restored.")
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    # -----------------------------------------------------------------------
    # 断点续训：加载权重 + 优化器 + RNG
    # -----------------------------------------------------------------------
    def _get_wandb_base_name(self):
        return getattr(
            self.config,
            "wandb_name",
            getattr(self.config, "config_name", "experiment")
        )

    def _resolve_wandb_resume(self):
        """
        返回:
            run_name: 当前 run 名称
            run_id:   wandb run id（可为 None）
            resume_mode: "allow" / "never" / None
        规则:
            - always: 永远复用 checkpoint 中的 wandb_id
            - never:  永远创建新 run
            - auto:   若 checkpoint step 与远程 wandb run 的 _step 一致，则复用；
                      否则创建新 run
        """
        base_name = self._get_wandb_base_name()
        policy = getattr(self.config, "wandb_resume_policy", "auto")
        if policy not in {"auto", "always", "never"}:
            raise ValueError(f"Invalid wandb_resume_policy: {policy}")

        # 非续训：正常新建 run
        if not self.resume_path:
            return base_name, None, None

        ckpt_step = self._resume_ckpt_step
        ckpt_wandb_id = self._resume_wandb_id

        # checkpoint 里没有 wandb_id：只能新建 run
        if not ckpt_wandb_id:
            new_id = wandb.util.generate_id()
            new_name = f"{base_name}_resume_step_{self.step}"
            if self.is_main_process:
                print(f"[WandB] No wandb_id found in checkpoint. Start new run: {new_name} ({new_id})")
            return new_name, new_id, "never"

        # 强制复用旧 run
        if policy == "always":
            if self.is_main_process:
                print(f"[WandB] Resume existing run (policy=always): id={ckpt_wandb_id}")
            return base_name, ckpt_wandb_id, "allow"

        # 强制新建 run
        if policy == "never":
            new_id = wandb.util.generate_id()
            new_name = f"{base_name}_resume_step_{self.step}"
            if self.is_main_process:
                print(f"[WandB] Start new run (policy=never): {new_name} ({new_id})")
            return new_name, new_id, "never"

        # policy == auto
        entity = getattr(self.config, "wandb_entity", None)
        project = getattr(self.config, "wandb_project", "rolling_forcing")

        remote_step = None
        try:
            if entity is not None:
                api = wandb.Api()
                old_run = api.run(f"{entity}/{project}/{ckpt_wandb_id}")
                remote_step = old_run.summary.get("_step", None)
        except Exception as e:
            if self.is_main_process:
                print(f"[WandB] Failed to inspect remote run step for id={ckpt_wandb_id}: {e}")

        # 只有远程 step 和 checkpoint step 完全一致时才复用旧 run
        if remote_step is not None and remote_step == ckpt_step:
            if self.is_main_process:
                print(f"[WandB] Resume existing run (policy=auto): id={ckpt_wandb_id}, "
                      f"remote_step={remote_step}, ckpt_step={ckpt_step}")
            return base_name, ckpt_wandb_id, "allow"

        new_id = wandb.util.generate_id()
        new_name = f"{base_name}_resume_step_{self.step}"
        if self.is_main_process:
            print(f"[WandB] Start new run (policy=auto): "
                  f"remote_step={remote_step}, ckpt_step={ckpt_step}, "
                  f"name={new_name}, id={new_id}")
        return new_name, new_id, "never"

    def _load_checkpoint_state(self, ckpt):
        if self.is_main_process:
            print("[Resume] Restoring weights, optimizers and RNG...")
        fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        def _count_arch_keys(state_dict):
            if not state_dict:
                return 0
            return sum(
                ("dynamic_anchor_head" in k) or ("layer_classifiers" in k)
                for k in state_dict.keys()
            )

        if "generator" in ckpt:
            if self.is_main_process:
                arch_key_count = _count_arch_keys(ckpt["generator"])
                print(f"[Resume] Generator checkpoint arch keys: {arch_key_count}")
            with FSDP.state_dict_type(
                self.model.generator, StateDictType.FULL_STATE_DICT, fsdp_cfg
            ):
                self.model.generator.load_state_dict(ckpt["generator"])
        if "generator_optimizer" in ckpt:
            sharded = FSDP.optim_state_dict_to_load(
                self.model.generator, self.generator_optimizer,
                ckpt["generator_optimizer"],
            )
            self.generator_optimizer.load_state_dict(sharded)

        if "critic" in ckpt:
            with FSDP.state_dict_type(
                self.model.fake_score, StateDictType.FULL_STATE_DICT, fsdp_cfg
            ):
                self.model.fake_score.load_state_dict(ckpt["critic"])
        if "critic_optimizer" in ckpt:
            sharded = FSDP.optim_state_dict_to_load(
                self.model.fake_score, self.critic_optimizer,
                ckpt["critic_optimizer"],
            )
            self.critic_optimizer.load_state_dict(sharded)

        if "rng_state" in ckpt:
            _set_rng_state(ckpt["rng_state"])

        if self.is_main_process:
            print("[Resume] State restored.")

    # -----------------------------------------------------------------------
    # 保存（双轨：checkpoint_last.pt 覆盖写 + 定期历史快照）
    # -----------------------------------------------------------------------

    def save(self, is_final=False):
        if self.is_main_process:
            print(f"[Save] step={self.step} {'(FINAL)' if is_final else ''}")

        fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(
            self.model.generator, StateDictType.FULL_STATE_DICT, fsdp_cfg
        ):
            gen_sd     = self.model.generator.state_dict()
            gen_opt_sd = FSDP.optim_state_dict(self.model.generator, self.generator_optimizer)

        with FSDP.state_dict_type(
            self.model.fake_score, StateDictType.FULL_STATE_DICT, fsdp_cfg
        ):
            critic_sd     = self.model.fake_score.state_dict()
            critic_opt_sd = FSDP.optim_state_dict(self.model.fake_score, self.critic_optimizer)

        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)

            # EMA shadow dict 存在 CPU 上，不需要 FSDP allgather，直接读取
            ema_sd = None
            if self.generator_ema is not None and self.step >= self.config.ema_start_step:
                ema_sd = self.generator_ema.state_dict()

            # ---- 完整断点 checkpoint（原子覆盖写 checkpoint_last.pt）----
            resume_ckpt = {
                "generator":           gen_sd,
                "critic":              critic_sd,
                "generator_optimizer": gen_opt_sd,
                "critic_optimizer":    critic_opt_sd,
                "step":                self.step,
                "wandb_id":            wandb.run.id if wandb.run else None,
                "rng_state":           _get_rng_state(),
            }
            if ema_sd is not None:
                resume_ckpt["generator_ema"] = ema_sd

            last_path = os.path.join(self.output_path, "checkpoint_last.pt")
            tmp_path  = last_path + ".tmp"
            torch.save(resume_ckpt, tmp_path)
            if os.path.exists(last_path):
                os.remove(last_path)
            os.rename(tmp_path, last_path)
            ema_tag = " (+EMA)" if ema_sd is not None else " (EMA not started yet)"
            arch_key_count = sum(
                ("dynamic_anchor_head" in k) or ("layer_classifiers" in k)
                for k in gen_sd.keys()
            )
            print(f"[Save] Resume ckpt -> {last_path}{ema_tag} | arch_keys={arch_key_count}")


            # ---- 轻量历史快照（仅权重，推理用）----
            if (not self.config.no_save) and (
                self.step % getattr(self.config, "save_interval", 500) == 0 or is_final
            ):
                snap_dir = os.path.join(
                    self.output_path, f"checkpoint_model_{self.step:06d}"
                )
                os.makedirs(snap_dir, exist_ok=True)
                snap = {"generator": gen_sd, "critic": critic_sd}
                if ema_sd is not None:
                    snap["generator_ema"] = ema_sd
                torch.save(snap, os.path.join(snap_dir, "model.pt"))
                ema_tag = " (+EMA)" if ema_sd is not None else " (EMA not started yet)"
                print(f"[Save] Snapshot -> {snap_dir}/model.pt{ema_tag}")

    # -----------------------------------------------------------------------
    # 单步前向+反向
    # -----------------------------------------------------------------------

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.real_score.eval()
        self.model.text_encoder.eval()
        if hasattr(self.model.vae, "eval"):
            self.model.vae.eval()
        if train_generator:
            self.model.generator.train()
            self.model.fake_score.eval()
        else:
            self.model.generator.eval()
            self.model.fake_score.train()

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # 问题1：每步都根据当前 step 计算 lookahead，强制传 force_update_mask=True
        #
        # 背景：generator 和 critic 以 1:5 交替更新，两者共享同一个
        # generator.model（CausalWanModel）。若只在 generator 更新时刷新 mask，
        # 则 critic 的 5 次调用都沿用上一轮 generator 调用时缓存的旧 mask，
        # 可能跨越多个 step，造成训练目标与 lookahead 调度不一致。
        #
        # 解决方案：
        #   - 每步都从 transition_scheduler 取最新 lookahead_blocks。
        #   - 始终传 force_update_mask=True，由 causal_model._forward_train
        #     的 need_rebuild 逻辑判断是否真正重建（lookahead 值未变时复用
        #     缓存，实际开销极小）。
        # ------------------------------------------------------------------
        lookahead_blocks = 0
        force_update_mask = False
        if self.use_curriculum and self.transition_scheduler is not None:
            lookahead_blocks = self.transition_scheduler.get_lookahead_window(self.step)
            force_update_mask = True

        # 数据
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1].to(
                device=self.device, dtype=self.dtype
            )
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size
                )
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        if train_generator:
            generator_main_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
                lookahead_blocks=lookahead_blocks,
                force_update_mask=force_update_mask,
            )

            layer_reg_loss, layer_reg_stats = self._compute_layer_regularization()
            generator_loss = generator_main_loss
            if layer_reg_loss is not None:
                generator_loss = generator_loss + layer_reg_loss
                generator_log_dict.update(layer_reg_stats)

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator
            )
            generator_log_dict.update({
                "generator_loss":      generator_loss,
                "generator_main_loss": generator_main_loss.detach(),
                "generator_grad_norm": generator_grad_norm,
            })
            if layer_reg_loss is not None:
                generator_log_dict["layer_classifier/regularized_total_loss"] = generator_loss.detach()
            return generator_log_dict

        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None,
            lookahead_blocks=lookahead_blocks,
            force_update_mask=force_update_mask,
        )
        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic
        )
        critic_log_dict.update({
            "critic_loss":      critic_loss,
            "critic_grad_norm": critic_grad_norm,
        })
        return critic_log_dict

    # -----------------------------------------------------------------------
    # 可视化：解码潜变量上传 WandB（仿照 CausVid add_visualization）
    # -----------------------------------------------------------------------

    def _add_visualization(self, generator_log_dict, critic_log_dict, wandb_log):
        """
        上传 7 个视频到 WandB，与 CausVid 保持一致：

        Critic 侧（3 个）：
          critictrain_latent       生成的干净视频（generator x0 输出）
          critictrain_noisy_latent 加噪后送入 critic 的 xt（可观察噪声程度）
          critictrain_pred_image   critic 对 xt 的 x0 预测（观察 critic 拟合质量）

        Generator 侧（4 个）：
          dmdtrain_clean_latent    backward simulation 得到的 x0（generator 生成质量）
          dmdtrain_noisy_latent    加噪后送入 score 的 xt（观察 DMD 训练用的噪声水平）
          dmdtrain_pred_real_image real_score 对 xt 的 x0 预测（teacher 预测质量）
          dmdtrain_pred_fake_image fake_score 对 xt 的 x0 预测（critic 预测质量）
        """
        vae = self.model.vae

        def _decode_and_upload(latent, caption):
            """
            decode latent [B, F, 16, H, W] -> pixel [B, F, C, H, W] 范围 [-1,1]
            -> _prepare_for_wandb -> wandb.Video [B, F, C, H, W] 范围 [0,255]
            """
            try:
                with torch.no_grad():
                    pixel = vae.decode_to_pixel(latent)   # [B, F, C, H, W], [-1, 1]
                wandb_log[caption] = _prepare_for_wandb(pixel, caption=caption)
            except Exception as e:
                print(f"[Viz] Failed to decode {caption}: {e}")

        # ---- Critic 侧 3 个 ----
        for key in ("critictrain_latent", "critictrain_noisy_latent", "critictrain_pred_image"):
            if key in critic_log_dict:
                _decode_and_upload(critic_log_dict[key], key)

        # ---- Generator 侧 4 个 ----
        for key in ("dmdtrain_clean_latent", "dmdtrain_noisy_latent",
                    "dmdtrain_pred_real_image", "dmdtrain_pred_fake_image"):
            if key in generator_log_dict:
                _decode_and_upload(generator_log_dict[key], key)

    def _get_causal_generator_backbone(self):
        generator = self.model.generator
        if hasattr(generator, "module"):
            generator = generator.module
        return getattr(generator, "model", None)

    def _compute_layer_regularization(self):
        backbone = self._get_causal_generator_backbone()
        if backbone is None or not hasattr(backbone, "get_layer_classification_regularizer"):
            return None, {}

        ratio_weight = getattr(self.config, "layer_ratio_loss_weight", 0.0)
        return backbone.get_layer_classification_regularizer(
            ratio_weight=ratio_weight,
        )

    def _collect_generator_arch_stats(self):
        backbone = self._get_causal_generator_backbone()
        if backbone is None or not hasattr(backbone, "blocks"):
            return {}, {}

        stats = {}
        text_stats = {}
        stats["layer_classifier/auto_classification_active"] = 1.0 if getattr(backbone, "latest_layer_global_probs", None) is not None else 0.0
        global_probs = []
        anchor_switches = []
        anchor_lengths = []
        anchor_similarities = []

        for block_idx, block in enumerate(backbone.blocks):
            if getattr(block, "last_global_prob", None) is not None:
                prob_value = float(block.last_global_prob)
                global_probs.append(prob_value)
                stats[f"layer_classifier/prob_global_l{block_idx:02d}"] = prob_value

            anchor_head = getattr(block, "dynamic_anchor_head", None)
            if anchor_head is not None:
                anchor_switches.append(float(anchor_head.last_is_scene_change))
                anchor_lengths.append(float(anchor_head.last_anchor_length))
                if anchor_head.last_similarity is not None:
                    anchor_similarities.append(float(anchor_head.last_similarity))

        if global_probs:
            stats["layer_classifier/prob_global_mean_runtime"] = float(np.mean(global_probs))
            current_global_indices = [str(idx) for idx, prob in enumerate(global_probs) if prob >= 0.5]
            stats["layer_classifier/global_layer_count_runtime"] = float(len(current_global_indices))
            text_stats["layer_classifier/current_global_layers"] = ",".join(current_global_indices) if current_global_indices else "none"

        if anchor_switches:
            stats["dynamic_anchor/scene_change_rate"] = float(np.mean(anchor_switches))
            stats["dynamic_anchor/anchor_blocks_mean"] = float(np.mean(anchor_lengths))
        if anchor_similarities:
            stats["dynamic_anchor/block_similarity_mean"] = float(np.mean(anchor_similarities))

        return stats, text_stats

    # -----------------------------------------------------------------------
    # 主训练循环
    # -----------------------------------------------------------------------

    def train(self):
        start_step = self.step
        log_iters = getattr(self.config, "log_iters", 100)
        save_interval = getattr(self.config, "save_interval", 500)

        while True:
            # self.step 在循环开始时表示"当前正在训练第几步"（0-indexed）
            # logging 在 step += 1 之前执行，打印的 step 与训练轮次严格对齐
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
            VISUALIZE = (
                self.step % log_iters == 0
                and not self.config.no_visualize
                and self.is_main_process
            )

            # 记录本轮开始时间（训练之前）
            step_start_time = time.time()

            # ---- Generator update ----
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                batch = next(self.dataloader)
                generator_log_dict = self.fwdbwd_one_step(batch, train_generator=True)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
            else:
                generator_log_dict = {}

            # ---- Critic update ----
            self.critic_optimizer.zero_grad(set_to_none=True)
            batch = next(self.dataloader)
            critic_log_dict = self.fwdbwd_one_step(batch, train_generator=False)
            self.critic_optimizer.step()

            # 本步实际耗时（训练完成后立刻测量）
            step_elapsed = time.time() - step_start_time

            # ---- Logging（在 step += 1 之前，step 与训练轮次对齐）----
            if self.is_main_process:
                wandb_log = {}

                lookahead_blocks = 0
                if self.use_curriculum and self.transition_scheduler is not None:
                    lookahead_blocks = self.transition_scheduler.get_lookahead_window(
                        self.step
                    )

                # --- Critic scalars ---
                c_loss  = critic_log_dict["critic_loss"].mean().item()
                c_gnorm = critic_log_dict["critic_grad_norm"].mean().item()
                self.writer.add_scalar("critic_loss",      c_loss,  self.step)
                self.writer.add_scalar("critic_grad_norm", c_gnorm, self.step)
                wandb_log.update({"critic_loss": c_loss, "critic_grad_norm": c_gnorm})

                # --- Generator scalars ---
                if TRAIN_GENERATOR and generator_log_dict:
                    g_loss  = generator_log_dict["generator_loss"].mean().item()
                    g_gnorm = generator_log_dict["generator_grad_norm"].mean().item()
                    g_dmd   = generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                    self.writer.add_scalar("generator_loss",         g_loss,  self.step)
                    self.writer.add_scalar("generator_grad_norm",    g_gnorm, self.step)
                    self.writer.add_scalar("dmdtrain_gradient_norm", g_dmd,   self.step)
                    wandb_log.update({
                        "generator_loss":         g_loss,
                        "generator_grad_norm":    g_gnorm,
                        "dmdtrain_gradient_norm": g_dmd,
                    })
                    if "generator_main_loss" in generator_log_dict:
                        g_main = generator_log_dict["generator_main_loss"].mean().item()
                        self.writer.add_scalar("generator_main_loss", g_main, self.step)
                        wandb_log["generator_main_loss"] = g_main
                    if "layer_classifier/regularization_loss" in generator_log_dict:
                        reg_loss = generator_log_dict["layer_classifier/regularization_loss"].mean().item()
                        self.writer.add_scalar("layer_classifier/regularization_loss", reg_loss, self.step)
                        wandb_log["layer_classifier/regularization_loss"] = reg_loss
                    for key, value in generator_log_dict.items():
                        if not key.startswith("layer_classifier/") or key == "layer_classifier/regularization_loss":
                            continue
                        if torch.is_tensor(value):
                            value = value.mean().item()
                        elif not isinstance(value, (int, float)):
                            continue
                        self.writer.add_scalar(key, value, self.step)
                        wandb_log[key] = value

                    arch_stats, arch_text_stats = self._collect_generator_arch_stats()
                    for key, value in arch_stats.items():
                        self.writer.add_scalar(key, value, self.step)
                    wandb_log.update(arch_stats)
                    for key, value in arch_text_stats.items():
                        self.writer.add_text(key, value, self.step)
                    wandb_log.update(arch_text_stats)

                # --- 课程式 mask 参数 ---
                if self.use_curriculum:
                    self.writer.add_scalar(
                        "curriculum/lookahead_blocks", lookahead_blocks, self.step
                    )
                    wandb_log["curriculum/lookahead_blocks"] = lookahead_blocks

                # --- 每步耗时（TensorBoard + WandB）---
                self.writer.add_scalar("per_iteration_time", step_elapsed, self.step)
                wandb_log["per_iteration_time"] = step_elapsed

                # --- mask 打印 ---
                if self.step % log_iters == 0:
                    mask_type = "lookahead" if lookahead_blocks > 0 else "causal"
                    print(
                        f"[Step {self.step}] mask={mask_type} "
                        f"lookahead_blocks={lookahead_blocks} | "
                        f"time={step_elapsed:.2f}s | "
                        f"critic_loss={c_loss:.4f}"
                        + (f" | gen_loss={g_loss:.4f}"
                           if TRAIN_GENERATOR and generator_log_dict else "")
                    )

                # --- WandB 可视化 ---
                if VISUALIZE and not getattr(self.config, "disable_wandb", False):
                    try:
                        self._add_visualization(
                            generator_log_dict, critic_log_dict, wandb_log
                        )
                    except Exception as e:
                        print(f"[Viz] Failed at step {self.step}: {e}")

                # --- 上传 WandB ---
                if not getattr(self.config, "disable_wandb", False):
                    wandb.log(wandb_log, step=self.step)

            # ---- step 递增（logging 之后，保证打印 step 与训练轮次一致）----
            self.step += 1

            # EMA 延迟启动
            if (self.step >= self.config.ema_start_step and
                    self.generator_ema is None and self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(
                    self.model.generator, decay=self.config.ema_weight
                )

            # ---- 保存（step += 1 之后）----
            # self.step 此时等于"已完成的训练步数"
            # 满足 step % save_interval == 0 意味着刚好完成了整数倍步数的训练
            if (not self.config.no_save) and (self.step - start_step) > 0 \
                    and self.step % save_interval == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # ---- max_steps 终止 ----
            max_steps = getattr(self.config, "max_steps", 0)
            if max_steps > 0 and self.step >= max_steps:
                if self.is_main_process:
                    logging.info(f"[Train] max_steps={max_steps} reached.")
                # save() 内部用 FSDP allgather 收集权重，必须所有 rank 同时调用，
                # 不能放在 is_main_process 块里，否则其他 rank 不参与 allgather 导致超时
                self.save(is_final=True)
                dist.barrier()
                break

            # GC
            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("[GC] Running garbage collection.")
                gc.collect()
                torch.cuda.empty_cache()

        if self.is_main_process:
            print("[Train] Training finished.")
