"""Training progress and finite-gradient checks for native LoRA SFT."""
import json
import hashlib
import time
from pathlib import Path
import torch
from transformers import TrainerCallback
from scripts.common import baseline as b

class AuditCallback(TrainerCallback):
    def __init__(self, seconds, visual_modules=()):
        self.limit = seconds
        self.started = None
        self.stopped_for_budget = False
        self.initial_samples = {}
        self.visual_modules = tuple(visual_modules)
        self.visual_gradient_steps = 0
        self.initial_visual_digest = None

    def visual_digest(self, model):
        digest = hashlib.sha256()
        for name, value in model.named_parameters():
            if value.requires_grad and 'visual.' in name:
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.started = time.monotonic()
        trainable = [{"name": n, "parameters": p.numel()} for n, p in model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in p["name"] or "audio_tower." in p["name"] or
                               ("visual." in p["name"] and not any(f'.{module}.lora_' in p['name'] for module in self.visual_modules))
                               for p in trainable):
            raise RuntimeError("Unexpected trainable modules; only language and explicitly listed visual LoRA parameters are allowed")
        if self.visual_modules:
            if not any('visual.' in p['name'] for p in trainable):
                raise RuntimeError('Declared visual LoRA did not become trainable')
            self.initial_visual_digest = self.visual_digest(model)
        self.initial_samples = {n: p.detach().flatten()[:16].float().cpu().clone()
                                for n, p in model.named_parameters() if p.requires_grad}
        b.write_json(Path(args.output_dir) / "training_details.json", {
            "trainable_parameters": sum(r["parameters"] for r in trainable),
            "total_parameters": sum(p.numel() for p in model.parameters()), "trainable_modules": trainable,
            "microbatch": args.per_device_train_batch_size, "gradient_accumulation": args.gradient_accumulation_steps,
            "effective_batch": args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size,
            "maximum_steps": state.max_steps, "epochs": args.num_train_epochs, "wall_time_limit_seconds": self.limit,
            "explicit_visual_lora_modules": list(self.visual_modules),
        })

    def on_log(self, args, state, control, logs=None, **kwargs):
        path = Path(args.output_dir) / "audit_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": state.global_step, "epoch": state.epoch, **(logs or {})}, ensure_ascii=False) + "\n")

    def on_step_end(self, args, state, control, **kwargs):
        # Preserve resumable progress before a potentially lengthy scheduled evaluation.
        if state.global_step > 0 and state.global_step % 100 == 0:
            control.should_save = True
        if self.started is not None and time.monotonic() - self.started >= self.limit:
            self.stopped_for_budget = True
            control.should_training_stop = True
            control.should_save = True
        return control

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        checks = [torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not checks or not torch.stack(checks).all().item():
            raise RuntimeError("Missing or non-finite LoRA gradients")
        if self.visual_modules:
            nonzero = [p.grad.abs().max() > 0 for n, p in model.named_parameters()
                       if p.requires_grad and 'visual.' in n and p.grad is not None]
            if nonzero and torch.stack(nonzero).any().item(): self.visual_gradient_steps += 1

    def on_train_end(self, args, state, control, model=None, **kwargs):
        updates = sum(not torch.equal(self.initial_samples[n], p.detach().flatten()[:16].float().cpu())
                      for n, p in model.named_parameters() if n in self.initial_samples)
        if updates == 0:
            raise RuntimeError("Training ended without any observed LoRA parameter updates")
        visual_changed = self.visual_modules and self.visual_digest(model) != self.initial_visual_digest
        if self.visual_modules and (not visual_changed or self.visual_gradient_steps == 0):
            raise RuntimeError('Visual LoRA did not receive real gradients and parameter updates')
        b.write_json(Path(args.output_dir) / "training_completion.json", {
            "global_step": state.global_step, "epoch": state.epoch,
            "seconds": time.monotonic() - self.started, "stopped_for_time_budget": self.stopped_for_budget,
            "best_checkpoint": state.best_model_checkpoint, "best_eval_loss": state.best_metric,
            "sampled_parameter_tensors_updated": updates,
            "visual_parameter_update_verified": bool(visual_changed),
            "visual_nonzero_gradient_steps": self.visual_gradient_steps,
        })
