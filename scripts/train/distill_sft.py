"""Native LLaMAFactory continuing-LoRA SFT with recorded teacher/control inputs."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import time

import torch
from transformers import AutoTokenizer
import yaml

from scripts.common import baseline as b
from scripts.common.distill_tasks import target_kind
from scripts.data.prepare_distillation import DATA, validate_cache
from scripts.train.callbacks import AuditCallback


class DistillationAudit(AuditCallback):
    def __init__(self, seconds, config, resumed, variant, stop_after=None, visual_modules=(), freeze_language_lora=False):
        super().__init__(seconds, visual_modules=visual_modules)
        self.config, self.resumed, self.variant = config, resumed, variant
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], local_files_only=True, use_fast=False)
        self.counts = Counter()
        self.gradient_steps = 0
        self.hook = None
        self.initial_exact_match = False
        self.stop_after = stop_after
        self.freeze_language_lora = freeze_language_lora

    def freeze_language(self, model):
        if self.freeze_language_lora:
            for name, value in model.named_parameters():
                if 'lora_' in name: value.requires_grad_('visual.' in name)

    def on_init_end(self, args, state, control, model=None, **kwargs):
        # Before Trainer creates its optimizer, include only visual parameters.
        self.freeze_language(model)
        return control

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.freeze_language(model)
        super().on_train_begin(args, state, control, model=model, **kwargs)
        if not self.resumed:
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            actual = get_peft_model_state_dict(model)
            expected = load_file(str(Path(self.config["adapter_name_or_path"]) / "adapter_model.safetensors"))
            if set(actual) != set(expected) or any(not torch.equal(actual[k].detach().cpu(), expected[k]) for k in actual):
                raise ValueError("Native LF did not initialize the actor exactly from the selected SFT adapter")
            self.initial_exact_match = True
        self.hook = model.register_forward_pre_hook(self.audit_input, with_kwargs=True)

    def audit_input(self, module, args, kwargs):
        labels = kwargs.get("labels")
        if labels is None:
            return
        if labels.shape[0] != 1:
            raise ValueError("Input audit requires microbatch 1")
        target = json.loads(self.tokenizer.decode(labels[0][labels[0] != -100].tolist(), skip_special_tokens=True).strip())
        ids, attended = kwargs["input_ids"][0], kwargs["attention_mask"][0] > 0
        video = bool(((ids == self.tokenizer.convert_tokens_to_ids("<|VIDEO|>")) & attended).any())
        audio = bool(((ids == self.tokenizer.convert_tokens_to_ids("<|AUDIO|>")) & attended).any())
        if torch.any(labels[0][~attended] != -100):
            raise ValueError("Masked dummy media/padding entered supervised loss")
        if video and "pixel_values_videos" not in kwargs or audio and "input_features" not in kwargs:
            raise ValueError("Actual media were lost before the language-model forward")
        self.counts[target_kind(target, video, audio)] += 1

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        super().on_pre_optimizer_step(args, state, control, model=model, **kwargs)
        self.gradient_steps += 1

    def on_step_end(self, args, state, control, **kwargs):
        # Respect the declared save_steps; only force an extra save at the time limit.
        if self.started is not None and time.monotonic() - self.started >= self.limit:
            self.stopped_for_budget = True
            control.should_training_stop = True
            control.should_save = True
        if self.stop_after is not None and state.global_step >= self.stop_after:
            control.should_training_stop = True
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        desired = {200, 400, 800, 1200, state.max_steps}
        if state.global_step not in desired:
            return
        source = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        out = Path(args.output_dir).parent / "candidate_adapters" / f"{self.variant}_{state.global_step}"
        if out.exists():
            return
        temporary = out.with_name("." + out.name + ".tmp")
        temporary.mkdir(parents=True, exist_ok=True)
        for name in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json"):
            shutil.copy2(source / name, temporary / name)
        b.write_json(temporary / "candidate.json", {"variant": self.variant, "global_step": state.global_step,
                     "epoch": state.epoch, "native_source": str(source), "optimizer_state_included": False})
        temporary.rename(out)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        super().on_train_end(args, state, control, model=model, **kwargs)
        if self.hook:
            self.hook.remove()
        expected = json.loads((Path(self.config['dataset_dir']) / "preparation.json").read_text())["datasets"][self.config["dataset"]]["task_counts"]
        complete = (not self.config.get("max_samples") and not self.stopped_for_budget
                    and state.epoch is not None and state.epoch >= .999)
        if complete and not self.resumed and dict(self.counts) != expected:
            raise ValueError("Native training did not cover the declared task counts")
        if self.freeze_language_lora:
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            original = load_file(str(Path(self.config['adapter_name_or_path']) / 'adapter_model.safetensors'))
            actual = get_peft_model_state_dict(model)
            if any(not torch.equal(actual[k].detach().cpu(), v) for k, v in original.items() if '.visual.' not in k):
                raise ValueError('Frozen language adapter weights changed')
        b.write_json(Path(args.output_dir) / "input_and_gradient_audit.json", {
            "native_LLaMAFactory_SFT": True, "initial_adapter_exact_match": self.initial_exact_match,
            "resumed": self.resumed, "actual_task_counts_this_invocation": dict(self.counts),
            "expected_epoch_task_counts": expected, "completed_epoch": complete,
            "all_attended_media_match_task": True, "dummy_tokens_excluded_from_loss": True,
            "finite_gradient_steps_this_invocation": self.gradient_steps,
            "global_step": state.global_step, "epoch": state.epoch,
            "declared_stop_after": self.stop_after,
            "language_lora_frozen_and_bitwise_preserved": self.freeze_language_lora,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-after", type=int, help="Stop at a matched-exposure comparison without changing the LR schedule")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--adapter", type=Path, help="Override the starting adapter without changing preprocessing")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--train-vision", action="store_true", help="Allow only visual LoRA targets from a verified expanded initialization")
    parser.add_argument("--freeze-language-lora", action="store_true", help="Optimize only the added visual adapters")
    parser.add_argument("--verify-max-pixels", type=int, help="Reduced media budget for the two-update verification only")
    parser.add_argument("--verify-only", action="store_true",
                        help="Run two real updates on 64 source records before the full token cache is ready")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this executor")
    torch.set_num_threads(8)
    config = yaml.safe_load(args.config.read_text())
    if args.freeze_language_lora and not args.train_vision:
        parser.error('--freeze-language-lora requires --train-vision')
    if args.adapter: config['adapter_name_or_path'] = str(args.adapter.resolve())
    if args.learning_rate is not None:
        if args.learning_rate <= 0: parser.error('--learning-rate must be positive')
        config['learning_rate'] = args.learning_rate
    visual_modules = ()
    if args.train_vision:
        initial = Path(config['adapter_name_or_path'])
        metadata = json.loads((initial / 'initialization.json').read_text())
        if not (initial / 'native_verification.json').exists():
            raise ValueError('Run the native expanded-adapter verification first')
        if hashlib.sha256((initial / 'adapter_model.safetensors').read_bytes()).hexdigest() != metadata['initialized_sha256']:
            raise ValueError('Expanded initialization weights changed')
        visual_modules = metadata['visual_modules']
        config['freeze_vision_tower'] = False
    if args.verify_max_pixels is not None and not args.verify_only:
        parser.error('--verify-max-pixels is restricted to --verify-only')
    if args.verify_only:
        if args.resume or not args.output_dir:
            parser.error("--verify-only requires a separate --output-dir and cannot resume")
        config.update(max_samples=64, max_steps=2, tokenized_path=None, preprocessing_num_workers=1)
        if args.verify_max_pixels is not None:
            if args.verify_max_pixels < 3136: parser.error('Pixel budget is too small')
            config.update(video_max_pixels=args.verify_max_pixels, image_max_pixels=args.verify_max_pixels)
    else:
        if Path(config['dataset_dir']).resolve() == DATA.resolve():
            validate_cache(config)
        elif config['dataset'] == 'vision_replay':
            from scripts.data.prepare_vision_replay import validate_cache as validate_replay_cache
            validate_replay_cache(config)
        else:
            from scripts.data.prepare_vision_focus import validate_cache as validate_visual_cache
            validate_visual_cache(config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir.resolve())
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint
        checkpoint = get_last_checkpoint(config["output_dir"])
        if not checkpoint: raise ValueError("No native checkpoint to resume")
        config["resume_from_checkpoint"] = checkpoint
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    b.write_json(out / "effective_config.json", config)
    b.write_json(out / "environment.json", {
        "python": os.sys.executable, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
        "gpu_free_memory_gib_before_loading": torch.cuda.mem_get_info()[0] / 2**30,
        "packages": {n: version(n) for n in ("torch", "transformers", "peft", "trl", "llamafactory")},
        "starting_adapter_sha256": hashlib.sha256((Path(config["adapter_name_or_path"]) / "adapter_model.safetensors").read_bytes()).hexdigest(),
    })
    b.write_json(out / "process.json", {"pid": os.getpid(), "status": "running", "started_unix": time.time()})
    variant = ('vision_frozen_language' if args.freeze_language_lora else 'vision') if args.train_vision else "teacher" if config["dataset"] == "distill_train" else "control"
    if args.verify_only: variant = 'verification_' + variant
    from llamafactory.train.tuner import run_exp
    run_exp(config, callbacks=[DistillationAudit(args.max_hours * 3600, config, args.resume, variant, args.stop_after,
                                               visual_modules, args.freeze_language_lora)])
    b.write_json(out / "process.json", {"pid": os.getpid(), "status": "complete", "finished_unix": time.time()})


if __name__ == "__main__":
    main()
