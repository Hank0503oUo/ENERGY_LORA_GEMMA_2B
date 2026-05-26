"""v04 LoRA training script for Colab.

Key changes from v04:
- 1 epoch (down from 2) to reduce overfitting
- lr=5e-5 (down from 1e-4) more conservative
- save_strategy=steps (every 40 steps) for checkpoint selection
- load_best_model_at_end=True with eval_loss metric
- Supports v04 processed data + v04 augment samples

Usage (from Colab):
    export DRIVE_PROJECT_DIR=/content/drive/MyDrive/energy_lora_router_v04
    python train_lora.py

Or import as module:
    import train_lora
    train_lora.main()
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


# =============================================================================
# 0. Config
# =============================================================================

DRIVE_PROJECT_DIR = Path(os.getenv("DRIVE_PROJECT_DIR", "/content/drive/MyDrive/energy_lora_router_v04"))
DATA_DIR = DRIVE_PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = DRIVE_PROJECT_DIR / "outputs" / "gemma_router_strict_v04"

TRAIN_FILE = DATA_DIR / "train_v04.jsonl"
VAL_FILE = DATA_DIR / "val_v04.jsonl"
SMOKE_FILE = DATA_DIR / "smoke_v04.jsonl"
MANIFEST_FILE = DATA_DIR / "harness_v04_manifest.json"

MODEL_ID = os.getenv("MODEL_ID", "google/gemma-4-e2b-it")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "gemma-e2b-energy-router-v04-r32-1ep")
GGUF_BASENAME = os.getenv("GGUF_BASENAME", "gemma-e2b-it-energy-router-v04-r32-Q4_K_M.gguf")

MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() in {"1", "true", "yes"}
LORA_R = int(os.getenv("LORA_R", "32"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", str(LORA_R * 2)))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0"))

NUM_TRAIN_EPOCHS = float(os.getenv("NUM_TRAIN_EPOCHS", "1"))
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "32"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "1"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-5"))

INSTALL_DEPS = os.getenv("INSTALL_DEPS", "false").lower() in {"1", "true", "yes"}
USE_WANDB = os.getenv("USE_WANDB", "false").lower() in {"1", "true", "yes"}
RUN_FULL_VAL_EVAL = os.getenv("RUN_FULL_VAL_EVAL", "true").lower() in {"1", "true", "yes"}
EXPORT_GGUF = os.getenv("EXPORT_GGUF", "false").lower() in {"1", "true", "yes"}

WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.03"))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", "40"))
EVAL_STEPS = int(os.getenv("EVAL_STEPS", "40"))
SAVE_TOTAL_LIMIT = int(os.getenv("SAVE_TOTAL_LIMIT", "3"))


# =============================================================================
# 1. Environment
# =============================================================================

def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def maybe_mount_drive() -> None:
    if os.getenv("SKIP_DRIVE_MOUNT", "false").lower() in {"1", "true", "yes"}:
        print("[env] SKIP_DRIVE_MOUNT=true")
        return
    if Path("/content/drive/MyDrive").exists():
        print("[env] Drive already mounted")
        return
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception:
        print("[env] google.colab not available; assuming local or already mounted")


def maybe_install_deps() -> None:
    if not INSTALL_DEPS:
        print("[deps] INSTALL_DEPS=false, skipping")
        return
    print("[deps] Installing dependencies (~5-10 min) ...")
    run([sys.executable, "-m", "pip", "install", "-q", "-U",
         "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
         "--index-url", "https://download.pytorch.org/whl/cu124"])
    run([sys.executable, "-m", "pip", "install", "-q", "-U",
         "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"])
    run([sys.executable, "-m", "pip", "install", "-q", "-U", "--no-deps",
         "trl", "peft", "accelerate", "bitsandbytes"])
    run([sys.executable, "-m", "pip", "install", "-q", "-U",
         "datasets", "huggingface_hub", "sentencepiece", "protobuf", "wandb"])
    print("[deps] Done. Restart runtime if asked.")


def login_services() -> None:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
    else:
        print("[auth] No HF_TOKEN; model must be public")

    global USE_WANDB
    if USE_WANDB:
        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key:
            import wandb
            wandb.login(key=wandb_key)
        else:
            os.environ["WANDB_DISABLED"] = "true"
            USE_WANDB = False


# =============================================================================
# 2. Data
# =============================================================================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            assert "messages" in item, f"{path}:{line_no} missing messages"
            roles = [m.get("role") for m in item["messages"]]
            assert roles[:2] == ["system", "user"], f"{path}:{line_no} expected system,user first; got {roles}"
            assert roles[-1] == "assistant", f"{path}:{line_no} expected assistant last; got {roles}"
            rows.append(item)
    return rows


def parse_assistant_json(item: dict[str, Any]) -> dict[str, Any]:
    target = item["messages"][-1]["content"].strip()
    if target.startswith("```"):
        target = target.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(target)


def validate_data() -> None:
    t0 = time.time()
    print("[validate] Checking files ...")
    required = [TRAIN_FILE, VAL_FILE, SMOKE_FILE, MANIFEST_FILE]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing:\n" + "\n".join(missing))

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("profile") != "router_strict":
        raise ValueError(f"Expected profile=router_strict, got {manifest.get('profile')!r}")

    train_rows = read_jsonl(TRAIN_FILE)
    val_rows = read_jsonl(VAL_FILE)
    smoke_rows = read_jsonl(SMOKE_FILE)

    total_checks = len(train_rows) + len(val_rows) + len(smoke_rows)
    checked = 0
    for split_name, rows in [("train", train_rows), ("val", val_rows), ("smoke", smoke_rows)]:
        for idx, item in enumerate(rows, 1):
            parsed = parse_assistant_json(item)
            if parsed.get("tool") != item.get("expected_tool"):
                raise ValueError(
                    f"{split_name}:{idx} expected_tool mismatch: "
                    f"{item.get('expected_tool')} != {parsed.get('tool')}"
                )
            checked += 1
            if checked % 1000 == 0 or checked == total_checks:
                print(f"[validate] {checked}/{total_checks} ...")

    elapsed = time.time() - t0
    print(f"[validate] OK ({elapsed:.1f}s)")
    print(f"  manifest: {manifest['version']} {manifest['profile']} {manifest['total']}")
    print(f"  train: {len(train_rows)} | val: {len(val_rows)} | smoke: {len(smoke_rows)}")


# =============================================================================
# 3. Model + Tokenizer
# =============================================================================

def load_model_and_tokenizer():
    from unsloth import FastModel
    print(f"[model] Loading {MODEL_ID} ...")
    t0 = time.time()
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify chat_template works
    tokenizer.apply_chat_template(
        [{"role": "user", "content": "ping"}, {"role": "assistant", "content": "{}"}],
        tokenize=False, add_generation_prompt=False,
    )
    print(f"[model] Loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def attach_lora(model):
    from unsloth import FastModel
    BAD_TOWERS = ("vision_tower", "audio_tower", "mm_projector",
                  "multi_modal_projector", "visual", "image_tower")

    try:
        sig = inspect.signature(FastModel.get_peft_model).parameters
        modality_kwargs = dict(
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
        )
        if "finetune_audio_layers" in sig:
            modality_kwargs["finetune_audio_layers"] = False
        model = FastModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            **modality_kwargs,
        )
    except Exception as exc:
        print(f"[lora] FastModel path unavailable ({exc}); falling back to filtered targets")
        wanted = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")
        keep = [
            n for n, _ in model.named_modules()
            if n.split(".")[-1] in wanted
            and not any(b in n for b in BAD_TOWERS)
        ]
        import re as _re
        regex = "^(?:" + "|".join(_re.escape(n) for n in keep) + ")$"
        model = FastModel.get_peft_model(
            model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0,
            target_modules=regex, bias="none",
            use_gradient_checkpointing="unsloth", random_state=42,
        )

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    leaked = [n for n in trainable_names if any(b in n for b in BAD_TOWERS)]
    if leaked:
        raise RuntimeError(f"Vision/audio leaked into LoRA: {leaked[:10]}")

    n_train = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    n_total = sum(p.numel() for _, p in model.named_parameters())
    print(f"[lora] r={LORA_R} alpha={LORA_ALPHA} | trainable={n_train:,}/{n_total:,} ({n_train/max(n_total,1):.2%})")
    return model


# =============================================================================
# 4. Dataset
# =============================================================================

def build_datasets(tokenizer):
    from datasets import load_dataset
    print(f"[dataset] Loading from {DATA_DIR} ...")
    dataset = load_dataset("json", data_files={
        "train": str(TRAIN_FILE),
        "validation": str(VAL_FILE),
        "smoke": str(SMOKE_FILE),
    })

    def render_batch(batch):
        texts = []
        for messages in batch["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    remove_cols = dataset["train"].column_names
    rendered = dataset.map(render_batch, batched=True, remove_columns=remove_cols)

    print(f"[dataset] Rendered: {rendered}")
    sample = rendered["train"][0]["text"]
    print(f"[dataset] Sample[0] length={len(sample)} chars, preview:\n{sample[:500]}...")
    return rendered


# =============================================================================
# 5. Training Config
# =============================================================================

def build_sft_config(num_train_examples: int):
    from trl import SFTConfig
    from unsloth import is_bfloat16_supported

    params = inspect.signature(SFTConfig.__init__).parameters
    total_steps = (num_train_examples // TRAIN_BATCH_SIZE) * NUM_TRAIN_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    kwargs: dict[str, Any] = {
        "output_dir": str(OUTPUT_DIR / "checkpoints"),
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.01,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": "cosine",
        "logging_steps": 5,
        "save_strategy": "steps",
        "save_steps": SAVE_STEPS,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "report_to": "wandb" if USE_WANDB else "none",
        "run_name": EXPERIMENT_NAME,
        "bf16": bool(is_bfloat16_supported()),
        "fp16": not bool(is_bfloat16_supported()),
    }
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "steps"
    if "eval_steps" in params:
        kwargs["eval_steps"] = EVAL_STEPS
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = MAX_SEQ_LENGTH
    if "packing" in params:
        kwargs["packing"] = False
    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"
    if "save_only_model" in params:
        kwargs["save_only_model"] = True
    if "load_best_model_at_end" in params:
        kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in params:
        kwargs["metric_for_best_model"] = "eval_loss"
    if "greater_is_better" in params:
        kwargs["greater_is_better"] = False

    config = SFTConfig(**kwargs)
    print(f"[config] epochs={NUM_TRAIN_EPOCHS} batch={TRAIN_BATCH_SIZE} accum={GRAD_ACCUM_STEPS}")
    print(f"[config] lr={LEARNING_RATE} r={LORA_R} alpha={LORA_ALPHA}")
    print(f"[config] warmup_steps={warmup_steps} total_steps~={total_steps}")
    return config


# =============================================================================
# 6. Train
# =============================================================================

def train(model, tokenizer, rendered_dataset):
    from trl import SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = build_sft_config(len(rendered_dataset["train"]))

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs: dict[str, Any] = {
        "model": model, "args": training_args,
        "train_dataset": rendered_dataset["train"],
        "eval_dataset": rendered_dataset["validation"],
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
    if "packing" in trainer_params:
        trainer_kwargs["packing"] = False

    trainer = SFTTrainer(**trainer_kwargs)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    # Mask sanity check
    try:
        sample0 = trainer.train_dataset[0]
        labels = sample0.get("labels")
        if labels is not None:
            visible = sum(1 for x in labels if x != -100)
            total = len(labels)
            ratio = visible / max(total, 1)
            print(f"[mask-sanity] visible={visible}/{total} ({ratio:.1%})")
            if visible == 0:
                raise RuntimeError("Mask hid ALL tokens - check instruction_part/response_part markers")
            if ratio > 0.7:
                print(f"[mask-sanity] WARNING: >70% visible - mask may not be working")
    except RuntimeError:
        raise
    except Exception as exc:
        print(f"[mask-sanity] Could not verify: {exc}")

    print("[train] Starting SFTTrainer.train() ...")
    t0 = time.time()
    trainer.train()
    print(f"[train] Done in {time.time()-t0:.1f}s")

    adapter_dir = OUTPUT_DIR / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[train] Adapter saved to {adapter_dir}")

    # Save training manifest
    _save_training_manifest(training_args)
    return model


def _save_training_manifest(training_args) -> None:
    """Save metadata about this training run."""
    manifest = {
        "project": "energy_lora_router_v04",
        "base_model": MODEL_ID,
        "experiment": EXPERIMENT_NAME,
        "training": {
            "method": "LoRA",
            "rank": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "epochs": NUM_TRAIN_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": TRAIN_BATCH_SIZE,
            "grad_accum": GRAD_ACCUM_STEPS,
            "max_seq_length": MAX_SEQ_LENGTH,
            "warmup_ratio": WARMUP_RATIO,
            "load_in_4bit": LOAD_IN_4BIT,
        },
        "output_dir": str(OUTPUT_DIR),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = OUTPUT_DIR / "training_manifest.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# 7. Main
# =============================================================================

def main() -> None:
    t_start = time.time()

    print("=" * 60)
    print(f"v04 Training Pipeline | {EXPERIMENT_NAME}")
    print("=" * 60)
    print(f"  MODEL_ID     : {MODEL_ID}")
    print(f"  LORA_R/ALPHA : {LORA_R}/{LORA_ALPHA}")
    print(f"  EPOCHS       : {NUM_TRAIN_EPOCHS}")
    print(f"  BATCH_SIZE   : {TRAIN_BATCH_SIZE}")
    print(f"  LR           : {LEARNING_RATE}")
    print(f"  INSTALL_DEPS : {INSTALL_DEPS}")

    maybe_mount_drive()
    maybe_install_deps()
    login_services()
    validate_data()

    model, tokenizer = load_model_and_tokenizer()
    rendered_dataset = build_datasets(tokenizer)
    model = attach_lora(model)
    model = train(model, tokenizer, rendered_dataset)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE in {total:.0f}s ({total/60:.1f}min)")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
