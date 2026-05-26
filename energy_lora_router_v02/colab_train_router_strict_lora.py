# Colab script: first-round router-strict LoRA for the NTU energy assistant.
#
# Recommended runtime:
#   - Google Colab Pro
#   - A100 if available
#   - GPU runtime enabled
#
# Required Google Drive files:
#   MyDrive/energy_lora_router_v02/data/harness_v02_train.jsonl
#   MyDrive/energy_lora_router_v02/data/harness_v02_val.jsonl
#   MyDrive/energy_lora_router_v02/data/harness_v02_smoke.jsonl
#   MyDrive/energy_lora_router_v02/data/harness_v02_manifest.json
#
# Run:
#   python /content/drive/MyDrive/energy_lora_router_v02/colab_train_router_strict_lora.py
#
# Notes:
#   - This first round is JSON-only tool routing + safety refusal.
#   - Do not mix explainer_sft.jsonl or legacy_cleaned.jsonl into this run.

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# =============================================================================
# 0. User config
# =============================================================================

DRIVE_PROJECT_DIR = Path(os.getenv("DRIVE_PROJECT_DIR", "/content/drive/MyDrive/energy_lora_router_v02"))
DATA_DIR = DRIVE_PROJECT_DIR / "data"
OUTPUT_DIR = DRIVE_PROJECT_DIR / "outputs" / "gemma_router_strict_v02"

TRAIN_FILE = DATA_DIR / "harness_v02_train.jsonl"
VAL_FILE = DATA_DIR / "harness_v02_val.jsonl"
SMOKE_FILE = DATA_DIR / "harness_v02_smoke.jsonl"
MANIFEST_FILE = DATA_DIR / "harness_v02_manifest.json"

# Change MODEL_ID here if Hugging Face uses a different final Gemma 4 E2B repo id.
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-4-e2b-it")

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "gemma-e2b-energy-router-strict-v02")
GGUF_BASENAME = os.getenv("GGUF_BASENAME", "gemma-4-e2b-it-energy-router-v02-Q4_K_M.gguf")

MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() in {"1", "true", "yes"}
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", str(LORA_R)))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0"))  # 0 = Unsloth fast-patch path

NUM_TRAIN_EPOCHS = float(os.getenv("NUM_TRAIN_EPOCHS", "3"))
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "8"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "2"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))

INSTALL_DEPS = os.getenv("INSTALL_DEPS", "true").lower() in {"1", "true", "yes"}
USE_WANDB = os.getenv("USE_WANDB", "true").lower() in {"1", "true", "yes"}
RUN_FULL_VAL_EVAL = os.getenv("RUN_FULL_VAL_EVAL", "true").lower() in {"1", "true", "yes"}
EXPORT_GGUF = os.getenv("EXPORT_GGUF", "false").lower() in {"1", "true", "yes"}


# =============================================================================
# 1. Environment setup
# =============================================================================

def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def maybe_mount_drive() -> None:
    if os.getenv("SKIP_DRIVE_MOUNT", "false").lower() in {"1", "true", "yes"}:
        print("SKIP_DRIVE_MOUNT=true; assuming Drive is already mounted.")
        return
    if Path("/content/drive/MyDrive").exists() or DRIVE_PROJECT_DIR.exists():
        print("Drive appears mounted; skipping drive.mount().")
        return
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        print("google.colab not detected; assuming Drive is already mounted or using local paths.")
        return
    drive.mount("/content/drive")


def maybe_install_deps() -> None:
    if not INSTALL_DEPS:
        print("[deps] INSTALL_DEPS=false, skipping pip install (saves ~5-10 min)")
        return
    print("[deps] Installing dependencies (this will take 5-10 min) ...")
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    ])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "--no-deps",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
    ])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "datasets",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "wandb",
    ])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "torch==2.10.0",
        "torchvision==0.25.0",
        "torchaudio==2.10.0",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
    ])
    print("Dependencies installed. If Colab asks for a runtime restart, restart and rerun with INSTALL_DEPS=false.")


def login_services() -> None:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)
    else:
        print("No HF_TOKEN env var found. If the model is gated, run huggingface_hub.notebook_login() manually.")

    global USE_WANDB
    if USE_WANDB:
        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key:
            import wandb

            wandb.login(key=wandb_key)
        else:
            print("USE_WANDB=true but WANDB_API_KEY is not set. Disabling wandb to avoid interactive prompt.")
            os.environ["WANDB_DISABLED"] = "true"
            os.environ["USE_WANDB"] = "false"
            USE_WANDB = False  # also patch module-level constant so build_sft_config() agrees


# =============================================================================
# 2. Data validation
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
            assert roles[-1] == "assistant", f"{path}:{line_no} expected assistant target last; got {roles}"
            rows.append(item)
    return rows


def parse_assistant_json(item: dict[str, Any]) -> dict[str, Any]:
    target = item["messages"][-1]["content"].strip()
    if target.startswith("```"):
        target = target.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(target)


def validate_drive_files() -> None:
    import time as _time
    _t0 = _time.time()
    print("[validate] Checking Drive files...")

    required = [TRAIN_FILE, VAL_FILE, SMOKE_FILE, MANIFEST_FILE]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required Drive files:\n" + "\n".join(missing))

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("profile") != "router_strict":
        raise ValueError(f"Expected manifest profile=router_strict, got {manifest.get('profile')!r}")

    print(f"[validate] Reading train.jsonl ...")
    _t1 = _time.time()
    train_rows = read_jsonl(TRAIN_FILE)
    print(f"[validate] Read {len(train_rows)} train rows in {_time.time()-_t1:.1f}s")

    print(f"[validate] Reading val.jsonl ...")
    _t2 = _time.time()
    val_rows = read_jsonl(VAL_FILE)
    print(f"[validate] Read {len(val_rows)} val rows in {_time.time()-_t2:.1f}s")

    print(f"[validate] Reading smoke.jsonl ...")
    _t3 = _time.time()
    smoke_rows = read_jsonl(SMOKE_FILE)
    print(f"[validate] Read {len(smoke_rows)} smoke rows in {_time.time()-_t3:.1f}s")

    total_checks = len(train_rows) + len(val_rows) + len(smoke_rows)
    checked = 0
    for split_name, rows in [("train", train_rows), ("val", val_rows), ("smoke", smoke_rows)]:
        for idx, item in enumerate(rows, 1):
            parsed = parse_assistant_json(item)
            if parsed.get("tool") != item.get("expected_tool"):
                raise ValueError(
                    f"{split_name}:{idx} expected_tool mismatch: "
                    f"{item.get('expected_tool')} != assistant tool {parsed.get('tool')}"
                )
            checked += 1
            if checked % 500 == 0 or checked == total_checks:
                print(f"[validate] Validated {checked}/{total_checks} samples...")

    elapsed = _time.time() - _t0
    print(f"[validate] Data OK ({elapsed:.1f}s)")
    print("manifest:", manifest["version"], manifest["profile"], manifest["total"])
    print("train:", len(train_rows), Counter(x.get("category", "unknown") for x in train_rows))
    print("val:", len(val_rows), Counter(x.get("difficulty", "unknown") for x in val_rows))
    print("smoke:", len(smoke_rows))


# =============================================================================
# 3. Model + tokenizer
# =============================================================================

def load_model_and_tokenizer():
    import time as _time
    print(f"[model] Downloading/loading {MODEL_ID} (first run may take several minutes) ...")
    _t0 = _time.time()
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    print(f"[model] Base model loaded in {_time.time()-_t0:.1f}s")

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Gemma-4 E2B tokenizer ships with a usable chat_template (turn markers
    # like `<|turn>user`). We refuse to silently overwrite it with the gemma-3
    # template (`<start_of_turn>...`) because that would silently mis-align
    # train vs inference. If the built-in template ever breaks, fail loudly
    # so a human can decide which template fits Gemma-4.
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "ping"}, {"role": "assistant", "content": "{}"}],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Gemma-4 tokenizer's built-in chat_template failed to render. "
            "Refusing to fall back to the gemma-3 template because its turn "
            "markers (<start_of_turn>) do not match Gemma-4 (<|turn>...). "
            "Inspect the tokenizer manually before continuing. "
            f"Original error: {exc}"
        ) from exc

    return model, tokenizer


def attach_lora(model):
    """Attach LoRA only to language-tower modules.

    Gemma-3n / Gemma-4 E2B are multimodal (text + vision + audio).
    A bare `target_modules=["q_proj", ...]` list matches every q_proj in the
    model, including ones inside `vision_tower` / `audio_tower`, which we do
    NOT want to fine-tune for a router task. We try Unsloth's modality flags
    first, then fall back to an explicit name filter.
    """
    from unsloth import FastLanguageModel

    BAD_TOWERS = ("vision_tower", "audio_tower", "mm_projector",
                  "multi_modal_projector", "visual", "image_tower")

    # --- Path 1: newer Unsloth exposes modality flags via FastModel.
    try:
        from unsloth import FastModel  # type: ignore
        get_peft = FastModel.get_peft_model
        modality_kwargs = dict(
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
        )
        # Gemma 3n / 4 also have audio_tower; only pass kwarg if the API supports it
        sig = inspect.signature(get_peft).parameters
        if "finetune_audio_layers" in sig:
            modality_kwargs["finetune_audio_layers"] = False
        model = get_peft(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,  # 0 keeps Unsloth's fast patching path
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            **modality_kwargs,
        )
    except Exception as exc:
        # --- Path 2: classic FastLanguageModel + manual filter on module names.
        print(f"[attach_lora] FastModel path unavailable ({exc}); falling back to filtered target_modules.")
        wanted = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")
        target_modules = sorted({
            name.split(".")[-1]
            for name, mod in model.named_modules()
            if name.split(".")[-1] in wanted
            and not any(bad in name for bad in BAD_TOWERS)
        })
        # If we got here we still need name-level filtering, so PEFT's
        # `target_modules` (which only matches by suffix) is not enough.
        # Use the regex form to anchor the path.
        import re as _re
        keep = [
            name for name, _ in model.named_modules()
            if name.split(".")[-1] in wanted
            and not any(bad in name for bad in BAD_TOWERS)
        ]
        regex = "^(?:" + "|".join(_re.escape(n) for n in keep) + ")$"
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            target_modules=regex,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    # Final guard — if anything multimodal still leaked, fail loudly.
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    leaked = [n for n in trainable_names if any(b in n for b in BAD_TOWERS)]
    if leaked:
        raise RuntimeError(f"Vision/audio/mmproj params leaked into LoRA targets: {leaked[:10]}")

    n_train = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    n_total = sum(p.numel() for _, p in model.named_parameters())
    print(f"[attach_lora] trainable params = {n_train:,} / {n_total:,} ({n_train/max(n_total,1):.2%})")
    return model


# =============================================================================
# 4. Dataset rendering
# =============================================================================

def build_datasets(tokenizer):
    from datasets import load_dataset

    print(f"[dataset] Loading train/val/smoke from {DATA_DIR} ...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VAL_FILE),
            "smoke": str(SMOKE_FILE),
        },
    )

    def render_batch(batch: dict[str, Any]) -> dict[str, list[str]]:
        texts = []
        for messages in batch["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    remove_cols = dataset["train"].column_names
    rendered = dataset.map(render_batch, batched=True, remove_columns=remove_cols)

    print("[dataset] Rendered dataset:", rendered)
    print("[dataset] First rendered training sample:")
    print(rendered["train"][0]["text"][:1200])
    return rendered


# =============================================================================
# 5. Training
# =============================================================================

def build_sft_config():
    from trl import SFTConfig
    from unsloth import is_bfloat16_supported

    params = inspect.signature(SFTConfig.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(OUTPUT_DIR / "checkpoints"),
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.01,
        "warmup_steps": 3,
        "lr_scheduler_type": "cosine",
        "logging_steps": 5,
        "save_strategy": "epoch",
        # Keep more checkpoints so load_best_model_at_end can roll back to the
        # best epoch (in v0.2 we lost epoch-3-best because save_total_limit=2
        # only kept the last 2 = the most over-fit ones).
        "save_total_limit": 5,
        "report_to": "wandb" if USE_WANDB else "none",
        "run_name": EXPERIMENT_NAME,
        "bf16": bool(is_bfloat16_supported()),
        "fp16": not bool(is_bfloat16_supported()),
    }
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = MAX_SEQ_LENGTH
    if "packing" in params:
        kwargs["packing"] = False
    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"
    if "save_only_model" in params:
        kwargs["save_only_model"] = True
    # Auto-rollback to the val-best checkpoint at end of training. In v0.2
    # we observed train_loss → 0.007 with val_loss flat at 3.14, classic
    # over-fit; the val-best checkpoint was epoch 3 but got pruned.
    if "load_best_model_at_end" in params:
        kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in params:
        kwargs["metric_for_best_model"] = "eval_loss"
    if "greater_is_better" in params:
        kwargs["greater_is_better"] = False

    return SFTConfig(**kwargs)


def train(model, tokenizer, rendered_dataset):
    from trl import SFTTrainer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = build_sft_config()

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
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

    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    # Sanity-check: confirm the response-only mask actually targets the
    # assistant span. Visible-loss tokens should be roughly 5%-25% of the
    # sample (the JSON tool-call). 0% means the mask hid everything (loss
    # won't move), 100% means the markers didn't match and masking is a
    # no-op. Either case we want to know BEFORE burning A100 hours.
    try:
        sample0 = trainer.train_dataset[0]
        labels = sample0.get("labels")
        if labels is not None:
            visible = sum(1 for x in labels if x != -100)
            total = len(labels)
            ratio = visible / max(total, 1)
            print(f"[mask-sanity] sample 0: visible_loss_tokens={visible}/{total} ({ratio:.1%})")
            if visible == 0:
                raise RuntimeError(
                    "train_on_responses_only masked EVERY token (loss would be 0). "
                    "The marker strings probably don't match Gemma-4's tokenization. "
                    "Inspect tokenizer.apply_chat_template output and adjust "
                    "instruction_part / response_part."
                )
            if ratio > 0.7:
                print(
                    "[mask-sanity] WARNING: >70% of tokens are visible — "
                    "the response-only mask probably didn't take effect. "
                    "Training will still run, but loss will include the system "
                    "prompt and user query, hurting JSON learning signal."
                )
        else:
            print("[mask-sanity] sample 0 has no 'labels' key; cannot verify masking. Continuing.")
    except RuntimeError:
        raise
    except Exception as exc:
        print(f"[mask-sanity] could not verify mask (non-fatal): {exc}")

    trainer.train()

    adapter_dir = OUTPUT_DIR / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print("Adapter saved to:", adapter_dir)
    return model


# =============================================================================
# 6. In-memory smoke/val evaluation
# =============================================================================

def parse_tool_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {"tool": "__parse_error__", "arguments": {}, "raw": text[:300]}


def generate_tool_call(model, tokenizer, messages: list[dict[str, str]]) -> str:
    import torch

    prompt_messages = [m for m in messages if m["role"] in {"system", "user"}]

    # === 關鍵修正 ===
    # 之前用 inner text_tok 自己的 chat_template，可能跟訓練時 processor 用的
    # template 不同，導致 inference 渲染格式不對、LoRA 無法觸發。
    # 這裡改成「強制」把 processor 的 chat_template 同步到 inner tokenizer，
    # 確保 inference 渲染跟訓練 build_datasets 完全一致。
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    outer_tmpl = getattr(tokenizer, "chat_template", None)
    if outer_tmpl is not None:
        text_tok.chat_template = outer_tmpl  # 無條件覆寫

    text = text_tok.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 一次性 debug：印第一筆樣本的渲染結果，肉眼確認是不是 <bos><|turn>system 開頭
    if not getattr(generate_tool_call, "_debug_printed", False):
        print("=" * 60)
        print("[infer-render] FIRST inference prompt (first 600 chars):")
        print(text[:600])
        print("=" * 60)
        print("[infer-render] should start with: <bos><|turn>system")
        print("[infer-render] should end with:   <|turn>model\\n")
        print("=" * 60)
        generate_tool_call._debug_printed = True

    enc = text_tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to("cuda")
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=160,
            temperature=0.0,
            do_sample=False,
            use_cache=True,
        )

    new_tokens = outputs[0][input_ids.shape[-1]:]
    return text_tok.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_split(model, tokenizer, path: Path, output_name: str, max_samples: int | None = None) -> dict[str, Any]:
    from unsloth import FastModel

    FastModel.for_inference(model)
    rows = read_jsonl(path)
    if max_samples:
        rows = rows[:max_samples]

    results = []
    correct = 0
    malformed = 0
    by_diff: dict[str, dict[str, int]] = {}

    for idx, item in enumerate(rows, 1):
        raw = generate_tool_call(model, tokenizer, item["messages"])
        parsed = parse_tool_response(raw)
        predicted = parsed.get("tool", "__parse_error__")
        expected = item.get("expected_tool")
        difficulty = item.get("difficulty", "unknown")

        if predicted == "__parse_error__":
            malformed += 1
        if predicted == expected:
            correct += 1

        bucket = by_diff.setdefault(difficulty, {"total": 0, "correct": 0, "malformed": 0})
        bucket["total"] += 1
        bucket["correct"] += int(predicted == expected)
        bucket["malformed"] += int(predicted == "__parse_error__")

        results.append({
            "idx": idx,
            "sample_id": item.get("sample_id"),
            "difficulty": difficulty,
            "query": next((m["content"] for m in item["messages"] if m["role"] == "user"), ""),
            "expected_tool": expected,
            "predicted_tool": predicted,
            "is_correct": predicted == expected,
            "raw": raw,
        })

        mark = "OK" if predicted == expected else "MISS"
        print(f"[{mark}] {idx}/{len(rows)} expected={expected} predicted={predicted} diff={difficulty}")

    report = {
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / max(1, len(rows)),
        "malformed": malformed,
        "malformed_rate": malformed / max(1, len(rows)),
        "by_difficulty": by_diff,
    }

    eval_dir = OUTPUT_DIR / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = eval_dir / f"{output_name}.jsonl"
    out_json = eval_dir / f"{output_name}_summary.json"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Eval report:", json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved:", out_jsonl)
    return report


# =============================================================================
# 7. Merge + GGUF export
# =============================================================================

def export_outputs(model, tokenizer) -> None:
    merged_dir = OUTPUT_DIR / "merged_16bit"
    gguf_dir = OUTPUT_DIR / "gguf_q4_k_m"

    local_merged = Path("/content/merged_16bit_tmp")
    print("Saving merged 16-bit model to:", local_merged)
    local_merged.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(local_merged),
        tokenizer,
        save_method="merged_16bit",
    )

    if not EXPORT_GGUF:
        print("EXPORT_GGUF=false; skipping GGUF. Merged model is at:", local_merged)
        print("Copy to Drive manually if needed:  shutil.copytree(...)")
        return

    print("Exporting GGUF Q4_K_M to:", local_merged)
    model.save_pretrained_gguf(
        str(local_merged),
        tokenizer,
        quantization_method="Q4_K_M",
    )

    ggufs = sorted(local_merged.rglob("*.gguf"))
    if not ggufs:
        print("No GGUF file found after export. Check Unsloth logs.")
        return

    named_dir = OUTPUT_DIR / "final_gguf"
    named_dir.mkdir(parents=True, exist_ok=True)
    for i, gguf_path in enumerate(ggufs):
        suffix = f"_shard{i}" if len(ggufs) > 1 else ""
        dest = named_dir / f"{GGUF_BASENAME.replace('.gguf', suffix + '.gguf')}"
        if i == 0:
            dest = named_dir / GGUF_BASENAME
        shutil.copy2(gguf_path, dest)
        print(f"GGUF shard {i}: {dest}")
    print("Final named GGUF directory:", named_dir)


# =============================================================================
# 8. Main
# =============================================================================

def main() -> None:
    import time as _time
    _t_start = _time.time()

    print("Training config:")
    print(f"  MODEL_ID={MODEL_ID}")
    print(f"  MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}")
    print(f"  LOAD_IN_4BIT={LOAD_IN_4BIT}")
    print(f"  LORA_R={LORA_R}  LORA_ALPHA={LORA_ALPHA}  LORA_DROPOUT={LORA_DROPOUT}")
    print(f"  NUM_TRAIN_EPOCHS={NUM_TRAIN_EPOCHS}")
    print(f"  TRAIN_BATCH_SIZE={TRAIN_BATCH_SIZE}")
    print(f"  GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS}")
    print(f"  EFFECTIVE_BATCH_SIZE={TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"  LEARNING_RATE={LEARNING_RATE}")
    print(f"  INSTALL_DEPS={INSTALL_DEPS}  RUN_FULL_VAL_EVAL={RUN_FULL_VAL_EVAL}  EXPORT_GGUF={EXPORT_GGUF}")

    maybe_mount_drive()
    maybe_install_deps()
    login_services()
    validate_drive_files()

    _t1 = _time.time()
    print(f"\n[STEP 1/6] Loading model + tokenizer ({MODEL_ID}) ...")
    model, tokenizer = load_model_and_tokenizer()
    print(f"[STEP 1/6] Model loaded in {_time.time()-_t1:.1f}s")

    _t2 = _time.time()
    print(f"[STEP 2/6] Building datasets ...")
    rendered_dataset = build_datasets(tokenizer)
    print(f"[STEP 2/6] Datasets ready in {_time.time()-_t2:.1f}s")

    _t3 = _time.time()
    print(f"[STEP 3/6] Attaching LoRA (r={LORA_R}, alpha={LORA_ALPHA}) ...")
    model = attach_lora(model)
    print(f"[STEP 3/6] LoRA attached in {_time.time()-_t3:.1f}s")

    _t4 = _time.time()
    print(f"[STEP 4/6] Training (epochs={NUM_TRAIN_EPOCHS}, batch={TRAIN_BATCH_SIZE}) ...")
    model = train(model, tokenizer, rendered_dataset)
    print(f"[STEP 4/6] Training done in {_time.time()-_t4:.1f}s")

    print("[STEP 5/6] Smoke evaluation ...")
    evaluate_split(model, tokenizer, SMOKE_FILE, "smoke_after_train")

    if RUN_FULL_VAL_EVAL:
        print("[STEP 6/6] Full validation evaluation ...")
        evaluate_split(model, tokenizer, VAL_FILE, "val_after_train")

    export_outputs(model, tokenizer)

    total = _time.time() - _t_start
    print(f"\n{'='*60}")
    print(f"DONE. Total time: {total:.0f}s ({total/60:.1f}min)")
    print(f"Outputs are under: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
