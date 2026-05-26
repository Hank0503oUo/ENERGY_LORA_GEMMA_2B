# Aggressive Colab Notebook Cells

這份是 notebook 專用激進訓練版。重點：**不要用 `%run`、不要 import 訓練 script**，改用 `%%bash` 開新 process，環境變數一定會在 script import 前生效。

## Cell 1 — Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
PROJECT_DIR = Path('/content/drive/MyDrive/energy_lora_router_v02')
SCRIPT_PATH = PROJECT_DIR / 'colab_train_router_strict_lora.py'
DATA_DIR = PROJECT_DIR / 'data'

assert SCRIPT_PATH.exists(), SCRIPT_PATH
assert DATA_DIR.exists(), DATA_DIR

print('PROJECT_DIR =', PROJECT_DIR)
print('SCRIPT_PATH =', SCRIPT_PATH)
```

## Cell 2 — GPU Check

```python
!nvidia-smi
```

## Cell 3 — Optional First-Time Dependency Install

第一次 runtime 才跑。跑完如果 Colab 要求 restart，就 restart 後跳過這格。

```python
!INSTALL_DEPS=true RUN_FULL_VAL_EVAL=false EXPORT_GGUF=false NUM_TRAIN_EPOCHS=0 python /content/drive/MyDrive/energy_lora_router_v02/colab_train_router_strict_lora.py
```

如果 `NUM_TRAIN_EPOCHS=0` 讓 trainer 不接受，就不要用這格，直接跑正式訓練格，讓 script 自己安裝。

## Cell 4A — A100 40G Aggressive Safe

先用這個。你原本只吃 10.2GB，這檔應該很安全。

預期 log 應該看到：

```text
Batch size per device = 24
Gradient accumulation steps = 1
Total batch size (24 x 1 x 1) = 24
```

```bash
%%bash
set -e
export DRIVE_PROJECT_DIR=/content/drive/MyDrive/energy_lora_router_v02
export MODEL_ID=google/gemma-4-e2b-it
export EXPERIMENT_NAME=gemma-e2b-energy-router-strict-v03-b24
export GGUF_BASENAME=gemma-4-e2b-it-energy-router-v03-b24-Q4_K_M.gguf

export INSTALL_DEPS=false
export USE_WANDB=false
export RUN_FULL_VAL_EVAL=true
export EXPORT_GGUF=false

export LOAD_IN_4BIT=false
export MAX_SEQ_LENGTH=2048
export LORA_R=16
export LORA_ALPHA=16
export LORA_DROPOUT=0

export NUM_TRAIN_EPOCHS=3
export TRAIN_BATCH_SIZE=24
export GRAD_ACCUM_STEPS=1
export LEARNING_RATE=2e-4

python /content/drive/MyDrive/energy_lora_router_v02/colab_train_router_strict_lora.py
```

## Cell 4B — A100 40G Very Aggressive

如果 4A VRAM 還低於 25GB，再試這個。

```bash
%%bash
set -e
export DRIVE_PROJECT_DIR=/content/drive/MyDrive/energy_lora_router_v02
export MODEL_ID=google/gemma-4-e2b-it
export EXPERIMENT_NAME=gemma-e2b-energy-router-strict-v03-b32
export GGUF_BASENAME=gemma-4-e2b-it-energy-router-v03-b32-Q4_K_M.gguf

export INSTALL_DEPS=false
export USE_WANDB=false
export RUN_FULL_VAL_EVAL=true
export EXPORT_GGUF=false

export LOAD_IN_4BIT=false
export MAX_SEQ_LENGTH=2048
export LORA_R=16
export LORA_ALPHA=16
export LORA_DROPOUT=0

export NUM_TRAIN_EPOCHS=3
export TRAIN_BATCH_SIZE=32
export GRAD_ACCUM_STEPS=1
export LEARNING_RATE=2e-4

python /content/drive/MyDrive/energy_lora_router_v02/colab_train_router_strict_lora.py
```

## Cell 4C — A100 80G / RTX6000 96G Aggressive

80G/96G 才用。目標是減少 optimizer step，快速看 router 是否起來。

```bash
%%bash
set -e
export DRIVE_PROJECT_DIR=/content/drive/MyDrive/energy_lora_router_v02
export MODEL_ID=google/gemma-4-e2b-it
export EXPERIMENT_NAME=gemma-e2b-energy-router-strict-v03-b48
export GGUF_BASENAME=gemma-4-e2b-it-energy-router-v03-b48-Q4_K_M.gguf

export INSTALL_DEPS=false
export USE_WANDB=false
export RUN_FULL_VAL_EVAL=true
export EXPORT_GGUF=false

export LOAD_IN_4BIT=false
export MAX_SEQ_LENGTH=2048
export LORA_R=16
export LORA_ALPHA=16
export LORA_DROPOUT=0

export NUM_TRAIN_EPOCHS=3
export TRAIN_BATCH_SIZE=48
export GRAD_ACCUM_STEPS=1
export LEARNING_RATE=2e-4

python /content/drive/MyDrive/energy_lora_router_v02/colab_train_router_strict_lora.py
```

## Cell 4D — 80G/96G Extreme Throughput Test

只拿來試能不能跑，不建議第一個正式結果就用這檔。batch 太大會讓 optimizer steps 太少，可能泛化比較不穩。

```bash
%%bash
set -e
export DRIVE_PROJECT_DIR=/content/drive/MyDrive/energy_lora_router_v02
export MODEL_ID=google/gemma-4-e2b-it
export EXPERIMENT_NAME=gemma-e2b-energy-router-strict-v03-b64
export GGUF_BASENAME=gemma-4-e2b-it-energy-router-v03-b64-Q4_K_M.gguf

export INSTALL_DEPS=false
export USE_WANDB=false
export RUN_FULL_VAL_EVAL=true
export EXPORT_GGUF=false

export LOAD_IN_4BIT=false
export MAX_SEQ_LENGTH=2048
export LORA_R=16
export LORA_ALPHA=16
export LORA_DROPOUT=0

export NUM_TRAIN_EPOCHS=3
export TRAIN_BATCH_SIZE=64
export GRAD_ACCUM_STEPS=1
export LEARNING_RATE=2e-4

python /content/drive/MyDrive/energy_lora_router_v02/colab_train_router_strict_lora.py
```

## Cell 5 — Confirm Output

```python
from pathlib import Path
import json

eval_dir = Path('/content/drive/MyDrive/energy_lora_router_v02/outputs/gemma_router_strict_v02/eval')
for p in sorted(eval_dir.glob('*_summary.json')):
    print('\n', p.name)
    print(p.read_text(encoding='utf-8'))
```

## 推薦選擇

先跑：

```text
Cell 4A: TRAIN_BATCH_SIZE=24, GRAD_ACCUM_STEPS=1
```

如果 log 顯示 VRAM 還很低，再中止重跑：

```text
Cell 4B: batch 32
```

只有上 80G / 96G 時才試：

```text
Cell 4C: batch 48
Cell 4D: batch 64
```

## 注意

- 看到 log 還是 `Batch size per device = 8` 就代表你沒有跑 `%%bash` cell，或跑到舊 notebook cell。
- 不要把 `MAX_SEQ_LENGTH` 降到 1024，system prompt 有完整 tool 清單，截斷會傷 router。
- 不建議開 `LOAD_IN_4BIT=true`，你現在 VRAM 不是瓶頸。
- `NUM_TRAIN_EPOCHS=3` 先看結果。若 val 還在上升，再跑 4 epoch。
