# Configuration — Hyperparameters và Label Space

Tài liệu mô tả central config trong `notebook/finetuningBERT_con_train (1).ipynb:7`, dự kiến `src/config.py` (`README.md:33`), và cách config được sử dụng xuyên suốt pipeline.

## 1. Tổng quan

Mọi hằng số (paths, label maps, model, training) tập trung trong một cell config duy nhất (`notebook/finetuningBERT_con_train (1).ipynb:7`). Khi refactor sang `src/`, cell này sẽ thành `src/config.py` được import bởi `preprocessing.py`, `dataset.py`, `model.py`, `train.py`.

## 2. Paths

```python
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
# TRAIN_CSV = os.path.join(DATA_DIR, "Train.csv")
DATA_URL = "https://raw.githubusercontent.com/NVKQ2022/VN-ABSA/main/data"
TRAIN_CSV = f"{DATA_URL}/Train.csv"
DEV_CSV = f"{DATA_URL}/Dev.csv"
TEST_CSV = f"{DATA_URL}/Test.csv"
```

* **Hiện tại (notebook):** load trực tiếp từ GitHub raw URL (`NVKQ2022/VN-ABSA`). Tiện cho Colab nhưng phụ thuộc mạng.
* **Dự kiến (src):** dùng `PROJECT_ROOT/data/Train.csv` local (`data/Train.csv:1` đã có 9631 dòng). Nên hỗ trợ cả hai: thử local trước, fallback URL nếu không tồn tại.

## 3. Label Space

```python
ASPECTS = ["BATTERY", "CAMERA", "DESIGN", "FEATURES", "GENERAL",
           "PERFORMANCE", "PRICE", "SCREEN", "SER&ACC", "STORAGE"]
POLARITY2ID = {"None": 0, "Positive": 1, "Negative": 2, "Neutral": 3}
ID2POLARITY = {v: k for k, v in POLARITY2ID.items()}
NUM_POLARITY_CLASSES = 4
NUM_ASPECTS = 10
```

* **ASPECTS order là contract:** `ABSADataset.labels` (`notebook/finetuningBERT_con_train (1).ipynb:11`), `PhoBertABSA.heads` (`notebook/finetuningBERT_con_train (1).ipynb:13`), `compute_loss` (`notebook/finetuningBERT_con_train (1).ipynb:15`), `evaluate_detailed` (`notebook/finetuningBERT_con_train (1).ipynb:20`), và `predict_one` (`notebook/finetuningBERT_con_train (1).ipynb:22`) đều iterate theo `ASPECTS` — đổi order sẽ mismatch checkpoint cũ.
* **POLARITY2ID:** `None=0` đặc biệt — vừa là "không nhắc" vừa là một class trong CE loss và metric 4-class. Các metric detection nhị phân (`y != 0`) dựa trên quy ước này.

## 4. Model Config

```python
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LEN = 256
DROPOUT = 0.2
```

* `MODEL_NAME` quyết định tokenizer và encoder — đổi sang `xlm-roberta-base` sẽ bỏ được word segmentation (`docs/dataset/preprocessing.md:4`).
* `MAX_LEN=256` đủ cho review dài nhất UIT-ViSFD (~100 từ). Tăng lên 512 sẽ tốn VRAM gấp ~2× do attention O(n²).
* `DROPOUT` chỉ áp dụng sau pooling (`docs/model/architecture.md:3`).

## 5. Training Config

```python
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 4e-6
NUM_EPOCHS = 5
WARMUP_RATIO = 0.15
WEIGHT_DECAY = 0.01
SEED = 42
GRAD_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 3
DEVICE = "cuda"
CONTINUE_TRAINING = True
```

| Param | Ý nghĩa | Ghi chú |
|---|---|---|
| `BATCH_SIZE` | Train batch | 16 nhỏ, có thể tăng effective batch qua `grad_accum_steps` (`docs/training/training_method.md:3.2`) |
| `EVAL_BATCH_SIZE` | Eval batch | 32 lớn hơn vì không cần backward |
| `LEARNING_RATE` | Peak LR cho AdamW | 4e-6 rất nhỏ, phù hợp fine-tune; warmup 15% sẽ ramp từ 0 lên 4e-6 |
| `NUM_EPOCHS` | Số epoch tối đa | 5 ngắn, early stopping thường dừng sớm hơn |
| `WARMUP_RATIO` | Tỉ lệ warmup steps | 0.15 × total_steps |
| `WEIGHT_DECAY` | AdamW decay | 0.01 chuẩn transformer |
| `SEED` | Seed toàn cục | Dùng trong `set_seed` (`docs/mechanisms/reproducibility.md`) |
| `GRAD_CLIP_NORM` | Clip norm | 1.0 trong `train_model` |
| `EARLY_STOPPING_PATIENCE` | Số epoch không cải thiện thì dừng | 3 |
| `DEVICE` | Thiết bị | `"cuda"` với fallback `if torch.cuda.is_available() else "cpu"` |
| `CONTINUE_TRAINING` | Có resume từ HF checkpoint không | `True` → load `NVKQ2022/VN_ABSA_PhoBERT` từ HuggingFace (`notebook/finetuningBERT_con_train (1).ipynb:35`) |

## 6. Cách sử dụng trong pipeline

* **Preprocessing:** `ASPECTS`, `POLARITY2ID` cho `parse_label_string`; `MAX_LEN` cho tokenizer.
* **Dataset:** `MAX_LEN`, `ASPECTS`, `POLARITY2ID` cho `ABSADataset` và `compute_class_weights`.
* **Model:** `MODEL_NAME`, `DROPOUT`, `ASPECTS`, `NUM_POLARITY_CLASSES` cho `PhoBertABSA`.
* **Training:** toàn bộ training config cho `train_model`, `DataLoader`, `AdamW`, scheduler.
* **Checkpoint:** `hyperparams` dict ở cell 39 chỉ lưu subset `{lr, batch_size, num_epochs}` — nên mở rộng lưu full config (xem `docs/mechanisms/checkpointing.md:6`).

## 7. Tham chiếu

* Config: `notebook/finetuningBERT_con_train (1).ipynb:7`
* Preprocessing: `docs/dataset/preprocessing.md`
* Dataset: `docs/dataset/dataset.md`
* Model: `docs/model/architecture.md`
* Training: `docs/training/training_method.md`
* Dự kiến refactor: `src/config.py` (`README.md:33`), `README.md:93` (swap MODEL_NAME)
