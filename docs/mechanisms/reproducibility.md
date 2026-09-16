# Reproducibility — Seed, Determinism và Device

Tài liệu mô tả cơ chế đảm bảo tái lập trong `notebook/finetuningBERT_con_train (1).ipynb:25` (`set_seed`) và `notebook/finetuningBERT_con_train (1).ipynb:27` (device), cùng các mâu thuẫn hiện tại.

## 1. Tổng quan

Fine-tune transformer trên dataset nhỏ (9.6k) rất nhạy với seed (đặc biệt head STORAGE chỉ 91 sample, `docs/dataset/imbalanced_data_handling.md:70`). Notebook đã implement seed khá đầy đủ, nhưng còn mâu thuẫn giữa deterministic và performance.

## 2. `set_seed`

`notebook/finetuningBERT_con_train (1).ipynb:25`:

```python
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(SEED)  # SEED=42
g = torch.Generator()
g.manual_seed(SEED)
import os
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

| Dòng | Tác dụng |
|---|---|
| `np.random.seed` | Repro cho numpy (tính class weights, shuffle) |
| `torch.manual_seed` + `cuda.manual_seed*` | Repro cho CPU và mọi GPU |
| `cudnn.deterministic=True, benchmark=False` | Bắt cuDNN chọn thuật toán deterministic thay vì benchmark nhanh nhất |
| `torch.use_deterministic_algorithms(True)` | Ép toàn bộ op PyTorch dùng path deterministic (nếu op không deterministic sẽ raise) |
| `torch.Generator().manual_seed` + `DataLoader(..., generator=g)` | Deterministic shuffle của `train_loader` (`notebook/finetuningBERT_con_train (1).ipynb:33`) |
| `PYTHONHASHSEED` | Deterministic hash của Python (ảnh hưởng `set`/`dict` order) |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | Cần cho `use_deterministic_algorithms` trên CUDA (nếu không sẽ lỗi runtime) |

## 3. Device

`notebook/finetuningBERT_con_train (1).ipynb:27`:

```python
device = DEVICE if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# DEVICE="cuda" trong config cell 7
```

* Fallback tự động về CPU nếu không có GPU — đúng. Nhưng cell 43 hard-code `predict_one(..., "cuda")` (`notebook/finetuningBERT_con_train (1).ipynb:43`) sẽ lỗi trên CPU, nên dùng biến `device`.
* Mọi tensor class_weights cũng được chuyển lên device ở cell 31: `{a: w.to(device) for a,w in class_weights.items()}`.

## 4. Mâu thuẫn hiện tại

`notebook/finetuningBERT_con_train (1).ipynb:18:train_model` có:

```python
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
```

* Dòng này **mâu thuẫn** với `set_seed` đã đặt `benchmark=False` và `deterministic=True`. `benchmark=True` cho phép cuDNN chọn thuật toán nhanh nhất theo input shape, nhưng làm kết quả non-deterministic.
* **Khuyến nghị:** nếu ưu tiên reproducibility (theo `note.md:32` ⚠️ "Seed có nhưng chưa đầy đủ"), xóa hoặc đặt `benchmark=False` trong `train_model`. Nếu ưu tiên tốc độ, bỏ `use_deterministic_algorithms(True)` và chấp nhận dao động nhỏ giữa các run (thường <0.5% F1).

## 5. Những gì còn thiếu (theo `note.md:62`)

* **Chưa lưu seed/git commit/dataset version vào checkpoint:** `save_checkpoint` chỉ lưu `hyperparams`, `history`, `best_f1` (`notebook/finetuningBERT_con_train (1).ipynb:18`). Nên thêm `seed`, `git_commit`, `data_hash` (ví dụ md5 của `Train.csv`) để truy vết.
* **Chưa log đầy đủ:** `hyperparams` ở cell 39 chỉ có `lr`, `batch_size`, `num_epochs` — thiếu `weight_decay`, `dropout`, `warmup_ratio`, `model_name`, `seed`.
* **Tokenizer/config artifact:** chưa save tokenizer cùng checkpoint (cần khi deploy, nếu vocab khác sẽ mismatch). Xem `docs/mechanisms/checkpointing.md`.

## 6. Tham chiếu

* Seed: `notebook/finetuningBERT_con_train (1).ipynb:7` (SEED), `25` (set_seed)
* Device: `notebook/finetuningBERT_con_train (1).ipynb:27`
* Training benchmark: `notebook/finetuningBERT_con_train (1).ipynb:18`
* DataLoader generator: `notebook/finetuningBERT_con_train (1).ipynb:33`
* Dự kiến refactor: `src/config.py` (lưu seed, device), `docs/mechanisms/checkpointing.md`
