# Checkpointing — Lưu và khôi phục mô hình

Tài liệu mô tả `save_checkpoint`/`load_checkpoint` trong `notebook/finetuningBERT_con_train (1).ipynb:18`, logic best vs last trong `train_model`, và các cell lưu model `notebook/finetuningBERT_con_train (1).ipynb:45-47`.

## 1. Tổng quan

Huấn luyện PhoBERT 5 epoch trên Colab có thể bị ngắt (timeout, OOM). Notebook implement hai loại checkpoint để vừa **resume** vừa **deploy**:

* `_last.pt` — đầy đủ optimizer/scheduler/history để resume.
* `best_model.pt` — nhẹ, chỉ model_state để evaluate/deploy.

```
train_model loop
  → save_checkpoint(..._last.pt, include_optimizer=True) mỗi epoch
  → if dev_f1 > best_f1: save_checkpoint(...best_model.pt, include_optimizer=False)
```

## 2. `save_checkpoint` và `load_checkpoint`

`notebook/finetuningBERT_con_train (1).ipynb:18`:

```python
def save_checkpoint(path, model, optimizer, scheduler, epoch, best_f1,
                     history, hyperparams, include_optimizer=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {"epoch": epoch, "best_f1": best_f1, "history": history,
            "hyperparams": hyperparams, "model_state": model.state_dict()}
    if include_optimizer:
        ckpt["optimizer_state"] = optimizer.state_dict()
        ckpt["scheduler_state"] = scheduler.state_dict()
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_f1", -1.0), ckpt.get("history", [])
```

* **Path handling:** `os.makedirs(os.path.dirname(path), exist_ok=True)` — thêm trong bản cuối để tránh lỗi khi `checkpoints/` chưa tồn tại (bản đầu thiếu, `note.md:29` ⚠️ "Tạo dict nhưng chưa save").
* **include_optimizer:** `True` cho `_last.pt`, `False` cho `best_model.pt` (tiết kiệm dung lượng, không cần optimizer khi deploy).
* **map_location:** `torch.load(..., map_location=device)` đảm bảo load được trên CPU dù save trên GPU.

## 3. Best vs Last — Logic trong `train_model`

`notebook/finetuningBERT_con_train (1).ipynb:18:train_model` (sau mỗi epoch):

```python
history.append({"epoch": epoch, "train_loss": avg_loss,
                "dev_macro_f1": dev_f1, "epoch_time_sec": epoch_time})

save_checkpoint(checkpoint_path.replace(".pt", "_last.pt"),
                model, optimizer, scheduler, epoch, best_f1, history, hyperparams or {})

if dev_f1 > best_f1:
    best_f1 = dev_f1
    patience_counter = 0
    save_checkpoint(checkpoint_path, ..., include_optimizer=False)
    print(f"  -> lưu BEST checkpoint tại {checkpoint_path} (F1={best_f1:.4f})")
else:
    patience_counter += 1
    if patience_counter >= early_stopping_patience: break
```

* **_last.pt:** ghi đè mỗi epoch, luôn là epoch mới nhất (dù dev_f1 có tệ). Dùng để resume sau khi early stopping hoặc crash.
* **best_model.pt:** chỉ ghi khi `dev_f1` cải thiện (so sánh `>` không phải `>=`, tránh lưu khi bằng). Đây là checkpoint được `evaluate_detailed` trên Test và `predict_one` dùng.
* **History:** list `{"epoch", "train_loss", "dev_macro_f1", "epoch_time_sec"}` — có thể plot training curves (chưa làm, `note.md:24`).

## 4. Resume Training

```python
if resume_from is not None and os.path.exists(resume_from):
    last_epoch, best_f1, history = load_checkpoint(resume_from, model, optimizer, scheduler, device)
    start_epoch = last_epoch + 1
    print(f"Resume từ checkpoint: epoch {start_epoch}, best_f1={best_f1:.4f}")
# loop: for epoch in range(start_epoch, num_epochs+1):
```

* Cell 39 gọi `train_model(..., resume_from="/content/drive/MyDrive/absa_checkpoints/best_model_last.pt", checkpoint_path="/content/drive/MyDrive/absa_checkpoints/best_model.pt")` (`notebook/finetuningBERT_con_train (1).ipynb:39`).
* Khi refactor sang `src/train.py`, nên đổi về `checkpoints/best_model.pt` (`README.md:54`) và thêm CLI arg `--resume`.
* **Lưu ý:** `resume_from` trỏ tới `_last.pt`, không phải `best_model.pt` (vì best không có optimizer_state, không restore được scheduler).

## 5. Các cell lưu model thừa / chưa dùng

`notebook/finetuningBERT_con_train (1).ipynb:45`:

```python
checkpoint = {
    "training_hyperparameters": {"batch_size": ..., "learning_rate": ..., "num_epochs": ..., ...},
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),
}
# chưa hề torch.save(checkpoint, path) — chỉ tạo dict rồi bỏ
```

`note.md:29` đánh dấu "Tạo dict nhưng chưa save" — đúng, cell này vô hiệu.

`notebook/finetuningBERT_con_train (1).ipynb:47`:

```python
torch.save(model.state_dict(), 'phobert_absa_state_dict_best.pth')
files.download(model_path)
```

* Lưu **raw state_dict** (không bọc dict) — khác với `save_checkpoint` (bọc `{"model_state": ...}`). Khi load phải `model.load_state_dict(torch.load(path))` thay vì `ckpt["model_state"]`.
* `files.download` là Colab-specific để tải về local.
* Trên HuggingFace (`notebook/finetuningBERT_con_train (1).ipynb:35`) cũng lưu dạng raw state_dict: `phobert_absa_state_dict (3).pth`.

## 6. Những gì còn thiếu (theo `note.md:30-32`)

| Cơ chế | Trạng thái | Đề xuất |
|---|---|---|
| Tokenizer saving | Chưa | `tokenizer.save_pretrained(checkpoint_dir)` cùng với `model_state` để deploy không mismatch vocab |
| Config saving | Chưa đầy đủ — `hyperparams` chỉ có 3 key (cell 39) | Lưu full `MODEL_NAME`, `MAX_LEN`, `ASPECTS`, `POLARITY2ID`, `SEED`, `DROPOUT`, `WARMUP_RATIO`, `WEIGHT_DECAY` |
| Checkpoint đầy đủ | Một nửa | Thống nhất một format: hoặc luôn bọc `{"model_state": ...}` hoặc luôn raw, tránh nhầm khi load |
| Git commit / data version | Chưa | Thêm `git rev-parse HEAD` và hash `Train.csv` vào `hyperparams` để truy vết |

## 7. Tham chiếu

* Checkpoint: `notebook/finetuningBERT_con_train (1).ipynb:18` (`save_checkpoint`, `load_checkpoint`, `train_model`)
* Workflow: `notebook/finetuningBERT_con_train (1).ipynb:39` (hyperparams, checkpoint_path, resume_from), `45`, `47`
* Config: `notebook/finetuningBERT_con_train (1).ipynb:7`
* Training: `docs/training/training_method.md`
* Reproducibility: `docs/mechanisms/reproducibility.md`
* Dự kiến refactor: `src/train.py` (save/load), `checkpoints/` (`README.md:54`)
