# Training Method — Vòng lặp huấn luyện, tối ưu và checkpoint

Tài liệu mô tả toàn bộ pipeline huấn luyện trong `notebook/finetuningBERT_con_train (1).ipynb:18` (`train_model`, `save_checkpoint`, `load_checkpoint`) và cell workflow `notebook/finetuningBERT_con_train (1).ipynb:37-39`.

## 1. Hyperparameters (config)

`notebook/finetuningBERT_con_train (1).ipynb:7`:

```python
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 4e-6      # best trước đó 2e-6; scheduler sẽ điều chỉnh
NUM_EPOCHS = 5
WARMUP_RATIO = 0.15
WEIGHT_DECAY = 0.01
DROPOUT = 0.2
MAX_LEN = 256
GRAD_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 3
SEED = 42
DEVICE = "cuda"           # fallback cpu nếu không có GPU
CONTINUE_TRAINING = True
```

* `LEARNING_RATE` rất nhỏ (4e-6) — phù hợp fine-tune PhoBERT đã pre-train, tránh catastrophic forgetting.
* `NUM_EPOCHS=5` ngắn, bù bằng early stopping và resume (`CONTINUE_TRAINING`).

## 2. Optimizer và Scheduler

`notebook/finetuningBERT_con_train (1).ipynb:37`:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = get_linear_schedule_with_warmup(optimizer,
    num_warmup_steps=int(WARMUP_RATIO * NUM_EPOCHS * len(train_loader)),
    num_training_steps=NUM_EPOCHS * len(train_loader))
scheduler = get_cosine_schedule_with_warmup(optimizer, ...)  # ghi đè dòng trên
```

* **Optimizer:** `AdamW` (decoupled weight decay 0.01) — chuẩn cho transformer, tách regularizer khỏi gradient momentum.
* **Scheduler — BUG hiện tại:** linear schedule bị **ghi đè ngay** bằng cosine schedule, nên thực chất chỉ chạy cosine. Cả hai đều tính:

  ```
  num_warmup_steps = 0.15 * 5 * len(train_loader)  # ~15% steps đầu warmup
  num_training_steps = 5 * len(train_loader)
  ```

  Cosine sẽ decay LR theo cosin từ peak về 0, mượt hơn linear. `note.md:45` đánh dấu "Có code thừa".
* **Khuyến nghị:** chọn **một** scheduler, hoặc để cấu hình `SCHEDULER_TYPE = "cosine" | "linear"` trong `src/config.py`. Cell 39 lặp lại bug này cho `optimizer_for_best_model`.

## 3. Vòng lặp `train_model` — thiết kế tối ưu tốc độ

`notebook/finetuningBERT_con_train (1).ipynb:18` — phiên bản cuối (132 dòng, có resume):

### 3.1 Mixed Precision (AMP)

```python
amp_enabled = use_amp and torch.cuda.is_available()
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
# trong step:
with torch.autocast(device_type="cuda" if amp_enabled else "cpu",
                     enabled=amp_enabled, dtype=torch.float16):
    logits = model(input_ids, attention_mask)
    loss = compute_loss(logits, labels, class_weights) / grad_accum_steps
scaler.scale(loss).backward()
```

* FP16 autocast + GradScaler giảm VRAM ~2× và tăng tốc 1.5–2× trên T4/V100/A100, không mất độ chính xác nhờ scaling gradient tránh underflow.
* `use_amp=True` mặc định, tự tắt nếu không có CUDA.

### 3.2 Gradient Accumulation và Clipping

```python
if (step + 1) % grad_accum_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)  # 1.0
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad()
```

* `grad_accum_steps=1` mặc định (effective batch = 16). Tăng lên 2–4 để mô phỏng batch lớn hơn mà không tốn VRAM.
* `clip_grad_norm(1.0)` chống exploding gradient, đặc biệt quan trọng khi head STORAGE có weight lớn (`docs/dataset/imbalanced_data_handling.md`).
* `scheduler.step()` **mỗi optimizer step**, không phải mỗi epoch — đúng chuẩn cho `get_*_schedule_with_warmup`.

### 3.3 Data transfer tối ưu

```python
input_ids = batch["input_ids"].to(device, non_blocking=True)
# DataLoader pin_memory=True (cell 33)
```

`pin_memory=True` + `non_blocking=True` giấu latency copy CPU→GPU sau tính toán. Yêu cầu `num_workers` phải phù hợp (hiện `num_workers=2`, cần guard `if __name__=="__main__"` nếu chạy script).

### 3.4 Logging và timing

```python
if step % log_every == 0:  # 50
    print(f"epoch {epoch} step {step}/{len(train_loader)} loss {loss:.4f}")
# cuối epoch:
avg_loss = running_loss / len(train_loader)
epoch_time = time.time() - t0
```

## 4. Validation, Early Stopping và Best Checkpoint

### 4.1 Đánh giá sau mỗi epoch

```python
eval_result = evaluate_detailed(model, dev_loader, device, verbose=False)
dev_f1 = eval_result["macro_sentiment_f1_4cls"]
history.append({"epoch": epoch, "train_loss": avg_loss,
                "dev_macro_f1": dev_f1, "epoch_time_sec": epoch_time})
```

* Metric chọn checkpoint là **macro sentiment F1 4-class** (trung bình macro-F1 của 10 aspect, đã bao gồm `None`) — chi tiết tại `docs/evaluation/metrics.md`.
* **Lưu ý:** metric này bị `None` dominance làm "ảo cao" (xem `docs/evaluation/metrics.md:3` về Pos/Neg-only F1). Có thể cân nhắc `macro_sentiment_f1_posneg` hoặc `macro_detection_f1` cho checkpoint selection.

### 4.2 Checkpoint — `_last.pt` vs `best_model.pt`

```python
# mỗi epoch:
save_checkpoint(checkpoint_path.replace(".pt", "_last.pt"),
                model, optimizer, scheduler, epoch, best_f1, history, hyperparams)
# chỉ khi cải thiện:
if dev_f1 > best_f1:
    best_f1 = dev_f1
    patience_counter = 0
    save_checkpoint(checkpoint_path, ..., include_optimizer=False)
else:
    patience_counter += 1
    if patience_counter >= early_stopping_patience: break
```

* `_last.pt` — **đầy đủ** (`model_state` + `optimizer_state` + `scheduler_state` + `history` + `hyperparams`, `include_optimizer=True`) để resume.
* `best_model.pt` — **nhẹ** (`include_optimizer=False`, chỉ `model_state`) để deploy/evaluate. Tiết kiệm ~2× dung lượng.
* `save_checkpoint` tự `os.makedirs(os.path.dirname(path), exist_ok=True)` (`notebook/finetuningBERT_con_train (1).ipynb:18:save_checkpoint`).

### 4.3 Early Stopping

* `EARLY_STOPPING_PATIENCE=3` — dừng nếu 3 epoch liên tiếp không cải thiện `dev_f1`.
* `history` lưu `train_loss`, `dev_macro_f1`, `epoch_time_sec` để vẽ training curves (chưa có trong notebook, xem `note.md:24`).

### 4.4 Resume Training

```python
if resume_from is not None and os.path.exists(resume_from):
    last_epoch, best_f1, history = load_checkpoint(resume_from, model, optimizer, scheduler, device)
    start_epoch = last_epoch + 1
    print(f"Resume từ checkpoint: epoch {start_epoch}, best_f1={best_f1:.4f}")
```

* `load_checkpoint` restore `model_state` bắt buộc, `optimizer_state`/`scheduler_state` nếu có.
* Cell 39 dùng `resume_from="/content/drive/MyDrive/absa_checkpoints/best_model_last.pt"` và `checkpoint_path="/content/drive/MyDrive/absa_checkpoints/best_model.pt"` (Colab Drive). Khi refactor sang `src/train.py`, cần đổi về `checkpoints/best_model.pt` (`README.md:54`).

## 5. Luồng workflow thực tế (cell 33-39)

```python
# cell 33 — DataLoader
train_loader = DataLoader(train_ds, batch_size=16, num_workers=2, pin_memory=True, shuffle=True, generator=g)
dev_loader   = DataLoader(dev_ds,   batch_size=16, num_workers=2, pin_memory=True, shuffle=True)  # shuffle=True là thừa
test_loader  = DataLoader(test_ds,  batch_size=32)

# cell 35 — Model + resume từ HuggingFace
model = PhoBertABSA().to(device)
if CONTINUE_TRAINING:
    checkpoint = torch.load("/content/test_model.pth", map_location="cpu")  # từ NVKQ2022/VN_ABSA_PhoBERT
    best_model = PhoBertABSA(); best_model.load_state_dict(checkpoint)

# cell 37 — Optimizer/Scheduler (bug ghi đè)
# cell 39 — train_model(best_model, train_loader, dev_loader, ..., checkpoint_path="/content/drive/...")
```

* **Vấn đề `dev_loader` shuffle:** nên `shuffle=False` để validation deterministic.
* **CONTINUE_TRAINING:** khi `True`, train tiếp trên `best_model` (đã load weight HF), không phải `model` mới khởi tạo. Optimizer/scheduler được re-init riêng cho `best_model`.

## 6. Các cơ chế chưa hoàn thiện (theo `note.md:29`)

| Cơ chế | Hiện trạng | Đề xuất |
|---|---|---|
| Dynamic padding | Chưa — `padding="max_length"` cứng 256 | Dùng `DataCollatorWithPadding` hoặc `padding="longest"` theo batch, giảm compute ~30% |
| Training curves | Chưa vẽ | Plot `history` (train_loss vs dev_f1) bằng matplotlib |
| Experiment tracking | Chưa | Thêm MLflow/W&B log `hyperparams` + `history` |
| Checkpoint đầy đủ | Một nửa — `best_model.pt` thiếu optimizer | Lưu thêm `tokenizer` + `config` (xem `docs/mechanisms/checkpointing.md`) |
| Reproducibility | Có `set_seed` nhưng `cudnn.benchmark=True` trong train mâu thuẫn `benchmark=False` | Thống nhất `benchmark=False` nếu cần deterministic |

## 7. Tham chiếu

* Config: `notebook/finetuningBERT_con_train (1).ipynb:7`
* Training loop: `notebook/finetuningBERT_con_train (1).ipynb:18` (phiên bản cuối, có `save_checkpoint`/`load_checkpoint`)
* Optimizer/Scheduler: `notebook/finetuningBERT_con_train (1).ipynb:37`
* Workflow: `notebook/finetuningBERT_con_train (1).ipynb:33`, `35`, `39`
* Loss: `docs/model/architecture.md:5`, `notebook/finetuningBERT_con_train (1).ipynb:15`
* Evaluation: `docs/evaluation/metrics.md`
* Checkpointing chi tiết: `docs/mechanisms/checkpointing.md`
