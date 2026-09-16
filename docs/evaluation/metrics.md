# Evaluation — Metrics cho ABSA đa khía cạnh

Tài liệu mô tả `evaluate_detailed` trong `notebook/finetuningBERT_con_train (1).ipynb:20`, logic chọn checkpoint trong `train_model` (`notebook/finetuningBERT_con_train (1).ipynb:18`), và workflow đánh giá ở `notebook/finetuningBERT_con_train (1).ipynb:41`.

## 1. Tổng quan

ABSA trên UIT-ViSFD là **10 task 4-class song song** (mỗi aspect: None/Positive/Negative/Neutral). Đánh giá chỉ bằng accuracy sẽ vô nghĩa vì `None` chiếm 30–98% tùy aspect (`docs/dataset/imbalanced_data_handling.md:14`). Notebook triển khai 5 nhóm metric bổ sung nhau.

```
logits (N, 10, 4) → argmax → preds (N, 10) so với labels (N, 10)
  → (1) Detection (binary) per-aspect
  → (2) Sentiment 4-class macro-F1 per-aspect
  → (3) Sentiment Pos/Neg-only macro-F1
  → (4) Micro-F1 (all pairs vs mentioned-only)
  → (5) Exact-match ratio (10/10 đúng)
```

## 2. Inference cho evaluation

`notebook/finetuningBERT_con_train (1).ipynb:20:evaluate_detailed` — loop không gradient:

```python
@torch.no_grad()
def evaluate_detailed(model, loader, device, verbose=True, show_confusion=False):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        preds = logits.argmax(dim=-1).cpu()
        all_preds.append(preds); all_labels.append(batch["labels"])
    preds = torch.cat(all_preds).numpy()   # (N,10)
    labels = torch.cat(all_labels).numpy()
```

* Batch size test `EVAL_BATCH_SIZE=32` (`notebook/finetuningBERT_con_train (1).ipynb:7`).
* Không dùng AMP trong eval — không cần.

## 3. Nhóm metric chi tiết

### 3.1 Aspect Detection — Binary (Có nhắc hay không?)

```python
bin_true = (y_true != 0).astype(int)  # 0=None, 1=Positive,2=Negative,3=Neutral
bin_pred = (y_pred != 0).astype(int)
det_p = precision_score(bin_true, bin_pred, zero_division=0)
det_r = recall_score(bin_true, bin_pred, zero_division=0)
det_f1 = f1_score(bin_true, bin_pred, zero_division=0)
```

* **Ý nghĩa:** tách riêng khả năng **phát hiện aspect** khỏi phân biệt polarity. Hữu ích khi model hay nhầm `None ↔ Neutral` nhưng vẫn bắt được "có nhắc".
* **Support:** `support = (y_true != 0).sum()` — số lần aspect thực sự được nhắc, in ra trong `per_aspect` DataFrame để biết độ tin cậy metric (ví dụ STORAGE support ~11 rất thấp).

### 3.2 Sentiment 4-class Macro-F1

```python
sent_f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
```

* Trung bình F1 của 4 lớp (None/Pos/Neg/Neu) không trọng số — mỗi lớp quan trọng như nhau.
* **Vấn đề:** `None` áp đảo nên model đoán `None` tốt là đã kéo macro-F1 lên cao ("ảo cao"), che giấu khả năng phân biệt Pos/Neg thực sự. Vì vậy có thêm metric (3).

### 3.3 Sentiment Pos/Neg-only Macro-F1

```python
mask_posneg = np.isin(y_true, [1, 2])  # chỉ Pos/Neg ground-truth
posneg_f1 = f1_score(y_true[mask_posneg], y_pred[mask_posneg],
                     labels=[1,2], average="macro", zero_division=0)
```

* Chỉ tính trên sample mà ground-truth là Positive/Negative — **loại None+Neutral**. Cho biết model phân biệt Pos/Neg thật sự tốt tới đâu, không bị None làm loãng.
* Nếu aspect không có Pos/Neg nào trong split (hiếm), trả `nan` và `mean(skipna=True)` khi aggregate.

### 3.4 Micro-F1 — Gộp toàn bộ pairs

```python
micro_f1_all = f1_score(labels.flatten(), preds.flatten(), average="micro")
mask_flat = (labels.flatten() != 0) | (preds.flatten() != 0)
micro_f1_mentioned = f1_score(labels.flatten()[mask_flat], preds.flatten()[mask_flat], average="micro")
```

* **micro_f1_all:** flatten `(N,10)` thành `(N*10,)` — mỗi cặp (sample, aspect) là một prediction. Nhạy với aspect nhiều sample (GENERAL, PERFORMANCE).
* **micro_f1_mentioned_only:** chỉ tính cặp mà **ít nhất một phía** là mentioned (`!=0`). Loại bỏ cặp `None-None` đúng hàng loạt (chiếm ~73% pairs) để metric không bị inflating.

### 3.5 Exact-Match Ratio

```python
exact_match = float((preds == labels).all(axis=1).mean())
```

* Tỉ lệ sample mà **cả 10 aspect đều đúng** — metric khắt khe nhất, phản ánh model có "hiểu đúng" toàn bộ câu hay không. Thường rất thấp (ví dụ 0.2–0.4) vì chỉ cần sai 1/10 aspect là fail.

## 4. Tổng hợp và In ra

`notebook/finetuningBERT_con_train (1).ipynb:20` sau khi loop 10 aspect:

```python
df = pd.DataFrame(rows)  # rows: per-aspect dict với Detect_P/R/F1, Sentiment_F1(4cls), Sentiment_F1(Pos/Neg), Support
summary = {
    "n_samples": n_samples,
    "macro_detection_f1": df["Detect_F1"].mean(),
    "macro_sentiment_f1_4cls": df["Sentiment_F1(4cls)"].mean(),
    "macro_sentiment_f1_posneg": df["Sentiment_F1(Pos/Neg)"].mean(skipna=True),
    "micro_f1_all_pairs": micro_f1_all,
    "micro_f1_mentioned_only": micro_f1_mentioned,
    "exact_match_ratio": exact_match,
    "per_aspect": df,
    "confusion_matrices": confmats if show_confusion else None,
}
```

Khi `verbose=True` (mặc định), in bảng `per_aspect` + 6 dòng summary:

```
Macro Detection F1               : 0.xxxx
Macro Sentiment F1 (4cls)         : 0.xxxx  ← dùng để chọn checkpoint trong train
Macro Sentiment F1 (Pos/Neg only) : 0.xxxx
Micro F1 (toàn bộ cặp)            : 0.xxxx
Micro F1 (chỉ cặp có mention)     : 0.xxxx
Exact-match ratio (10/10 đúng)    : 0.xxxx
```

* `show_confusion=True` trả thêm `confusion_matrices: dict aspect → 4×4 matrix` (labels=[0,1,2,3]) để phân tích nhầm lẫn `None ↔ Neutral`.

## 5. Checkpoint Selection — Metric nào được dùng?

`notebook/finetuningBERT_con_train (1).ipynb:18:train_model`:

```python
eval_result = evaluate_detailed(model, dev_loader, device, verbose=False)
dev_f1 = eval_result["macro_sentiment_f1_4cls"]
if dev_f1 > best_f1: save_checkpoint(..., include_optimizer=False)
```

* **Hiện tại:** `macro_sentiment_f1_4cls` — macro trung bình 10 aspect 4-class.
* **Cân nhắc thay thế:** theo `docs/dataset/imbalanced_data_handling.md:111` và phân tích ở §3.2, nên thử `macro_sentiment_f1_posneg` hoặc `macro_detection_f1` để objective khớp hơn với "phát hiện đúng aspect + polarity hiếm". Có thể log cả ba và chọn theo `posneg` nếu muốn model ít đoán `None` mù.

## 6. Workflow đánh giá thực tế

`notebook/finetuningBERT_con_train (1).ipynb:41`:

```python
if CONTINUE_TRAINING:
    result = evaluate_detailed(best_model, test_loader, device, show_confusion=True)
else:
    result = evaluate_detailed(model, test_loader, device, show_confusion=True)
# result["per_aspect"].to_csv(...) để lưu
```

* Đánh giá trên `test_loader` (2726 sample) sau khi train xong, với `show_confusion=True`.
* `result["per_aspect"]` là DataFrame có thể `to_csv` để lưu báo cáo.

## 7. Các metric chưa có (theo `note.md:21-24`)

| Metric | Trạng thái | Gợi ý |
|---|---|---|
| Per-class precision/recall | Chưa — chỉ có F1 | Thêm `precision_score`/`recall_score` per polarity để biết model thiên về Pos hay Neg |
| Confusion matrix viz | Có ma trận nhưng chưa plot | Heatmap bằng seaborn, highlight `None↔Neutral` confusion |
| Error analysis | Chưa | In top misclassified samples (ví dụ `y_true=1, y_pred=2` cho BATTERY) để soi lỗi segmentation/tokenization |
| Training curves | Chưa | Plot `history` từ `train_model` (train_loss vs dev_f1) |
| Data leakage check | Chưa | Kiểm tra overlap comment giữa splits |

## 8. Tham chiếu

* Evaluation: `notebook/finetuningBERT_con_train (1).ipynb:20`
* Training selection: `notebook/finetuningBERT_con_train (1).ipynb:18`
* Imbalance context: `docs/dataset/imbalanced_data_handling.md`
* Model logits: `docs/model/architecture.md:4`
* Inference: `docs/inference/prediction.md`
* Dự kiến refactor: `src/evaluate.py` (`README.md:38`)
