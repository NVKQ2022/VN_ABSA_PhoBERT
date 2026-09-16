# Inference — Dự đoán trên text mới

Tài liệu mô tả `predict_one` trong `notebook/finetuningBERT_con_train (1).ipynb:22`, các cell demo `notebook/finetuningBERT_con_train (1).ipynb:42-43`, và dự kiến `src/predict.py` (`README.md:39`).

## 1. Tổng quan

Sau khi train, model trả về `logits (1, 10, 4)` cho mỗi câu. `predict_one` chuyển logits thành dict `{aspect: polarity}` chỉ giữ aspect được nhắc (bỏ `None`), để dùng cho demo, API, hoặc batch aggregation.

```
raw text → clean_text → segment_text → tokenizer (MAX_LEN=256, padding max_length)
         → model → logits (1,10,4) → argmax → filter None → {ASPECT: polarity}
```

## 2. `predict_one`

`notebook/finetuningBERT_con_train (1).ipynb:22`:

```python
@torch.no_grad()
def predict_one(text, model, tokenizer, device):
    processed = preprocess_for_model(clean_text(text))  # clean → segment
    encoding = tokenizer(processed, truncation=True, max_length=256,
                         padding="max_length", return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    logits = model(input_ids, attention_mask)  # (1, 10, 4)
    pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()  # (10,)

    results = {}
    for aspect, pred_id in zip(ASPECTS, pred_ids):
        polarity = ID2POLARITY[pred_id]  # 0→None,1→Positive,...
        if polarity != "None":
            results[aspect] = polarity
    return results
```

* **`@torch.no_grad()`:** tắt autograd, giảm VRAM và tăng tốc.
* **Preprocess:** `preprocess_for_model(clean_text(text))` — `clean_text` bị gọi hai lần (trong `preprocess_for_model` đã có `clean_text` rồi), dư nhưng không sai. Nên rút gọn thành `preprocess_for_model(text)`.
* **Tokenizer:** `truncation=True, max_length=256, padding="max_length"` — giống training để shape khớp, dù với single sample `padding="max_length"` hơi phí (có thể dùng `padding=False`).
* **Filter None:** chỉ trả aspect có polarity khác `None` — output gọn cho downstream. Nếu cần full 10 aspect (ví dụ để so sánh với ground-truth), bỏ filter.
* **Device:** `input_ids.to(device)` — cần `model` và `tokenizer` cùng device. Notebook cell 43 hard-code `"cuda"` thay vì biến `device` (`notebook/finetuningBERT_con_train (1).ipynb:43`) — nên sửa thành `device`.

## 3. Demo trong notebook

### 3.1 Predict trên test sample (cell 42)

`notebook/finetuningBERT_con_train (1).ipynb:42`:

```python
results = []
for i in range(1):
    sample_text = test_ds.texts[i]  # BUG: ABSADataset mới không có .texts
    prediction = predict_one(sample_text, model, tokenizer, device)
    results.append(prediction)
```

* **Bug:** `test_ds.texts` không tồn tại sau khi `ABSADataset` được tối ưu cache (`notebook/finetuningBERT_con_train (1).ipynb:11`). Fix: lưu `self.texts = raw_texts` trong `__init__` hoặc đọc lại từ `pd.read_csv(TEST_CSV)["comment"][i]`.
* Hàm `aggregate_results` sau đó đếm tần suất `{aspect: Counter(polarity)}` cho batch — hữu ích để thống kê distribution prediction.

### 3.2 Predict câu tự do (cell 43)

`notebook/finetuningBERT_con_train (1).ipynb:43`:

```python
predict_one("May mới mua được 1 tháng. Pin ổn. Chụp hình xấu. Cái viền máy mới đây bị tróc sơn rồi. Quá thất vọng", model, tokenizer, "cuda")
# → {'BATTERY': 'Positive', 'CAMERA': 'Negative', ...} (ví dụ)
predict_one("Hàng Sài tạm thì được không mượt mà lắm . Vi xử lý kém...", model, tokenizer, "cuda")
predict_one("Mình mua tháng từ 12/2017 đến nay dùng vẫn OK, máy chưa vấn đề j hết, pin còn 87%. Rất trâu bò!", model, tokenizer, "cuda")
```

* Ba câu test thực tế: mixed sentiment, negative performance, positive general.
* Nên bổ sung thêm ngưỡng confidence (softmax) để lọc prediction yếu, thay vì argmax mù.

## 4. Batch inference (chưa có)

Notebook chỉ có `predict_one` single-sample. Để inference hàng loạt (ví dụ đánh giá trên Test.csv hoặc serve API), nên thêm:

```python
def predict_batch(texts, model, tokenizer, device, batch_size=32):
    # preprocess batch → DataLoader → model → List[Dict]
```

Tương tự `evaluate_detailed` nhưng không cần labels, có thể kèm `softmax` để trả confidence.

## 5. Liên kết với các thành phần khác

* **Preprocessing:** `clean_text`/`segment_text`/`preprocess_for_model` (`docs/dataset/preprocessing.md`).
* **Model:** `PhoBertABSA` trả `logits (B,10,4)` (`docs/model/architecture.md`).
* **Checkpoint:** cần `model.load_state_dict(torch.load(path, map_location=device)["model_state"])` hoặc trực tiếp `state_dict` nếu lưu `torch.save(model.state_dict(), path)` như cell 47 (`notebook/finetuningBERT_con_train (1).ipynb:47`, xem `docs/mechanisms/checkpointing.md`).
* **Evaluation:** `evaluate_detailed` dùng cùng `argmax` logic nhưng giữ `None` để tính metric (`docs/evaluation/metrics.md`).

## 6. Tham chiếu

* Inference: `notebook/finetuningBERT_con_train (1).ipynb:22`, `42`, `43`
* Preprocessing: `notebook/finetuningBERT_con_train (1).ipynb:9`, `docs/dataset/preprocessing.md`
* Model: `notebook/finetuningBERT_con_train (1).ipynb:13`, `docs/model/architecture.md`
* Dự kiến refactor: `src/predict.py` (`README.md:39`)
