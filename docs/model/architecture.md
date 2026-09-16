# Model Architecture — PhoBertABSA (Multi-Head)

Tài liệu mô tả kiến trúc mô hình ABSA được triển khai trong notebook `notebook/finetuningBERT_con_train (1).ipynb:13` (`PhoBertABSA`) và dự kiến tách ra `src/model.py`.

## 1. Tổng quan

Bài toán UIT-ViSFD là **multi-aspect, multi-polarity** nhưng mỗi aspect độc lập:

* 10 aspect cố định (`notebook/finetuningBERT_con_train (1).ipynb:7` → `config.ASPECTS`):

  ```
  BATTERY, CAMERA, DESIGN, FEATURES, GENERAL,
  PERFORMANCE, PRICE, SCREEN, SER&ACC, STORAGE
  ```
* Mỗi aspect có 4 trạng thái (`notebook/finetuningBERT_con_train (1).ipynb:7` → `POLARITY2ID`):

  ```python
  {"None": 0, "Positive": 1, "Negative": 2, "Neutral": 3}
  ```

Thay vì một head 40-way (10×4) hoặc CRF phức tạp, mô hình dùng **shared encoder + 10 head độc lập** — cùng học biểu diễn ngôn ngữ chung, nhưng quyết định polarity riêng cho từng aspect.

```
Raw text
  → clean_text + segment_text (pyvi)
  → PhoBERT tokenizer (MAX_LEN=256)
  → vinai/phobert-base-v2 encoder (hidden_size=768)
  → pooled [CLS] (768-dim)
  → Dropout(0.2)
  → 10 × Linear(768 → 4)   # ModuleDict heads[aspect]
  → logits (B, 10, 4)
```

`README.md:22` tóm tắt đúng luồng này.

## 2. Encoder — `vinai/phobert-base-v2`

* `notebook/finetuningBERT_con_train (1).ipynb:7` đặt `MODEL_NAME = "vinai/phobert-base-v2"` (RoBERTa-base cho tiếng Việt, vocab BPE, pre-train trên 20GB text VN).
* Trong `PhoBertABSA.__init__` (`notebook/finetuningBERT_con_train (1).ipynb:13`):

  ```python
  self.encoder = AutoModel.from_pretrained(model_name)
  hidden_size = self.encoder.config.hidden_size  # 768
  ```

* PhoBERT **yêu cầu word segmentation** trước khi tokenize (ví dụ `đăng nhập → đăng_nhập`). Notebook xử lý qua `segment_text` trước khi gọi `AutoTokenizer.from_pretrained(MODEL_NAME)` (`notebook/finetuningBERT_con_train (1).ipynb:29`). Nếu đổi sang `xlm-roberta-base` có thể bỏ bước segmentation (`README.md:93`).

## 3. Pooling và Dropout

```python
outputs = self.encoder(input_ids, attention_mask)
if getattr(outputs, "pooler_output", None) is not None:
    pooled = outputs.pooler_output
else:
    pooled = outputs.last_hidden_state[:, 0, :]  # fallback CLS token
pooled = self.dropout(pooled)  # DROPOUT=0.2
```

* Ưu tiên `pooler_output` (CLS qua dense + tanh của RoBERTa/BERT). Fallback về `last_hidden_state[:,0]` đảm bảo tương thích với encoder không có pooler (ví dụ XLM-R).
* `DROPOUT = 0.2` (`notebook/finetuningBERT_con_train (1).ipynb:7`) áp dụng **một lần duy nhất** sau pooling, trước khi chia ra các head — giúp regularize shared representation thay vì từng head riêng.

## 4. Multi-Head Classifier

```python
self.heads = nn.ModuleDict({
    aspect: nn.Linear(hidden_size, NUM_POLARITY_CLASSES)
    for aspect in ASPECTS
})
# forward
logits = torch.stack([self.heads[a](pooled) for a in ASPECTS], dim=1)
# shape: (B, 10, 4)
```

* **Tại sao ModuleDict thay vì nn.Linear(768, 40)?** Để mỗi aspect có weight/bias riêng, dễ áp `class_weights` riêng trong loss (`notebook/finetuningBERT_con_train (1).ipynb:15`), và dễ thay thế/gỡ head khi transfer sang MoMo (9-class single-label) như mô tả ở `README.md:71`.
* **Thứ tự head cố định theo `ASPECTS`** — `torch.stack` theo đúng thứ tự list, nên index `i` trong loss khớp `ASPECTS[i]` (`notebook/finetuningBERT_con_train (1).ipynb:15:7`).
* Tổng tham số thêm: `10 × (768×4 + 4) = 30,760` — không đáng kể so với ~135M của PhoBERT encoder.

## 5. Loss — `compute_loss`

`notebook/finetuningBERT_con_train (1).ipynb:15`:

```python
def compute_loss(logits, labels, class_weights=None):
    total_loss = 0.0
    for i, aspect in enumerate(ASPECTS):
        weight = class_weights[aspect] if class_weights is not None else None
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        total_loss += loss_fn(logits[:, i, :], labels[:, i])
    return total_loss / len(ASPECTS)
```

* **Input:** `logits (B,10,4)`, `labels (B,10)` với giá trị 0..3.
* **Per-aspect weighted CE:** mỗi head có `weight` riêng từ `compute_class_weights` (`notebook/finetuningBERT_con_train (1).ipynb:11:compute_class_weights`, chi tiết tại `docs/dataset/imbalanced_data_handling.md`).
* **Average:** chia cho `len(ASPECTS)` để scale loss không phụ thuộc số aspect, giữ LR ổn định nếu thêm/bớt aspect.
* **Đặc tính:** mô hình học **jointly** — gradient từ 10 head cùng backprop qua shared encoder, encoder học representation hữu ích cho cả 10 task.

## 6. Biến thể và giới hạn hiện tại

| Vấn đề | Trạng thái trong notebook | Gợi ý cải thiện |
|---|---|---|
| Chỉ dùng CLS pooling | Done. Chưa thử mean-pooling / attention-pooling | Mean-pool trên `attention_mask` có thể tốt hơn cho câu dài |
| Không có CRF / dependency giữa aspect | Chủ ý — giữ đơn giản; thực tế aspect tương quan (ví dụ `PRICE` negative thường đi kèm `GENERAL` negative) | Thêm auxiliary loss hoặc label-correlation regularizer |
| Dropout một lớp duy nhất | `DROPOUT=0.2` sau pooling | Có thể thêm dropout trong head hoặc stochastic depth trong encoder |
| Weight sharing giữa head | Không — mỗi head độc lập hoàn toàn | Thử share một `Linear` rồi fine-tune riêng (adapter) khi data ít (STORAGE) |
| Chưa hỗ trợ focal loss | Dùng CE weighted | `docs/dataset/imbalanced_data_handling.md:111` đề xuất focal loss để khớp metric macro-F1 |

## 7. Liên kết với các thành phần khác

* **Preprocessing:** `segment_text` + `AutoTokenizer` phải dùng cùng `MODEL_NAME`, nếu không vocab mismatch (`docs/dataset/preprocessing.md`).
* **Dataset:** `ABSADataset` cache `input_ids/attention_mask/labels` đã khớp shape `(N,10)` cho loss (`docs/dataset/dataset.md`).
* **Training:** `train_model` gọi `compute_loss(logits, labels, class_weights)` trong `autocast` fp16 (`docs/training/training_method.md`).
* **Evaluation:** `evaluate_detailed` argmax trên `logits` dim=-1 để lấy `pred_ids` (`docs/evaluation/metrics.md`).
* **Inference:** `predict_one` lọc bỏ `None` trước khi trả về (`docs/inference/prediction.md`).

## 8. Tham chiếu mã nguồn

* Config: `notebook/finetuningBERT_con_train (1).ipynb:7`
* Model: `notebook/finetuningBERT_con_train (1).ipynb:13`
* Loss: `notebook/finetuningBERT_con_train (1).ipynb:15`
* Dự kiến refactor: `src/model.py`, `src/config.py` (theo `README.md:32`)
