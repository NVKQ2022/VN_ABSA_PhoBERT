# Dataset — ABSADataset, DataLoader và Class Weights

Tài liệu mô tả `ABSADataset` và `compute_class_weights` trong `notebook/finetuningBERT_con_train (1).ipynb:11`, dự kiến tách ra `src/dataset.py`, cùng cách `DataLoader` được khởi tạo ở `notebook/finetuningBERT_con_train (1).ipynb:33`.

## 1. Tổng quan

UIT-ViSFD được chia sẵn thành `data/Train.csv` (9631), `data/Dev.csv` (1360), `data/Test.csv` (2726) (đã trừ header). Mỗi dòng có `comment` (text) và `label` (chuỗi `{ASPECT#Polarity};`). Dataset cần biến chúng thành `input_ids/attention_mask/labels` cho multi-head model.

```
CSV → clean_text → parse_label_string → segment_text → tokenizer → tensors (N, 256)/(N, 10)
```

Thiết kế cốt lõi: **cache toàn bộ trong `__init__`**, `__getitem__` chỉ index — tối ưu tốc độ train.

## 2. `ABSADataset` — Eager Caching

### 2.1 Khởi tạo (`__init__`)

`notebook/finetuningBERT_con_train (1).ipynb:11`:

```python
def __init__(self, csv_path, tokenizer, max_len=256, segment=True, show_progress=True):
    df = pd.read_csv(csv_path)
    assert {"comment", "label"}.issubset(df.columns)
    df = df.dropna(subset=["comment", "label"]).reset_index(drop=True)

    raw_texts = df["comment"].apply(clean_text).tolist()
    label_dicts = df["label"].apply(parse_label_string).tolist()

    if segment:
        iterator = tqdm(raw_texts, desc="Word-segmenting (pyvi)") if show_progress else raw_texts
        processed_texts = [segment_text(t) for t in iterator]
    else:
        processed_texts = raw_texts

    encodings = tokenizer(processed_texts, truncation=True, max_length=max_len,
                          padding="max_length", return_tensors="pt")
    self.input_ids = encodings["input_ids"]            # (N, max_len)
    self.attention_mask = encodings["attention_mask"]  # (N, max_len)
    self.labels = torch.tensor([[d[a] for a in ASPECTS] for d in label_dicts], dtype=torch.long)  # (N, 10)
```

* **Dropna + assert:** đảm bảo mọi sample có đủ `comment`/`label`.
* **Segmentation là bottleneck:** loop `segment_text` từng câu, có `tqdm` progress bar. Với Train ~9.6k sample, bước này chiếm >80% thời gian `__init__`, nhưng chỉ chạy **một lần**.
* **Batch tokenize:** `tokenizer(processed_texts, ...)` một lần cho toàn bộ dataset — nhanh hơn nhiều so với tokenize từng câu trong `__getitem__` (nhờ vectorized Rust backend của HuggingFace).
* **Trade-off:** tốn RAM để giữ `(N, 256)` int64 + `(N,10)` labels (~9.6k × 256 × 4B ≈ 10MB + overhead) — rất đáng đổi lấy tốc độ. Comment gốc trong code: *"vài chục MB — rất đáng đánh đổi"*.

### 2.2 `__len__` và `__getitem__`

```python
def __len__(self): return self.input_ids.shape[0]

def __getitem__(self, idx):
    return {"input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]}
```

* Cực nhẹ — chỉ indexing tensor, không còn `clean/segment/tokenize` mỗi step.
* **Known bug:** `get_info()` (`notebook/finetuningBERT_con_train (1).ipynb:11:get_info`) truy cập `self.label_dicts` và `self.texts` nhưng `__init__` mới không lưu chúng (đã tối ưu bỏ để tiết kiệm RAM). Kết quả `AttributeError` nếu gọi `dataset.get_info()`. Fix: lưu thêm `self.label_dicts = label_dicts` và `self.texts = raw_texts` hoặc tính lại từ `self.labels`.

### 2.3 So với bản cũ (commented out)

Bản cũ trong cùng cell (`notebook/finetuningBERT_con_train (1).ipynb:11` phần comment) tokenize **lười** trong `__getitem__`, mỗi batch phải segment+tokenize lại — chậm, không tận dụng batch encode, và tốn CPU trong DataLoader workers.

## 3. `compute_class_weights` — Inverse Frequency

`notebook/finetuningBERT_con_train (1).ipynb:11:compute_class_weights` — chi tiết đã có tại `docs/dataset/imbalanced_data_handling.md`, tóm tắt ngắn ở đây:

```python
def compute_class_weights(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=["label"])
    label_dicts = df["label"].apply(parse_label_string).tolist()
    weights = {}
    for aspect in ASPECTS:
        counts = np.zeros(4)
        for d in label_dicts: counts[d[aspect]] += 1
        counts = np.clip(counts, 1, None)
        inv = 1.0 / counts
        norm = inv / inv.sum() * 4
        weights[aspect] = torch.tensor(norm, dtype=torch.float)
    return weights
```

* Tính **riêng cho từng aspect** (vì phân bố khác nhau: STORAGE 98% None vs GENERAL 30% None).
* Chuẩn hóa `*4` để trung bình weight ≈1, không đổi scale loss.
* Trả về `dict aspect → FloatTensor(4,)` — được chuyển lên device ở `notebook/finetuningBERT_con_train (1).ipynb:31: class_weights = {a: w.to(device) for a,w in class_weights.items()}`.

## 4. DataLoader

`notebook/finetuningBERT_con_train (1).ipynb:33`:

```python
train_ds = ABSADataset(TRAIN_CSV, tokenizer)
test_ds  = ABSADataset(TEST_CSV, tokenizer)
dev_ds   = ABSADataset(DEV_CSV, tokenizer)

train_loader = DataLoader(train_ds, batch_size=16, num_workers=2, pin_memory=True, shuffle=True, generator=g)
dev_loader   = DataLoader(dev_ds,   batch_size=16, num_workers=2, pin_memory=True, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32)
```

* **Batch size:** Train/Dev 16, Test 32 (`EVAL_BATCH_SIZE`).
* **Shuffle:** Train `shuffle=True` + `generator=g` (seeded) — đúng. Dev `shuffle=True` là **thừa/gây non-deterministic** cho validation, nên `False`.
* **num_workers=2:** tăng throughput nhưng cần `if __name__=="__main__"` guard khi chạy `src/train.py` script (nếu không sẽ lỗi spawn trên Windows/Colab).
* **pin_memory=True:** kết hợp `non_blocking=True` trong `train_model` để copy async lên GPU.

## 5. Data Splits và Leakage

* Splits là **official UIT-ViSFD** — không tự split lại, tránh leakage do random.
* **Chưa có data leakage check** (`note.md:25` ❌) — nên thêm kiểm tra overlap `comment` giữa Train/Dev/Test (ví dụ hash exact match hoặc near-duplicate via TF-IDF cosine) và báo cáo trong `docs/dataset/dataset.md` hoặc notebook EDA.

## 6. Các cơ chế chưa có (theo `note.md:27`)

| Cơ chế | Trạng thái | Gợi ý |
|---|---|---|
| Dynamic padding | Chưa — `padding="max_length"` cứng | Dùng `DataCollatorWithPadding(tokenizer)` + `padding="longest"` trong DataLoader collate_fn, giảm ~30% FLOPs |
| Class distribution viz | Chưa | Plot histogram per-aspect (None/Pos/Neg/Neu) từ `compute_class_weights` |
| `get_info` fix | Bug như trên | Lưu `self.label_dicts` hoặc recompute từ `self.labels` |
| `test_ds.texts` | Bug — cell 42 `test_ds.texts[i]` sẽ lỗi | Lưu `self.texts` hoặc dùng `df["comment"]` gốc |

## 7. Tham chiếu

* Dataset: `notebook/finetuningBERT_con_train (1).ipynb:11`
* Preprocessing: `docs/dataset/preprocessing.md`, `notebook/finetuningBERT_con_train (1).ipynb:9`
* Imbalance: `docs/dataset/imbalanced_data_handling.md`
* DataLoader: `notebook/finetuningBERT_con_train (1).ipynb:33`
* Config: `notebook/finetuningBERT_con_train (1).ipynb:7` (`BATCH_SIZE`, `MAX_LEN`, `SEED`)
* Dự kiến refactor: `src/dataset.py` (`README.md:35`)
