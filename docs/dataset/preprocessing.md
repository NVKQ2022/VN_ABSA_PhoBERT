# Preprocessing — Label Parsing, Text Cleaning và Word Segmentation

Tài liệu mô tả pipeline tiền xử lý trong `notebook/finetuningBERT_con_train (1).ipynb:9` (`parse_label_string`, `clean_text`, `segment_text`, `preprocess_for_model`), dự kiến tách ra `src/preprocessing.py`.

## 1. Tổng quan

PhoBERT (`vinai/phobert-base-v2`) là RoBERTa được pre-train trên text **đã word-segment** bằng `pyvi` (ví dụ `điện thoại → điện_thoại`). Vì vậy mọi text phải qua `clean → segment` trước khi tokenize, nếu không vocab mismatch và performance giảm mạnh (`README.md:89`). Label string của UIT-ViSFD cũng cần parse từ dạng thô `{ASPECT#Polarity};` sang vector số.

```
CSV raw (comment, label)
  → clean_text (NFC, whitespace)
  → segment_text (pyvi ViTokenizer)
  → AutoTokenizer (PhoBERT BPE)
  → input_ids / attention_mask
Đồng thời:
  label string → parse_label_string → dict {aspect: polarity_id (0..3)}
```

## 2. Label Parsing

### 2.1 Format gốc

`data/Train.csv:1` ví dụ:

```
"{CAMERA#Positive};{FEATURES#Positive};{BATTERY#Positive};{PRICE#Positive};"
"Pin kém còn lại miễn chê ..." → {BATTERY#Negative};{GENERAL#Positive};{OTHERS};
```

* Mỗi tag là `{ASPECT#Polarity}` hoặc bare `{OTHERS}` (không có `#`, không mang polarity).
* Aspect không xuất hiện trong chuỗi → ngầm định `None`.

### 2.2 `parse_label_string`

`notebook/finetuningBERT_con_train (1).ipynb:9`:

```python
_TAG_RE = re.compile(r"\{([^}]*)\}")

def parse_label_string(label_str: str) -> dict:
    result = {aspect: POLARITY2ID["None"] for aspect in ASPECTS}  # default None=0
    if not isinstance(label_str, str):
        return result
    for tag in _TAG_RE.findall(label_str):
        if "#" not in tag:
            continue  # bỏ OTHERS
        aspect, polarity = tag.split("#", 1)
        aspect = aspect.strip(); polarity = polarity.strip()
        if aspect in result and polarity in POLARITY2ID:
            result[aspect] = POLARITY2ID[polarity]
    return result
```

* **Regex** `_TAG_RE = r"\{([^}]*)\}"` — bắt nội dung trong `{}` không tham lam.
* **Default None:** khởi tạo mọi aspect về 0, chỉ ghi đè khi tag hợp lệ.
* **Bỏ OTHERS:** `if "#" not in tag: continue` — OTHERS không có polarity, không thuộc 10 aspect chính. Nếu cần tín hiệu phụ, dùng `has_others_tag` riêng.
* **An toàn:** `strip()` trước khi lookup, kiểm tra `aspect in result and polarity in POLARITY2ID` để bỏ tag lỗi chính tả.

### 2.3 `has_others_tag`

```python
def has_others_tag(label_str: str) -> bool:
    return any(tag.strip() == "OTHERS" for tag in _TAG_RE.findall(label_str))
```

Hiện chưa dùng trong training, nhưng có thể làm feature phụ hoặc filter.

## 3. Text Cleaning — `clean_text`

`notebook/finetuningBERT_con_train (1).ipynb:9`:

```python
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

* **NFC normalize:** chuẩn hóa unicode tiếng Việt (ví dụ `e + ́ → é`), tránh tokenization khác nhau cho cùng một từ do encoding khác nhau.
* **Collapse whitespace:** `re.sub(r"\s+", " ", text)` gộp nhiều space/newline/tab thành một space — review thường có xuống dòng, emoji xen kẽ.
* **Strip:** bỏ space đầu/cuối.
* **Không làm:** không lowercasing (PhoBERT cased), không bỏ dấu, không remove emoji/punctuation — giữ nguyên sentiment signal (ví dụ `!!!`, `:(`).

## 4. Word Segmentation — `segment_text`

`notebook/finetuningBERT_con_train (1).ipynb:9`:

```python
_word_segmenter = None  # lazy-loaded

def segment_text(text: str) -> str:
    global _word_segmenter
    if _word_segmenter is None:
        from pyvi import ViTokenizer
        _word_segmenter = ViTokenizer
    return _word_segmenter.tokenize(text)
```

* **Tại sao cần:** PhoBERT vocab được build trên text đã segment bằng `pyvi` (dùng `VnCoreNLP` style, `điện thoại → điện_thoại`). Nếu không segment, tokenizer sẽ tách thành subword vô nghĩa.
* **Lazy load:** `_word_segmenter` chỉ import `pyvi` khi lần đầu gọi — tránh cost import nếu chỉ parse label hoặc test logic khác.
* **Ví dụ:**

  ```
  "Pin trâu, camera chụp đẹp" → "Pin trâu , camera chụp đẹp"
  "đăng nhập thất bại" → "đăng_nhập thất_bại"
  ```

* **Lưu ý performance:** `ViTokenizer.tokenize` là bottleneck CPU lớn nhất trong `ABSADataset.__init__` (`notebook/finetuningBERT_con_train (1).ipynb:11` có `tqdm` progress bar riêng cho bước này). Đã tối ưu bằng cách chạy **một lần duy nhất** trong `__init__`, không phải mỗi `__getitem__`.

## 5. Full Pipeline — `preprocess_for_model`

```python
def preprocess_for_model(text: str) -> str:
    return segment_text(clean_text(text))
```

* Thứ tự **clean trước, segment sau** — quan trọng vì `clean_text` chuẩn hóa whitespace/unicode trước khi `pyvi` phân đoạn từ.
* Dùng trong hai nơi:
  1. `ABSADataset.__init__` — batch segment toàn bộ `raw_texts` trước khi tokenize (`notebook/finetuningBERT_con_train (1).ipynb:11`).
  2. `predict_one` — `preprocess_for_model(clean_text(text))` cho single inference (`notebook/finetuningBERT_con_train (1).ipynb:22`). Chú ý `clean_text` bị gọi **hai lần** ở đây (thừa một lần, không hại nhưng dư).

## 6. Tích hợp với Tokenizer

`notebook/finetuningBERT_con_train (1).ipynb:29,11`:

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # vinai/phobert-base-v2
encodings = tokenizer(processed_texts, truncation=True, max_length=256,
                      padding="max_length", return_tensors="pt")
```

* `truncation=True, max_length=256` — cắt câu dài, giữ 256 token đầu (đủ cho review trung bình ~30-50 từ).
* `padding="max_length"` — pad cứng về 256 (đơn giản nhưng tốn compute; xem `docs/training/training_method.md:6` về dynamic padding).
* PhoBERT tokenizer tự thêm `<s>` và `</s>` (tương đương `[CLS]/[SEP]`).

## 7. Edge Cases và Known Issues

| Case | Xử lý hiện tại | Ghi chú |
|---|---|---|
| `label` là `NaN` (dropna) | `if not isinstance(label_str, str): return default None` | An toàn, nhưng `ABSADataset` đã `dropna(subset=["comment","label"])` nên ít gặp |
| Tag lỗi `{CAMERA#Positiv}` | Bỏ qua do `polarity not in POLARITY2ID` | Không raise, im lặng — nên log warning nếu debug |
| `comment` là `NaN` | `clean_text` trả `""` → segment `""` → tokenize ra `[CLS][SEP]` | Bị dropna nên không vào dataset |
| Text rất dài >256 | Cắt truncation, mất tail | Có thể thử sliding window nếu review dài |
| `pyvi` chưa cài | `pip install pyvi` (`notebook/finetuningBERT_con_train (1).ipynb:3`) | Yêu cầu trong `requirements.txt:5` |

## 8. Tham chiếu

* Preprocessing: `notebook/finetuningBERT_con_train (1).ipynb:9`
* Dataset tích hợp: `notebook/finetuningBERT_con_train (1).ipynb:11`
* Inference: `notebook/finetuningBERT_con_train (1).ipynb:22`
* Config label space: `notebook/finetuningBERT_con_train (1).ipynb:7`
* Dự kiến refactor: `src/preprocessing.py` (`README.md:34`)
