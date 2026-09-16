# Xử lý dữ liệu mất cân bằng (Class Imbalance) trong ABSA

Tài liệu này mô tả cách pipeline hiện tại xử lý vấn đề mất cân bằng nhãn
trong bộ dữ liệu UIT-ViSFD, được triển khai ở `src/dataset.py`
(`compute_class_weights`) và `src/model.py` (`compute_loss`).

## 1. Vấn đề gốc

UIT-ViSFD là dữ liệu **đa-nhãn thưa**: mỗi comment chỉ nhắc trung bình
~2.65/10 aspect, các aspect còn lại là **None ngầm định** (không xuất hiện
trong chuỗi nhãn). Hệ quả là mọi aspect đều nghiêng mạnh về None, và các
class hiếm (Neutral, STORAGE) chỉ có vài chục mẫu trong Train:

| Aspect | None | Positive | Negative | Neutral |
|---|---|---|---|---|
| GENERAL | 2920 | 3627 | 949 | 290 |
| PERFORMANCE | 3646 | 2253 | 1496 | 391 |
| BATTERY | 4182 | 2027 | 1228 | 349 |
| FEATURES | 5144 | 785 | 1659 | 198 |
| CAMERA | 5640 | 1231 | 627 | 288 |
| PRICE | 5725 | 609 | 316 | 1136 |
| SER&ACC | 5791 | 1401 | 487 | 107 |
| DESIGN | 6408 | 999 | 302 | 77 |
| SCREEN | 6837 | 514 | 379 | 56 |
| STORAGE | 7695 | 59 | 21 | 11 |

Nếu học cross-entropy thô, mô hình chỉ cần **đoán None cho mọi thứ** là đạt
accuracy rất cao nhưng hoàn toàn vô dụng; gradient cũng bị nhánh majority
nuốt chửng, khiến class hiếm không bao giờ được học.

## 2. Giải pháp: class weight nghịch đảo tần suất (inverse frequency)

Hàm `compute_class_weights` (dataset.py:56-74) tính trọng số **riêng cho
từng aspect**, gồm 4 bước:

**Bước 1 — Đếm** (dataset.py:65-69)
Với mỗi aspect, đếm số mẫu thuộc từng class trong 4 polarity
(None/Positive/Negative/Neutral). None được tính bằng số mẫu *không nhắc*
aspect đó trong nhãn.

**Bước 2 — Chống chia cho 0** (dataset.py:70)
```python
counts = np.clip(counts, 1, None)
```
Nếu một class không có mẫu nào thì `1/counts` sẽ nổ; gán mức tối thiểu 1
để an toàn.

**Bước 3 — Nghịch đảo tần suất** (dataset.py:71)
```python
inv = 1.0 / counts
```
Class càng hiếm thì weight càng lớn. Ví dụ BATTERY: None có 4182 mẫu →
`1/4182 ≈ 0.00024`, Neutral có 349 mẫu → `1/349 ≈ 0.0029` (gấp ~12 lần).

**Bước 4 — Chuẩn hóa** (dataset.py:72)
```python
norm = inv / inv.sum() * 4
```
Chia cho tổng rồi nhân với số class (4) để **trọng số trung bình ≈ 1**,
không làm đổi scale tổng loss, chỉ tái phân bố gradient giữa các class.

Kết quả trọng số thực tế (tính theo đúng logic code, trên Train):

| Aspect | None | Positive | Negative | Neutral |
|---|---|---|---|---|
| BATTERY | 0.217 | 0.447 | 0.738 | 2.598 |
| SCREEN | 0.026 | 0.345 | 0.467 | 3.162 |
| STORAGE | 0.003 | 0.436 | 1.224 | 2.337 |
| GENERAL | 0.268 | 0.215 | 0.823 | 2.694 |

## 3. Cách áp dụng vào loss

Trong `compute_loss` (model.py:44-54), mỗi head aspect dùng một
`nn.CrossEntropyLoss(weight=...)` riêng, sau đó 10 loss được **cộng trung
bình**:

```python
loss_fn = nn.CrossEntropyLoss(weight=class_weights[aspect])  # weight của chính aspect đó
total_loss = total_loss + loss_fn(logits[:, i, :], labels[:, i])
return total_loss / len(ASPECTS)
```

Ý nghĩa: một mẫu `STORAGE#Neutral` bị nhân weight ~2.3, còn một mẫu
`STORAGE#None` (chiếm 98.8%) chỉ nhân ~0.003 → **lỗi trên mẫu hiếm "đắt"
hơn**, mô hình buộc phải quan tâm để giảm loss. Weight được tính riêng theo
từng aspect vì phân bố của chúng rất khác nhau (GENERAL 37% None so với
STORAGE 98.8%).

## 4. Hiệu quả và giới hạn

### Hiệu quả
- Trọng số trung bình ≈1 giữ nguyên scale tổng loss, các hyperparameter
  (learning rate, weight decay...) không cần điều chỉnh lại.
- Neutral/STORAGE được "phóng đại" → mô hình không còn đoán mù một chiều
  None.

### Giới hạn
1. **Không tạo ra tín hiệu mới**: STORAGE chỉ có 91 mẫu thật. Weight 2.3
   cho Neutral không thay thế được việc thiếu ví dụ — gradient của head này
   vẫn nhiễu, macro-F1 vẫn thấp và dao động giữa các lần chạy. Đây là giới
   hạn của dữ liệu, không phải của loss.
2. **Weight cực đoan có thể gây bất ổn**: với SCREEN/STORAGE, weight None
   chỉ ~0.003–0.026 → gradient của head đó gần như chỉ đến từ vài chục mẫu
   minority, dễ dao động giữa các epoch.
3. **`np.clip(counts, 1, None)` là cạm bẫy tương lai**: nếu thêm
   aspect/label mà có class không xuất hiện trong Train, class đó nhận
   weight = 1.0 (không được upweight như mong muốn) rồi bị chuẩn hóa kéo
   xuống. Hiện tại Train không bị ảnh hưởng vì class nào cũng có ≥1 mẫu,
   nhưng cần lưu ý khi adapt sang dataset mới (ví dụ MoMo).

### Điểm còn lệch
Objective (CE có class weight) chưa hoàn toàn khớp với metric dùng để chọn
checkpoint (macro-F1 trên dev). Hướng cải thiện tiềm năng: dùng focal loss,
hoặc giảm weight của None mạnh hơn để mục tiêu tối ưu trùng với tiêu chí
đánh giá.
