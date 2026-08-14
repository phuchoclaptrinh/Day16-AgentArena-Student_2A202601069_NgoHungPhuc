# Day 16 - Giải thích 5 middleware

Tài liệu này giải thích năm TODO đã hoàn thiện trong `harness/layers/` và các
quy tắc quan trọng để bài làm chạy tốt trên cả brief công khai lẫn brief ẩn.

## Bức tranh tổng thể

Agent mắc năm lỗi có chủ ý: tin prompt injection trong tài liệu, bịa claim,
gắn sai nguồn trích dẫn, gọi tool quá ngân sách và không xử lý kết quả tool bị
hỏng. Ta sửa các lỗi này bằng middleware, không thay đổi mã đóng băng trong
`arena/` hoặc dữ liệu trong `data/`.

Thứ tự cài đặt là:

```text
InjectionGuard -> Critic -> CitationChecker -> BudgetPolicy -> Retry
```

`wrap_tool_call` chạy từ trái sang phải khi đi vào lời gọi và từ phải sang trái
khi kết quả đi ra. `after_agent` chạy theo thứ tự ngược lại. Vì vậy,
`CitationChecker` sửa nguồn trước, `Critic` loại claim không có căn cứ sau đó,
và `InjectionGuard` là lớp cuối cùng quét câu trả lời.

## 1. InjectionGuard - chống prompt injection

File: `harness/layers/injection_guard.py`

`wrap_tool_call` coi nội dung tool là dữ liệu không đáng tin. Mỗi đoạn bắt đầu
bằng `BLOCK_START` và kết thúc bằng `BLOCK_END` được thay bằng một placeholder.
Nếu tài liệu bị cắt và thiếu mốc kết thúc, toàn bộ phần từ mốc mở đến cuối chuỗi
vẫn bị loại. Vòng lặp xử lý được nhiều block trong cùng một kết quả.

`after_agent` tiếp tục xóa `INJECTION_CANARY` nếu chuỗi này còn lọt vào
`report["answer"]`. Lớp tuyệt đối không sửa `claim["text"]`, vì sửa chữ làm
mất provenance của claim.

## 2. Critic - loại bỏ nội dung bịa

File: `harness/layers/critic.py`

Một claim bình thường chỉ được giữ khi nội dung của nó xuất hiện nguyên văn
trong `ctx.observed_text`. Claim không có trong bằng chứng đã đọc sẽ bị loại.
Nếu không còn claim nào, báo cáo đặt `abstain = True`, xóa citations và trả lời
rằng chưa đủ căn cứ.

Trường hợp đặc biệt là model ghép hai nguồn mâu thuẫn bằng chuỗi `" và "`.
Critic thử từng vị trí nối thay vì tách mọi chữ “và” cùng lúc. Chỉ khi hai vế
đều là substring đã quan sát và thuộc hai tài liệu khác nhau, chúng mới được
giữ thành hai claim riêng. Đây vẫn là thao tác cắt substring từ chính câu model
đã viết, nên không phá provenance. Báo cáo đồng thời đặt `abstain = True` để
không tự ý chọn một phía của mâu thuẫn.

## 3. CitationChecker - sửa nguồn trích dẫn

File: `harness/layers/citation_checker.py`

Một citation đúng khi `claim["text"]` nằm nguyên văn trong một dòng của tài
liệu được trích. Nếu `doc_id` hiện tại sai, lớp tìm tài liệu đã được fetch đầy
đủ, có body xuất hiện trong `ctx.observed_text`, và có một dòng chứa đúng claim.
Khi tìm thấy, lớp chỉ đổi `doc_id` rồi cập nhật danh sách `citations`.

Không được quét corpus rồi gắn claim vào tài liệu chưa quan sát, vì scorer sẽ
chấm `UNRETRIEVED`. Cũng không được chỉnh dấu câu hoặc khoảng trắng trong claim.

## 4. BudgetPolicy - giữ đúng ngân sách

File: `harness/layers/budget_policy.py`

`Tools.calls` tính cả lượt `submit`, do đó mặc định phải dành lại một lượt.
Ngân sách được xem là đã cạn khi:

```python
ctx.tools.calls >= ctx.max_tool_calls - reserve
```

Khi cạn, `before_model` thêm một user message mới có `FINALIZE_SENTINEL` để yêu
cầu model chốt FINAL ngay. Danh sách mới được tạo bằng `messages + [...]`, tránh
làm lời nhắc tồn tại vĩnh viễn trong lịch sử.

`wrap_tool_call` là hàng rào thứ hai: nó từ chối gọi tool nếu chỉ còn phần ngân
sách dành cho submit. Điều này ngăn một vòng model hoặc retry làm vượt giới hạn.

## 5. Retry - phục hồi kết quả tool bị hỏng

File: `harness/layers/retry.py`

Kết quả cần thử lại khi `result.ok` là false hoặc
`is_degraded(result.content)` là true. Điều kiện thứ hai rất quan trọng vì kết
quả bị cắt hoặc nhiễu vẫn có thể mang `ok=True`.

Mỗi lần retry phải dùng đúng `name` và `args` ban đầu, tối đa ba lần tính cả lần
đầu. Trước mỗi lần gọi lại, lớp kiểm tra giới hạn để không dùng mất lượt submit.
Kết quả cuối cùng được trả nguyên trạng, kể cả khi vẫn hỏng; middleware không
được bịa dữ liệu thay cho tool.

## Quy tắc giúp giữ điểm

- Được đổi `claim["doc_id"]`.
- Được xóa claim, đặt `abstain` và sửa `answer`.
- Chỉ được cắt `claim["text"]` thành substring model đã viết.
- Không được thêm dấu câu, chuẩn hóa khoảng trắng hoặc viết lại claim.
- Chỉ gắn citation vào tài liệu thực sự đã được quan sát.
- Không dựa vào `Doc.tags`; nhãn bẫy bị xóa ở vòng chấm.
- Không hard-code `brief_id`, câu hỏi, đáp án hoặc `doc_id` công khai.

## Cách kiểm tra

Từ thư mục bài Day 16, chạy:

```powershell
python -m pytest tests\test_layers_stubs.py tests\test_student_layers.py -q
$env:PYTHONIOENCODING='utf-8'
python scripts\run_practice.py --out runs\practice.json
python scripts\selfeval.py --summary
```

Nên đánh giá theo kiểu leave-one-out: chạy full stack, sau đó lần lượt bỏ một
layer. Nếu bỏ layer nào làm điểm hoặc độ ổn định giảm, layer đó đang đóng góp.
Điểm public chỉ là công cụ gỡ lỗi; mục tiêu thật là hành vi tổng quát trên brief
ẩn và model thật.

## Chạy với OpenAI bằng file `.env`

Điền API key thật vào `ARENA_API_KEY` trong file `.env`. File này đã được thêm
vào `.gitignore`, còn `.env.example` chỉ chứa giá trị mẫu để có thể commit an
toàn. Không gửi hoặc commit API key thật.

Chạy smoke test một brief trước:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_openai.ps1
```

Sau khi smoke test thành công, chạy toàn bộ brief công khai:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_openai.ps1 -Full
```

Chạy riêng một brief để gỡ lỗi và tiết kiệm phí API:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_openai.ps1 `
  -Brief pub-03-ticket-doi-tra
```

Script nạp `.env` vào môi trường của đúng tiến trình hiện tại rồi gọi đường
`--model real`; nó không thay đổi mã đóng băng trong `arena/`.

Khi chạy qua `run_openai.ps1`, `harness/openai_model.py` chuyển tham số cũ
`max_tokens` thành `max_completion_tokens` và bỏ `temperature=0`, vì các model
OpenAI reasoning hiện tại không chấp nhận hai giá trị cũ này. Adapter chỉ tác
động lên đường chạy OpenAI; mock model và các provider tương thích khác giữ
nguyên hành vi.
