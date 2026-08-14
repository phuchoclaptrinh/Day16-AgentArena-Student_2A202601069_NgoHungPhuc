from arena.model import ModelResponse, parse_output
from harness.openai_model import OpenAIRealModel, _structural_query


def test_structural_query_maps_partner_language_to_supplier_taxonomy():
    query = _structural_query(
        "Bên Phòng Đào tạo giữ thống kê các vụ với đơn vị hợp tác lần đầu.",
        "thống kê hồ sơ đối tác mới bị trả lại",
    )

    assert "báo cáo nội bộ" in query
    assert "quy trình làm việc với nhà cung cấp" in query
    assert "mới" in query
    assert "Phòng Đào tạo" in query
    assert "giữ thống kê" not in query


def test_structural_query_maps_loading_accident_to_warehouse_safety():
    query = _structural_query(
        "Công nhân bị thương khi bốc dỡ; theo quy định phải báo trong bao lâu?",
        "quy định bốc dỡ thông báo thời hạn",
    )

    assert "an toàn lao động tại kho" in query
    assert "văn bản chính thức" in query


def test_structural_query_maps_returns_to_refund_report():
    query = _structural_query(
        "Ticket khách hàng về đổi trả hàng lỗi đã được xử lý thế nào?",
        "quy trình đổi trả hàng lỗi",
    )

    assert "báo cáo nội bộ" in query
    assert "chính sách hoàn tiền cho khách hàng" in query


def test_complete_rewrites_only_the_third_search(monkeypatch):
    model = OpenAIRealModel("https://api.openai.com/v1", "test", "test")
    raw = ModelResponse(
        text='THOUGHT: tìm sâu\nACTION: {"tool":"search","args":{"query":"hồ sơ đối tác"}}',
        prompt_tokens=100,
        completion_tokens=20,
    )
    monkeypatch.setattr("arena.model.RealModel.complete", lambda *_args, **_kw: raw)
    messages = [
        {"role": "user", "content": "Thống kê đơn vị hợp tác lần đầu là bao nhiêu?"},
        {"role": "assistant", "content": 'ACTION: {"tool":"search","args":{"query":"một"}}'},
        {"role": "user", "content": "[]"},
        {"role": "assistant", "content": 'ACTION: {"tool":"search","args":{"query":"hai"}}'},
        {"role": "user", "content": "[]"},
    ]

    parsed = parse_output(model.complete(messages).text)

    assert parsed.tool == "search"
    assert "nhà cung cấp" in parsed.args["query"]
    assert parsed.args["k"] == 10


def test_complete_inserts_deep_report_search_after_ticket_fetch(monkeypatch):
    model = OpenAIRealModel("https://api.openai.com/v1", "test", "test")
    raw = ModelResponse(
        text='THOUGHT: đọc chính sách\nACTION: {"tool":"fetch_doc","args":{"doc_id":"doc-0021"}}',
        prompt_tokens=100,
        completion_tokens=20,
    )
    monkeypatch.setattr("arena.model.RealModel.complete", lambda *_args, **_kw: raw)
    messages = [
        {"role": "user", "content": "Ticket đổi trả hàng lỗi đã xử lý thế nào?"},
        {"role": "assistant", "content": 'ACTION: {"tool":"search","args":{"query":"ticket"}}'},
        {"role": "user", "content": "[]"},
        {"role": "assistant", "content": 'ACTION: {"tool":"search","args":{"query":"đổi trả"}}'},
        {"role": "user", "content": "[]"},
        {"role": "assistant", "content": 'ACTION: {"tool":"fetch_doc","args":{"doc_id":"doc-0006"}}'},
        {"role": "user", "content": "nội dung ticket"},
    ]

    parsed = parse_output(model.complete(messages).text)

    assert parsed.tool == "search"
    assert "chính sách hoàn tiền" in parsed.args["query"]
    assert parsed.args["k"] == 10


def test_complete_rejects_premature_final(monkeypatch):
    model = OpenAIRealModel("https://api.openai.com/v1", "test", "test")
    raw = ModelResponse(
        text='FINAL: {"answer":"không biết","claims":[],"citations":[],"abstain":true}',
        prompt_tokens=100,
        completion_tokens=20,
    )
    monkeypatch.setattr("arena.model.RealModel.complete", lambda *_args, **_kw: raw)

    parsed = parse_output(
        model.complete([{"role": "user", "content": "Quy định bốc dỡ là gì?"}]).text
    )

    assert parsed.tool == "search"
    assert parsed.args["k"] == 5


def test_complete_expands_claim_to_full_physical_line(monkeypatch):
    model = OpenAIRealModel("https://api.openai.com/v1", "test", "test")
    raw = ModelResponse(
        text='FINAL: {"answer":"thử lại","claims":[{"doc_id":"doc-0008","text":"Quy trình chuẩn là thử lại truy vấn"}],"citations":["doc-0008"],"abstain":false}',
        prompt_tokens=100,
        completion_tokens=20,
    )
    monkeypatch.setattr("arena.model.RealModel.complete", lambda *_args, **_kw: raw)
    messages = [
        {"role": "user", "content": "Cảm biến mất kết nối xử lý thế nào?"},
        {"role": "assistant", "content": 'ACTION: {"tool":"search","args":{"query":"một"}}'},
        {"role": "user", "content": "[]"},
        {"role": "assistant", "content": 'ACTION: {"tool":"search","args":{"query":"hai"}}'},
        {"role": "user", "content": "[]"},
        {"role": "assistant", "content": 'ACTION: {"tool":"fetch_doc","args":{"doc_id":"doc-0008"}}'},
        {"role": "user", "content": "Tiêu đề\nQuy trình chuẩn là thử lại truy vấn trước khi báo cáo sự cố."},
    ]

    parsed = parse_output(model.complete(messages).text)

    assert parsed.final["claims"][0]["text"] == (
        "Quy trình chuẩn là thử lại truy vấn trước khi báo cáo sự cố."
    )
