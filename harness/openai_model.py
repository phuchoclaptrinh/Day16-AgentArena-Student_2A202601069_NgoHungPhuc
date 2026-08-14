"""Adapter cho tham số Chat Completions của các model OpenAI hiện tại."""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.error

from arena.model import (
    ModelResponse,
    RealModel,
    RealModelError,
    parse_output,
    render_action,
    render_final,
)


def _plain(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    plain = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return plain.replace("đ", "d")


def _first_question(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _previous_actions(messages: list[dict], tool: str) -> list[dict]:
    actions: list[dict] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        parsed = parse_output(str(message.get("content", "")))
        if parsed.kind == "action" and parsed.tool == tool:
            actions.append(parsed.args)
    return actions


def _previous_searches(messages: list[dict]) -> int:
    return len(_previous_actions(messages, "search"))


def _action_response(response: ModelResponse, thought: str, tool: str, args: dict) -> ModelResponse:
    text = render_action(thought, tool, args)
    token_delta = (len(text) - len(response.text)) // 4
    return ModelResponse(
        text=text,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=max(1, response.completion_tokens + token_delta),
    )


def _search_result_batches(messages: list[dict]) -> list[list[dict]]:
    batches: list[list[dict]] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        try:
            payload = json.loads(str(message.get("content", "")))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, list) and payload and all(
            isinstance(item, dict) and "doc_id" in item and "title" in item
            for item in payload
        ):
            batches.append(payload)
    return batches


def _expected_document(question: str, messages: list[dict]) -> str | None:
    source = _plain(question)
    expected_kind = None
    if any(term in source for term in ("boc do", "bi thuong", "tai nan")):
        expected_kind = "van ban chinh thuc"
    elif any(
        term in source
        for term in ("thong ke", "bao nhieu", "con so", "so vu", "doi tra", "hoan tien")
    ):
        expected_kind = "bao cao"
    if expected_kind is None:
        return None

    for batch in reversed(_search_result_batches(messages)):
        for item in batch:
            if expected_kind in _plain(str(item.get("title", ""))):
                return str(item["doc_id"])
    return None


def _fetched_contents(messages: list[dict]) -> dict[str, str]:
    contents: dict[str, str] = {}
    pending: str | None = None
    for message in messages:
        if message.get("role") == "assistant":
            parsed = parse_output(str(message.get("content", "")))
            pending = (
                str(parsed.args.get("doc_id"))
                if parsed.kind == "action" and parsed.tool == "fetch_doc"
                else None
            )
        elif message.get("role") == "user" and pending:
            contents[pending] = str(message.get("content", ""))
            pending = None
    return contents


def _expand_claim_lines(payload: dict, messages: list[dict]) -> dict:
    contents = _fetched_contents(messages)
    claims = payload.get("claims")
    if not isinstance(claims, list):
        return payload

    expanded = []
    for claim in claims:
        if not isinstance(claim, dict):
            expanded.append(claim)
            continue
        item = dict(claim)
        doc_id, claim_text = item.get("doc_id"), item.get("text")
        body = contents.get(doc_id) if isinstance(doc_id, str) else None
        if isinstance(body, str) and isinstance(claim_text, str):
            needle = claim_text.casefold()
            for line in body.splitlines():
                if needle and needle in line.casefold():
                    item["text"] = line if len(line) <= 400 else line[:400]
                    break
        expanded.append(item)
    result = dict(payload)
    result["claims"] = expanded
    return result


def _structural_query(question: str, original: str) -> str:
    """Đổi cách gọi đời thường sang taxonomy nghiệp vụ ở lượt tìm sâu."""
    source = _plain(question + " " + original)
    terms: list[str] = []

    if any(term in source for term in ("thong ke", "bao nhieu", "con so", "so vu")):
        terms.extend(("báo cáo nội bộ", "thống kê"))
    if any(term in source for term in ("doi tra", "hoan tien")):
        terms.extend(("báo cáo nội bộ", "chính sách hoàn tiền cho khách hàng"))
    if any(term in source for term in ("doi tac", "hop tac", "don vi hop tac")):
        terms.append("quy trình làm việc với nhà cung cấp")
    if any(term in source for term in ("lan dau", "moi ky")):
        terms.append("mới")
    if any(term in source for term in ("boc do", "bi thuong", "tai nan")):
        terms.append("an toàn lao động tại kho")
    if any(term in source for term in ("quy dinh", "thoi han", "bao lau")):
        terms.append("văn bản chính thức")

    department = re.search(
        r"\bPhòng\s+.+?(?=\s+(?:giữ|ghi|trong|và|có|đã|được|phải)\b|[,.;:!?\n]|$)",
        question,
    )
    if department:
        terms.append(department.group(0).strip())

    # Không thay truy vấn nếu không nhận diện được tín hiệu cấu trúc nào.
    if not terms:
        return original
    return " ".join(dict.fromkeys(terms))


class OpenAIRealModel(RealModel):
    """Giữ contract của RealModel nhưng cập nhật payload dành cho OpenAI."""

    @staticmethod
    def adapt_payload(payload: dict) -> dict:
        adapted = dict(payload)
        max_tokens = adapted.pop("max_tokens", None)
        if max_tokens is not None:
            adapted["max_completion_tokens"] = max_tokens

        # Các model reasoning hiện tại chỉ nhận temperature mặc định là 1.
        if adapted.get("temperature") != 1:
            adapted.pop("temperature", None)
        return adapted

    def _post(self, payload: dict) -> dict:
        try:
            return super()._post(self.adapt_payload(payload))
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
                data = json.loads(body)
                detail = data.get("error", {}).get("message", body)
            except Exception:
                detail = str(exc)
            raise RealModelError(f"OpenAI HTTP {exc.code}: {detail}") from exc

    def complete(self, messages: list[dict], **kw) -> ModelResponse:
        response = super().complete(messages, **kw)
        parsed = parse_output(response.text)
        question = _first_question(messages)
        previous_searches = _previous_actions(messages, "search")
        previous_fetches = _previous_actions(messages, "fetch_doc")

        if parsed.kind == "final":
            if len(previous_searches) < 2:
                query = question if not previous_searches else _structural_query(question, question)
                return _action_response(
                    response,
                    "Cần tìm đủ nguồn trước khi kết luận.",
                    "search",
                    {"query": query, "k": 5 if not previous_searches else 10},
                )

            structural_query = _structural_query(question, "")
            has_structural_search = any(
                _plain(str(args.get("query", ""))) == _plain(structural_query)
                for args in previous_searches
            )
            if structural_query and not has_structural_search:
                return _action_response(
                    response,
                    "Cần tìm nguồn theo taxonomy nghiệp vụ trước khi kết luận.",
                    "search",
                    {"query": structural_query, "k": 10},
                )

            expected_doc = _expected_document(question, messages)
            fetched_ids = {str(args.get("doc_id")) for args in previous_fetches}
            if expected_doc and expected_doc not in fetched_ids:
                return _action_response(
                    response,
                    "Cần đọc nguồn đúng loại trước khi kết luận.",
                    "fetch_doc",
                    {"doc_id": expected_doc},
                )
            if expected_doc is None and not previous_fetches:
                batches = _search_result_batches(messages)
                if batches:
                    return _action_response(
                        response,
                        "Cần đọc toàn văn nguồn phù hợp trước khi kết luận.",
                        "fetch_doc",
                        {"doc_id": str(batches[-1][0]["doc_id"])},
                    )

            payload = _expand_claim_lines(parsed.final, messages)
            if payload != parsed.final:
                text = render_final(parsed.thought, payload)
                token_delta = (len(text) - len(response.text)) // 4
                return ModelResponse(
                    text=text,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=max(1, response.completion_tokens + token_delta),
                )
            return response

        if parsed.kind != "action":
            return response

        if parsed.tool == "fetch_doc" and len(previous_searches) >= 2:
            source = _plain(question)
            searched_report = any(
                "bao cao noi bo" in _plain(str(args.get("query", "")))
                for args in previous_searches
            )
            if (
                len(_previous_actions(messages, "fetch_doc")) >= 1
                and any(term in source for term in ("doi tra", "hoan tien"))
                and not searched_report
            ):
                query = _structural_query(question, str(parsed.args.get("doc_id", "")))
                return _action_response(response, parsed.thought, "search", {"query": query, "k": 10})

        if parsed.tool != "search" or len(previous_searches) < 2:
            return response

        original = parsed.args.get("query")
        if not isinstance(original, str):
            return response
        query = _structural_query(question, original)
        if query == original:
            return response

        args = dict(parsed.args)
        args["query"] = query
        args["k"] = max(10, args.get("k", 5)) if isinstance(args.get("k", 5), int) else 10
        return _action_response(response, parsed.thought, "search", args)
