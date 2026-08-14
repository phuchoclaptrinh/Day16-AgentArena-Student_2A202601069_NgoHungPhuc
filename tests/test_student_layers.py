"""Các ca biên quan trọng cho năm middleware do sinh viên cài đặt."""

from types import SimpleNamespace

from arena.corpus import Corpus, Doc, INJECTION_CANARY
from arena.model import DEGRADED_MARKERS, FINALIZE_SENTINEL
from arena.tools import ToolResult
from harness.layers.budget_policy import BudgetPolicy
from harness.layers.citation_checker import CitationChecker
from harness.layers.critic import Critic
from harness.layers.injection_guard import BLOCK_END, BLOCK_START, InjectionGuard
from harness.layers.retry import Retry
from harness.openai_model import OpenAIRealModel


def _ctx(*, docs=(), observed_text="", calls=0, limit=8):
    corpus = Corpus(list(docs))
    return SimpleNamespace(
        corpus=corpus,
        observed_text=observed_text,
        saw=lambda text: bool(text) and text in observed_text,
        tools=SimpleNamespace(calls=calls),
        max_tool_calls=limit,
        state={},
    )


def test_critic_tries_each_and_boundary_without_rewriting_the_halves():
    left = "Quy định A có điều kiện X và điều kiện Y"
    right = "Quy định B yêu cầu phê duyệt trước"
    docs = (
        Doc("doc-0001", "A", left, ()),
        Doc("doc-0002", "B", right, ()),
    )
    ctx = _ctx(docs=docs, observed_text=f"{left}\n{right}")
    fused = f"{left} và {right}"

    report = Critic().after_agent(
        ctx,
        {
            "answer": fused,
            "claims": [{"text": fused, "doc_id": "doc-0001"}],
            "citations": ["doc-0001"],
            "abstain": False,
        },
    )

    assert report["claims"] == [
        {"text": left, "doc_id": "doc-0001"},
        {"text": right, "doc_id": "doc-0002"},
    ]
    assert report["abstain"] is True


def test_citation_checker_only_reattributes_to_a_fully_observed_document():
    line = "Nội thành giao trong 2 ngày làm việc."
    wrong = Doc("doc-0001", "Cũ", "Nội thành giao trong 5 ngày.", ())
    right = Doc("doc-0002", "Mới", f"Tiêu đề\n{line}", ())
    ctx = _ctx(docs=(wrong, right), observed_text=right.body)
    report = {
        "claims": [{"text": line, "doc_id": wrong.doc_id}],
        "citations": [wrong.doc_id],
    }

    fixed = CitationChecker().after_agent(ctx, report)

    assert fixed["claims"][0] == {"text": line, "doc_id": right.doc_id}
    assert fixed["citations"] == [right.doc_id]


def test_injection_guard_removes_complete_and_unclosed_blocks():
    dirty = (
        f"trước {BLOCK_START}\n{INJECTION_CANARY}\n{BLOCK_END} giữa "
        f"{BLOCK_START}\n{INJECTION_CANARY}"
    )
    result = InjectionGuard().wrap_tool_call(
        _ctx(), lambda _name, _args: ToolResult(True, dirty), "fetch_doc", {}
    )

    assert BLOCK_START not in result.content
    assert INJECTION_CANARY not in result.content


def test_budget_policy_reserves_submit_and_uses_finalize_sentinel():
    ctx = _ctx(calls=7, limit=8)
    policy = BudgetPolicy(reserve=1)
    called = False

    def call(_name, _args):
        nonlocal called
        called = True
        return ToolResult(True, "unexpected")

    messages = policy.before_model(ctx, [])
    result = policy.wrap_tool_call(ctx, call, "search", {})

    assert FINALIZE_SENTINEL in messages[-1]["content"]
    assert not called
    assert not result.ok


def test_retry_handles_ok_but_degraded_results_without_spending_reserve():
    ctx = _ctx(calls=5, limit=8)
    outputs = iter(
        [
            ToolResult(True, DEGRADED_MARKERS[0]),
            ToolResult(False, "", "timeout"),
            ToolResult(True, "sạch"),
        ]
    )

    def call(_name, _args):
        ctx.tools.calls += 1
        return next(outputs)

    result = Retry(max_attempts=3, reserve=1).wrap_tool_call(ctx, call, "fetch_doc", {})

    assert not result.ok
    assert ctx.tools.calls == 7
    assert ctx.state["retry_attempts"] == 2
    assert ctx.state["retry_extra_calls"] == 1


def test_openai_adapter_uses_current_chat_completion_parameters():
    original = {
        "model": "gpt-5.6-terra",
        "messages": [{"role": "user", "content": "test"}],
        "temperature": 0.0,
        "max_tokens": 1200,
    }

    adapted = OpenAIRealModel.adapt_payload(original)

    assert adapted["max_completion_tokens"] == 1200
    assert "max_tokens" not in adapted
    assert "temperature" not in adapted
    assert original["max_tokens"] == 1200
