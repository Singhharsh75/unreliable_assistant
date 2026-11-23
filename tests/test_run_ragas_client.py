import asyncio
from types import SimpleNamespace as NS

import eval.run_ragas as rr


class FakeOpenAI:
    def __init__(self, finish="stop", content=" hi "):
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))
        self.finish, self.content = finish, content

    async def _create(self, **kw):
        self.calls.append(kw)
        return NS(choices=[NS(finish_reason=self.finish, message=NS(content=self.content))])


def test_openrouter_branch(monkeypatch):
    monkeypatch.setattr(rr, "OPENROUTER", True)
    c = FakeOpenAI()
    assert asyncio.run(rr.complete(c, "m", "q", system="s")) == "hi"
    assert [m["role"] for m in c.calls[0]["messages"]] == ["system", "user"] and c.calls[0]["model"] == "m"
    assert asyncio.run(rr.complete(FakeOpenAI(finish="content_filter"), "m", "q")) is None
    assert asyncio.run(rr.complete(FakeOpenAI(content=None), "m", "q")) == ""


def test_anthropic_branch(monkeypatch):
    monkeypatch.setattr(rr, "OPENROUTER", False)
    calls = []

    async def create(**kw):
        calls.append(kw)
        return NS(stop_reason="end_turn", content=[NS(type="text", text="ok")])

    c = NS(beta=NS(messages=NS(create=create)))
    assert asyncio.run(rr.complete(c, "m", "q")) == "ok" and "system" not in calls[0]
    assert calls[0]["fallbacks"] == "default"
    asyncio.run(rr.complete(c, "claude-haiku-5-5", "q", system="s"))
    assert calls[1]["system"] == "s" and "fallbacks" not in calls[1] and "betas" not in calls[1]


def test_id_metrics_follow_ragas_definitions():
    assert rr.id_precision(["g", "x", "y", "z"], ["g"]) == 1.0  # rank-weighted, not set precision (0.25)
    assert rr.id_precision(["x", "g"], ["g"]) == 0.5
    assert rr.id_precision(["g1", "x", "g2"], ["g1", "g2"]) == (1 + 2 / 3) / 2
    assert rr.id_precision([], ["g"]) == 0.0
    assert rr.id_recall(["a", "b"], ["a", "c"]) == 0.5 and rr.id_recall([], ["a"]) == 0.0


def test_sample_is_stratified_and_deterministic():
    rows = [{"id": f"{c}-{i}", "category": c} for c in rr.quotas(250) for i in range(rr.quotas(250)[c])]
    s = rr.sample(rows, 50)
    assert len(s) == 50 and s == rr.sample(rows, 50)
    counts = {c: sum(r["category"] == c for r in s) for c in rr.quotas(50)}
    assert counts == rr.quotas(50)
    assert rr.sample(rows, 0) is rows and rr.sample(rows, 999) is rows
