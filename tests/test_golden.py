from eval.build_golden import STRATA, Candidate, check, is_dup, quotas

CHUNKS = [
    {"note_id": "a", "tenants": None},
    {"note_id": "b", "tenants": None},
    {"note_id": "p", "tenants": ["tenant_acme"]},
    {"note_id": "q", "tenants": ["tenant_other"]},
]


def cand(*ids):
    return Candidate(question="q", reference_answer="a", gold_chunk_ids=list(ids))


def test_quotas_match_strata_and_sum():
    assert quotas(250) == {"factoid": 63, "complex_reasoning": 63, "multi_hop": 50, "summary": 37, "negative": 37}
    for n in (1, 7, 20, 250):
        assert sum(quotas(n).values()) == n and set(quotas(n)) == set(STRATA)


def test_check_rules():
    assert check("factoid", cand("C0"), CHUNKS) == (None, [0])
    assert check("factoid", cand(), CHUNKS)[0] == "missing_gold"
    assert check("factoid", cand("C9"), CHUNKS)[0] == "bad_chunk_id"
    assert check("factoid", cand("x"), CHUNKS)[0] == "bad_chunk_id"
    assert check("negative", cand("C0"), CHUNKS)[0] == "negative_has_gold"
    assert check("negative", cand(), CHUNKS) == (None, [])
    assert check("multi_hop", cand("C0"), CHUNKS)[0] == "multi_hop_single_note"
    assert check("multi_hop", cand("C0", "C1"), CHUNKS) == (None, [0, 1])
    assert check("multi_hop", cand("C2", "C3"), CHUNKS)[0] == "mixed_private_tenants"


def test_near_duplicates_are_caught():
    assert is_dup("What temperature is SKU-099 stored at?", ["what temperature is SKU-099 stored at"])
    assert not is_dup("Who approves a lot release?", ["What temperature is SKU-099 stored at?"])


def test_from_files_applies_judge_threshold_and_tenant(tmp_path):
    import json

    from eval.build_golden import from_files

    rows = [
        {"cid": "a", "category": "factoid", "question": "Range?", "reference_answer": "2-8 C", "gold_chunk_ids": ["C0"]},
        {"cid": "b", "category": "factoid", "question": "Weak one?", "reference_answer": "x", "gold_chunk_ids": ["C0"]},
        {"cid": "c", "category": "factoid", "question": "No verdict?", "reference_answer": "x", "gold_chunk_ids": ["C0"]},
        {"cid": "d", "category": "multi_hop", "question": "Acme escalation and HOLD?", "reference_answer": "x",
         "gold_chunk_ids": ["C6", "C1"]},
    ]
    ok = {"grounded": 5, "gold_complete": 5, "category_fit": 4, "standalone": 5, "reason": "clean"}
    verdicts = [{"cid": "a", **ok}, {"cid": "b", **ok, "grounded": 2}, {"cid": "d", **ok}]
    cands, verd = tmp_path / "c.jsonl", tmp_path / "v.jsonl"
    cands.write_text("".join(json.dumps(r) + "\n" for r in rows))
    verd.write_text("".join(json.dumps(r) + "\n" for r in verdicts))
    kept, report = from_files(cands, verd, 20, "gen", "judge")
    assert [r["question"] for r in kept] == ["Range?", "Acme escalation and HOLD?"]
    assert report["rejected"] == {"judge_grounded": 1, "judge_refused": 1}
    assert kept[1]["tenant_id"] == "tenant_acme" and kept[0]["tenant_id"] == "tenant_demo"
