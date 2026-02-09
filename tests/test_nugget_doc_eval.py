"""Smoke tests for nugget-doc evaluation functions."""

import json
from pathlib import Path

from autojudge_base.qrels import Qrels, QrelRow
from autojudge_evaluate.nugget_doc_eval import (
    evaluate_nugget_docs,
    nugget_docs_to_qrels,
    read_nugget_docs_collaborator,
    set_overlap,
)
from autojudge_base.nugget_doc_models import (
    NuggetDocEntry,
    TopicNuggetDocs,
    write_nugget_docs_collaborator,
)


def _make_topics() -> dict[str, TopicNuggetDocs]:
    return {
        "t1": TopicNuggetDocs(topic_id="t1", entries=[
            NuggetDocEntry(question="What is X?", doc_ids=["d1", "d2"]),
            NuggetDocEntry(question="What is Y?", doc_ids=["d2", "d3"]),
        ]),
        "t2": TopicNuggetDocs(topic_id="t2", entries=[
            NuggetDocEntry(question="What is Z?", doc_ids=["d4"]),
        ]),
    }


def test_read_write_roundtrip(tmp_path: Path):
    topics = _make_topics()
    write_nugget_docs_collaborator(topics, tmp_path)
    loaded = read_nugget_docs_collaborator(tmp_path)

    assert set(loaded.keys()) == {"t1", "t2"}
    assert len(loaded["t1"].entries) == 2
    assert loaded["t1"].entries[0].question == "What is X?"
    assert loaded["t1"].entries[0].doc_ids == ["d1", "d2"]


def test_nugget_docs_to_qrels():
    topics = _make_topics()
    qrels = nugget_docs_to_qrels(topics)
    # t1: d1, d2, d3 (union); t2: d4
    assert len(qrels.rows) == 4
    doc_ids_t1 = {r.doc_id for r in qrels.rows if r.topic_id == "t1"}
    assert doc_ids_t1 == {"d1", "d2", "d3"}


def test_set_overlap_perfect():
    qrels = Qrels(rows=[QrelRow("t1", "d1", 1), QrelRow("t1", "d2", 1)])
    results = set_overlap(qrels, qrels)
    assert len(results) == 1
    assert results[0].precision == 1.0
    assert results[0].recall == 1.0
    assert results[0].f1 == 1.0


def test_set_overlap_partial():
    truth = Qrels(rows=[QrelRow("t1", "d1", 1), QrelRow("t1", "d2", 1)])
    predicted = Qrels(rows=[QrelRow("t1", "d1", 1), QrelRow("t1", "d3", 1)])
    results = set_overlap(truth, predicted)
    assert results[0].tp == 1
    assert results[0].fp == 1  # d3
    assert results[0].fn == 1  # d2
    assert results[0].precision == 0.5
    assert results[0].recall == 0.5


def test_evaluate_nugget_docs_end_to_end(tmp_path: Path):
    truth_dir = tmp_path / "truth"
    pred_dir = tmp_path / "pred"

    truth = _make_topics()
    # Predicted: same for t1, missing d4 for t2, extra d5
    pred = {
        "t1": truth["t1"],
        "t2": TopicNuggetDocs(topic_id="t2", entries=[
            NuggetDocEntry(question="What is Z?", doc_ids=["d5"]),
        ]),
    }

    write_nugget_docs_collaborator(truth, truth_dir)
    write_nugget_docs_collaborator(pred, pred_dir)

    results = evaluate_nugget_docs(truth_dir, pred_dir)
    assert len(results) == 2
    # t1 should be perfect, t2 should have 0 overlap
    t1 = next(r for r in results if r.topic_id == "t1")
    t2 = next(r for r in results if r.topic_id == "t2")
    assert t1.precision == 1.0
    assert t2.tp == 0