"""Evaluate nugget-document overlap between truth and predicted sets.

Provides:
- Reading collaborator-format nugget-doc JSON files
- Converting nugget-doc mappings to binary Qrels
- Set precision/recall/F1 between two Qrels
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from autojudge_base.qrels import Qrels, QrelRow
from autojudge_base.nugget_doc_models import NuggetDocEntry, TopicNuggetDocs


# =============================================================================
# Read collaborator format
# =============================================================================


def read_nugget_docs_collaborator(input_dir: Path) -> Dict[str, TopicNuggetDocs]:
    """Read nuggets_{topic_id}.json files from a directory.

    Parses the collaborator format: {question: ["OR", {"OPEN_ENDED_ANSWER": [doc_ids]}]}

    Returns:
        Dict mapping topic_id -> TopicNuggetDocs
    """
    topics: Dict[str, TopicNuggetDocs] = {}
    for path in sorted(input_dir.glob("nuggets_*.json")):
        # Extract topic_id from filename: nuggets_{topic_id}.json
        topic_id = path.stem.removeprefix("nuggets_")
        data = json.loads(path.read_text())

        entries = []
        for question, value in data.items():
            aggregator = value[0]
            answer_dict = value[1]
            answer_type = next(iter(answer_dict))
            doc_ids = answer_dict[answer_type]
            entries.append(NuggetDocEntry(
                question=question,
                doc_ids=doc_ids,
                aggregator=aggregator,
                answer_type=answer_type,
            ))

        topics[topic_id] = TopicNuggetDocs(topic_id=topic_id, entries=entries)

    return topics


# =============================================================================
# Convert to Qrels
# =============================================================================


def nugget_docs_to_qrels(topics: Dict[str, TopicNuggetDocs]) -> Qrels:
    """Convert nugget-doc mappings to binary Qrels.

    For each topic, unions all doc_ids across all nugget entries -> grade=1.
    """
    rows: List[QrelRow] = []
    for topic_id, topic in topics.items():
        all_doc_ids: Set[str] = set()
        for entry in topic.entries:
            all_doc_ids.update(entry.doc_ids)
        for doc_id in sorted(all_doc_ids):
            rows.append(QrelRow(topic_id=topic_id, doc_id=doc_id, grade=1))
    return Qrels(rows=rows)


# =============================================================================
# Set precision / recall
# =============================================================================


@dataclass
class SetOverlapResult:
    """Per-topic set overlap metrics."""

    topic_id: str
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def set_overlap(
    truth: Qrels,
    predicted: Qrels,
    on_missing: str = "default",
    relevance_threshold: int = 1,
    *,
    truth_relevance_threshold: Optional[int] = None,
    predict_relevance_threshold: Optional[int] = None,
) -> List[SetOverlapResult]:
    """Per-topic set precision/recall between truth and predicted qrels.

    Only considers rows with grade >= relevance_threshold as relevant.

    Args:
        on_missing: How to handle topics present in only one side.
            "error"   - raise ValueError listing mismatched topics
            "warn"    - warn and skip mismatched topics
            "skip"    - silently skip mismatched topics
            "default" - include all topics; missing side treated as empty set
        relevance_threshold: Minimum grade to consider a document relevant
            (used as fallback when per-side thresholds are not set).
        truth_relevance_threshold: Threshold for truth side (overrides relevance_threshold).
        predict_relevance_threshold: Threshold for predicted side (overrides relevance_threshold).
    """
    t_thresh = truth_relevance_threshold if truth_relevance_threshold is not None else relevance_threshold
    p_thresh = predict_relevance_threshold if predict_relevance_threshold is not None else relevance_threshold

    def _group_by_topic(qrels: Qrels, threshold: int) -> Dict[str, Set[str]]:
        groups: Dict[str, Set[str]] = defaultdict(set)
        for row in qrels.rows:
            if row.grade >= threshold:
                groups[row.topic_id].add(row.doc_id)
        return groups

    truth_groups = _group_by_topic(truth, t_thresh)
    pred_groups = _group_by_topic(predicted, p_thresh)

    truth_only = set(truth_groups.keys()) - set(pred_groups.keys())
    pred_only = set(pred_groups.keys()) - set(truth_groups.keys())

    if truth_only or pred_only:
        if on_missing == "error":
            parts = []
            if truth_only:
                parts.append(f"truth-only: {sorted(truth_only)}")
            if pred_only:
                parts.append(f"predicted-only: {sorted(pred_only)}")
            raise ValueError(f"Topic mismatch: {'; '.join(parts)}")
        elif on_missing == "warn":
            import click as _click
            if truth_only:
                _click.echo(f"Warning: topics only in truth (skipped): {sorted(truth_only)}", err=True)
            if pred_only:
                _click.echo(f"Warning: topics only in predicted (skipped): {sorted(pred_only)}", err=True)

    if on_missing in ("warn", "skip"):
        all_topics = sorted(set(truth_groups.keys()) & set(pred_groups.keys()))
    else:
        all_topics = sorted(set(truth_groups.keys()) | set(pred_groups.keys()))
    results: List[SetOverlapResult] = []

    for topic_id in all_topics:
        truth_docs = truth_groups.get(topic_id, set())
        pred_docs = pred_groups.get(topic_id, set())

        tp = len(truth_docs & pred_docs)
        fp = len(pred_docs - truth_docs)
        fn = len(truth_docs - pred_docs)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append(SetOverlapResult(
            topic_id=topic_id,
            precision=precision,
            recall=recall,
            f1=f1,
            tp=tp, fp=fp, fn=fn,
        ))

    return results


# =============================================================================
# Convenience: end-to-end from directories
# =============================================================================


def evaluate_nugget_docs(
    truth_dir: Path,
    predicted_dir: Path,
    on_missing: str = "default",
) -> List[SetOverlapResult]:
    """Read two collaborator-format directories, convert to qrels, compute overlap."""
    truth_topics = read_nugget_docs_collaborator(truth_dir)
    pred_topics = read_nugget_docs_collaborator(predicted_dir)

    truth_qrels = nugget_docs_to_qrels(truth_topics)
    pred_qrels = nugget_docs_to_qrels(pred_topics)

    return set_overlap(truth_qrels, pred_qrels, on_missing=on_missing)


# =============================================================================
# Inter-annotator agreement
# =============================================================================


def _resolve_topics(
    truth_topics: Set[str],
    pred_topics: Set[str],
    on_missing: str,
) -> List[str]:
    """Determine which topics to process given the on_missing policy.

    Shared logic for both set_overlap and grade_agreement.
    """
    truth_only = truth_topics - pred_topics
    pred_only = pred_topics - truth_topics

    if truth_only or pred_only:
        if on_missing == "error":
            parts = []
            if truth_only:
                parts.append(f"truth-only: {sorted(truth_only)}")
            if pred_only:
                parts.append(f"predicted-only: {sorted(pred_only)}")
            raise ValueError(f"Topic mismatch: {'; '.join(parts)}")
        elif on_missing == "warn":
            import click as _click
            if truth_only:
                _click.echo(f"Warning: topics only in truth (skipped): {sorted(truth_only)}", err=True)
            if pred_only:
                _click.echo(f"Warning: topics only in predicted (skipped): {sorted(pred_only)}", err=True)

    if on_missing in ("warn", "skip"):
        return sorted(truth_topics & pred_topics)
    else:
        return sorted(truth_topics | pred_topics)


def _align_grades_by_topic(
    truth: Qrels,
    predicted: Qrels,
    on_missing: str,
    default_grade: int,
) -> Dict[str, Tuple[List[int], List[int]]]:
    """Align grade vectors by (topic_id, doc_id).

    For each topic, collects the union of doc_ids from both sides and
    produces paired grade vectors. Missing docs get ``default_grade``.

    Returns:
        Dict mapping topic_id -> (truth_grades, pred_grades) aligned vectors.
    """
    def _group_grades(qrels: Qrels) -> Dict[str, Dict[str, int]]:
        groups: Dict[str, Dict[str, int]] = defaultdict(dict)
        for row in qrels.rows:
            groups[row.topic_id][row.doc_id] = row.grade
        return groups

    truth_groups = _group_grades(truth)
    pred_groups = _group_grades(predicted)

    all_topics = _resolve_topics(
        set(truth_groups.keys()), set(pred_groups.keys()), on_missing
    )

    result: Dict[str, Tuple[List[int], List[int]]] = {}
    for topic_id in all_topics:
        t_docs = truth_groups.get(topic_id, {})
        p_docs = pred_groups.get(topic_id, {})
        all_doc_ids = sorted(set(t_docs.keys()) | set(p_docs.keys()))

        truth_vec: List[int] = []
        pred_vec: List[int] = []
        for doc_id in all_doc_ids:
            truth_vec.append(t_docs.get(doc_id, default_grade))
            pred_vec.append(p_docs.get(doc_id, default_grade))

        result[topic_id] = (truth_vec, pred_vec)

    return result


@dataclass
class AgreementResult:
    """Per-topic inter-annotator agreement metrics."""

    topic_id: str
    jaccard: float
    kappa_binary: Dict[Tuple[int, int], float]  # (truth_threshold, pred_threshold) -> Cohen's kappa
    kappa_weighted: float
    kripp_alpha: float
    ari: float


def grade_agreement(
    truth: Qrels,
    predicted: Qrels,
    on_missing: str = "default",
    max_grade: int = 1,
    relevance_threshold: int = 1,
    default_grade: int = 0,
    *,
    truth_max_grade: Optional[int] = None,
    predict_max_grade: Optional[int] = None,
    truth_relevance_threshold: Optional[int] = None,
    predict_relevance_threshold: Optional[int] = None,
) -> List[AgreementResult]:
    """Compute inter-annotator agreement metrics per topic.

    Args:
        truth: Ground-truth qrels.
        predicted: Predicted qrels.
        on_missing: Policy for topics in only one side.
        max_grade: Fallback upper bound when per-side max_grade is not set.
        relevance_threshold: Fallback threshold when per-side threshold is not set.
        default_grade: Grade assigned to doc_ids present in only one side.
        truth_max_grade: Grade scale upper bound for truth (overrides max_grade).
        predict_max_grade: Grade scale upper bound for predicted (overrides max_grade).
        truth_relevance_threshold: Binary threshold for truth side (overrides relevance_threshold).
        predict_relevance_threshold: Binary threshold for predicted side (overrides relevance_threshold).

    Returns:
        List of AgreementResult, one per topic.
    """
    import math

    import krippendorff
    import numpy as np
    from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

    t_max = truth_max_grade if truth_max_grade is not None else max_grade
    p_max = predict_max_grade if predict_max_grade is not None else max_grade
    t_rel_thresh = truth_relevance_threshold if truth_relevance_threshold is not None else relevance_threshold
    p_rel_thresh = predict_relevance_threshold if predict_relevance_threshold is not None else relevance_threshold

    aligned = _align_grades_by_topic(truth, predicted, on_missing, default_grade)

    results: List[AgreementResult] = []
    for topic_id, (truth_vec, pred_vec) in aligned.items():
        # Clip grades to per-side scales
        t = np.clip(truth_vec, 0, t_max)
        p = np.clip(pred_vec, 0, p_max)

        # --- Binary kappa: cross-product of thresholds ---
        kappa_binary: Dict[Tuple[int, int], float] = {}
        for tt in range(1, t_max + 1):
            for pt in range(1, p_max + 1):
                t_bin = (t >= tt).astype(int)
                p_bin = (p >= pt).astype(int)

                # Cohen's kappa: undefined when either rater has zero variance
                if len(set(t_bin)) < 2 and len(set(p_bin)) < 2:
                    kappa_binary[(tt, pt)] = float("nan")
                else:
                    kappa_binary[(tt, pt)] = cohen_kappa_score(t_bin, p_bin)

        # --- Jaccard at per-side relevance thresholds ---
        t_rel = t >= t_rel_thresh
        p_rel = p >= p_rel_thresh
        tp = int(np.sum(t_rel & p_rel))
        fp = int(np.sum(~t_rel & p_rel))
        fn = int(np.sum(t_rel & ~p_rel))
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")

        # --- Weighted Cohen's kappa (linear weights) on raw grades ---
        # Only meaningful when both sides share the same scale
        if t_max != p_max:
            kappa_w = float("nan")
        elif len(set(t)) < 2 and len(set(p)) < 2:
            kappa_w = float("nan")
        else:
            labels = list(range(0, t_max + 1))
            kappa_w = cohen_kappa_score(t, p, weights="linear", labels=labels)

        # --- Krippendorff's alpha (ordinal) ---
        # Only meaningful when both sides share the same scale
        if t_max != p_max:
            k_alpha = float("nan")
        else:
            reliability_matrix = np.array([t, p])
            try:
                k_alpha = krippendorff.alpha(
                    reliability_data=reliability_matrix,
                    level_of_measurement="ordinal",
                )
                if k_alpha is None or (isinstance(k_alpha, float) and math.isnan(k_alpha)):
                    k_alpha = float("nan")
            except Exception:
                k_alpha = float("nan")

        # --- Adjusted RAND Index at per-side relevance thresholds ---
        t_bin_ari = (t >= t_rel_thresh).astype(int)
        p_bin_ari = (p >= p_rel_thresh).astype(int)
        ari = adjusted_rand_score(t_bin_ari, p_bin_ari)

        results.append(AgreementResult(
            topic_id=topic_id,
            jaccard=jaccard,
            kappa_binary=kappa_binary,
            kappa_weighted=kappa_w,
            kripp_alpha=k_alpha,
            ari=ari,
        ))

    return results