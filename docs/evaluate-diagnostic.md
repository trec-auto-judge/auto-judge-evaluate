# Plan: `evaluate-diagnostic` — Correlation input verification + per-judge diagnostics

**Status:** Shelved (2026-04-20). No code changes until unshelved.

## Problem

`LeaderboardEvaluator.evaluate()` interleaves three concerns in one loop:
1. Measure-pair iteration
2. Per-method parsing + `top_k` filtering
3. Input validation + correlation computation

Consequences:
- **All-tied detection must happen *after* `top_k` filtering** (filtering can introduce ties that weren't in the unfiltered ranking). Current structure makes that hard to express.
- **`_check_input_or_raise`** at `evaluation.py:348` enforces `len < 3` and length-equality, but is scattered: each correlation helper calls it individually. No central source of truth for "what makes an aligned `(a, b)` pair valid for correlation".
- **Degenerate-data warnings** (`print(...)` at `pyircore.py:126`) live inside the math layer with no access to judge/measure/method context and no dedup.
- **Data issues that depend only on the filtered ranking (not on the measure name)** — e.g., `len < 3` after `top_k`, all-tied after `top_k` — have no shared validation surface.

## Proposal

### 1. New verification module — follows the existing `eval_results/verification.py` pattern

The codebase already has a verification convention in `eval_results/verification.py`:
- `VerificationError(Exception)` for raised failures.
- `OnFail = Literal["error", "warn", "ignore"]` for pluggable severity.
- `_handle_failure(on_fail, message)` as the central dispatcher.
- `check_*(...)` functions that **return the issues they found** and call `_handle_failure(...)` if there are any.

Reuse that pattern. New file: `auto-judge-evaluate/src/autojudge_evaluate/correlation_verification.py`:

```python
from typing import Literal

from .eval_results.verification import VerificationError, OnFail, _handle_failure

CorrelationIssue = Literal["too_few", "length_mismatch", "all_tied"]


def check_correlation_inputs(
    a: list[float],
    b: list[float],
    on_fail: OnFail = "warn",
) -> CorrelationIssue | None:
    """
    Check that (a, b) is a valid input pair for correlation.

    Returns the issue name if any, else None. Calls _handle_failure if an issue
    is found (subject to on_fail policy).
    """
    if len(a) != len(b):
        _handle_failure(on_fail, f"Length mismatch: {len(a)} != {len(b)}")
        return "length_mismatch"
    if len(a) < 3:
        _handle_failure(on_fail, f"Too few elements for correlation: {len(a)}")
        return "too_few"
    if all(x == a[0] for x in a) and all(y == b[0] for y in b):
        _handle_failure(on_fail, f"All {len(a)} elements tied (a={a[0]!r}, b={b[0]!r})")
        return "all_tied"
    return None
```

Key choice: **callers in the per-judge loop pass `on_fail="ignore"`** so each check is silent; the loop collects issues and emits ONE summary warning per `evaluate()` call. Other (non-loop) callers can pick `"warn"` or `"error"` as they see fit.

If `OnFail` / `VerificationError` / `_handle_failure` end up reused in three+ places, promote them out of `eval_results/verification.py` into a shared `verification_base.py`. For now, importing from `eval_results.verification` is fine.

### 2. Three-phase refactor of `evaluate()`

**Phase A — task preparation** (new generator method):

```python
@dataclass
class CorrelationTask:
    truth_m: str
    eval_m: str
    method: str             # e.g., "tauap_b@5"
    base_method: str        # e.g., "tauap_b"
    top_k: int | None
    a: list[float]          # post-filter, aligned
    b: list[float]          # post-filter, aligned
    issue: CorrelationIssue | None  # None == OK

def _prepare_tasks(self, truth_filtered, eval_filtered, eval_raw) -> Iterable[CorrelationTask]:
    for truth_m, eval_m in self.get_measure_pairs(eval_raw):
        for method in self.correlation_methods:
            base_method, top_k = parse_correlation_method(method)
            truth_ranking = self.extract_ranking(truth_filtered, truth_m)
            eval_ranking  = self.extract_ranking(eval_filtered, eval_m)
            if truth_ranking is None or eval_ranking is None:
                continue
            if top_k is not None:
                sorted_runs = sorted(truth_ranking.items(), key=lambda x: x[1], reverse=True)
                top_ids = {r for r, _ in sorted_runs[:top_k]}
                truth_ranking = {r: v for r, v in truth_ranking.items() if r in top_ids}
                eval_ranking  = {r: v for r, v in eval_ranking.items()  if r in top_ids}
            a, b = self._align_rankings(truth_ranking, eval_ranking)
            yield CorrelationTask(truth_m, eval_m, method, base_method, top_k,
                                  a, b, check_correlation_inputs(a, b, on_fail="ignore"))
```

Note the verification runs AFTER `top_k` filtering — this is the invariant the current code cannot express cleanly.

**Phase B — aggregation + diagnostics**:

```python
def evaluate(self, eval_raw):
    # ... existing filtering setup unchanged ...
    ret: dict = {}
    issues: list[tuple[str, str, str, CorrelationIssue]] = []
    for task in self._prepare_tasks(truth_filtered, eval_filtered, eval_raw):
        bucket = ret.setdefault((task.truth_m, task.eval_m), {})
        if task.issue is not None:
            issues.append((task.truth_m, task.eval_m, task.method, task.issue))
            bucket[task.method] = 0.0  # existing contract
            continue
        bucket[task.method] = _compute(task.a, task.b, task.base_method)

    if issues:
        self._emit_diagnostic_summary(eval_raw, issues)
    return ret

def _emit_diagnostic_summary(self, eval_raw, issues):
    eval_label = getattr(eval_raw, "name", None) or str(self.truth_leaderboard)
    by_issue: dict[str, list] = {}
    for truth_m, eval_m, method, issue in issues:
        by_issue.setdefault(issue, []).append((truth_m, eval_m, method))
    parts = [f"[{eval_label}] correlation input issues:"]
    for issue, cases in by_issue.items():
        sample = cases[:5]
        suffix = " ..." if len(cases) > 5 else ""
        parts.append(f"  {issue}: {len(cases)} case(s): {sample}{suffix}")
    warnings.warn("\n".join(parts), RuntimeWarning, stacklevel=2)
```

**Phase C — pure compute**:

```python
def _compute(a, b, base_method) -> float:
    """Inputs pre-verified. All correlation methods go through here."""
    if base_method == "tauap_b":
        return _pyircore_tauap_b(a, b)
    df = pd.DataFrame([{"a": x, "b": y} for x, y in zip(a, b)])
    return float(df.corr(base_method).iloc[0]["b"])
```

### 3. `pyircore.py` cleanup (separate, small)

- Remove `print(...)` at line 126 (was the original symptom).
- Keep `return 0.0` as defensive belt (now unreachable, since verification catches first).
- Drop vestigial `decreasing` param from `tauap_b`, `tauap_b_ties`, wrapper — hardcode the negate.
- Delete unused branches of `check_inputs` (`default`, `a`) if grep confirms no callers.

### 4. Deprecation path for `_check_input_or_raise`

- Remove from `evaluation.py` — behavior subsumed by `verify_correlation_inputs`.
- `correlation(a, b, method)` and `tauap_b(a, b)` (in `evaluation.py`) are kept as thin dispatchers that go through the verification path. If they're called outside the `evaluate()` loop, they return `0.0` on any non-OK verification (existing contract) and DO NOT warn (the caller that bypasses `evaluate()` opts out of diagnostics).

## Files affected

| File | Change |
|------|--------|
| `autojudge_evaluate/correlation_verification.py` (new) | `CorrelationIssue` literal, `check_correlation_inputs(a, b, on_fail)` — reuses `OnFail` / `_handle_failure` from `eval_results/verification.py` |
| `autojudge_evaluate/evaluation.py` | `CorrelationTask`, `_prepare_tasks`, refactored `evaluate()`, new `_emit_diagnostic_summary`; remove old `_check_input_or_raise`; thin `tauap_b`/`correlation` dispatchers |
| `autojudge_evaluate/pyircore.py` | Remove `print`, drop `decreasing` param, delete unused `check`/`check_a` branches |
| `autojudge_evaluate/__init__.py` | Export `check_correlation_inputs`, `CorrelationIssue` (public API) |
| `tests/` | New tests: `check_correlation_inputs` table-driven unit tests (one per `CorrelationIssue` value); `_prepare_tasks` snapshot test; `evaluate()` integration tests with tied/too-few inputs |

## Open questions

1. **`eval_raw.name` attribute**: does `EvalResult` expose a judge label? If not, `str(self.truth_leaderboard)` is a weak fallback (labels the truth, not the eval). Worth adding a `name` or `source_path` attribute to `EvalResult` as a prerequisite.
2. **Should `tauap_b` / `correlation` in `evaluation.py` stay as public functions?** Currently they're used by `_compute_single_correlation`. If the refactor removes that method, they might be internal-only. Grep for external callers before deciding.
3. **`top_k` filtering before verification**: is there a case where we'd want to verify BEFORE `top_k` filtering too (to flag "pre-filter all-tied" separately)? Probably no — what matters is what reaches the math layer.
4. **Per-judge vs. per-measure-pair warning granularity**: current plan emits one warning per `evaluate()` call covering all issues. Alternative: group by `(truth_m, eval_m)` for readability. Decide when implementing.

## Non-goals (explicitly out of scope)

- Changing the `0.0`-on-all-tied contract (the user confirmed this is intended behavior).
- Changing the `on_missing="default"` behavior that fills all-zeros (also confirmed intentional).
- Restructuring `extract_ranking` / `_align_rankings` / `get_measure_pairs`.

## Unshelve trigger

Revisit when any of:
- Users report confusion about silent `0.0` in correlation tables.
- A new data-integrity check needs to join the validation surface.
- Someone adds a new correlation method that has its own degenerate-input behavior.