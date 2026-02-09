import click
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from autojudge_evaluate.eval_results import load as load_eval_result, EvalResult

# TODO: Consider unifying with leaderboard.OnMissing which uses "fix_aggregate" instead of "skip"
OnMissing = Literal["error", "warn", "skip", "default"]
EvalResultFormat = Literal["trec_eval", "tot", "ir_measures", "ranking", "jsonl"]
BASE_CORRELATION_METHODS = ["kendall", "pearson", "spearman", "tauap_b"]
TOP_K_VALUES = [10]
CORRELATION_METHODS: List[str] = (
    BASE_CORRELATION_METHODS + 
    ["kendall@10"]
    # [f"{m}@{k}" for m in BASE_CORRELATION_METHODS for k in TOP_K_VALUES]
)


def parse_correlation_method(method: str) -> tuple[str, int | None]:
    """Parse correlation method string into (base_method, top_k).

    Examples:
        "kendall" -> ("kendall", None)
        "kendall@10" -> ("kendall", 10)

    Raises:
        ValueError: if base method is invalid or k is not a positive integer.
    """
    if "@" in method:
        base, k_str = method.split("@", 1)
        if base not in BASE_CORRELATION_METHODS:
            raise ValueError(f"Unknown base method '{base}'")
        k = int(k_str)  # raises ValueError if not int
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        return (base, k)
    if method not in BASE_CORRELATION_METHODS:
        raise ValueError(f"Unknown method '{method}'")
    return (method, None)


class CorrelationMethodType(click.ParamType):
    """Click parameter type for correlation methods (method or method@k)."""
    name = "correlation"

    def convert(self, value, param, ctx):
        try:
            parse_correlation_method(value)
            return value
        except ValueError as e:
            methods = ", ".join(BASE_CORRELATION_METHODS)
            self.fail(
                f"{e}. Valid: [{methods}] or method@k (e.g., kendall@15).",
                param,
                ctx,
            )


class LeaderboardEvaluator():
    """Compute correlation between predicted leaderboards and ground truth."""

    def __init__(
        self,
        truth_leaderboard: Path,
        truth_measures: List[str] | None = None,
        eval_measures: List[str] | None = None,
        on_missing: OnMissing = "error",
        truth_format: EvalResultFormat = "ir_measures",
        truth_has_header: bool = False,
        truth_drop_aggregate: bool = False,
        eval_format: EvalResultFormat = "tot",
        eval_has_header: bool = False,
        eval_drop_aggregate: bool = False,
        correlation_methods: List[str] | None = None,
        topic_ids: set[str] | None = None,
        run_ids: set[str] | None = None,
        only_shared_runs: bool = False,
    ):
        self.on_missing = on_missing
        self.truth_leaderboard = truth_leaderboard
        self.truth_format = truth_format
        self.truth_has_header = truth_has_header
        self.truth_drop_aggregate = truth_drop_aggregate
        self.eval_format = eval_format
        self.eval_has_header = eval_has_header
        self.eval_drop_aggregate = eval_drop_aggregate
        self.truth_measures_filter = truth_measures  # None means all
        self.eval_measures_filter = eval_measures    # None means all
        self.correlation_methods = correlation_methods if correlation_methods else CORRELATION_METHODS
        self.topic_ids = topic_ids  # Pre-determined topics, or None = all
        self.run_ids = run_ids  # Explicit run_ids filter, or None = all
        self.only_shared_runs = only_shared_runs  # Filter to common run_ids (truth ∩ eval)

        # Lazy: truth_result loaded on first access
        self._truth_result: EvalResult | None = None

    @property
    def truth_result(self) -> EvalResult:
        """Lazily load truth result on first access (no filtering applied here)."""
        if self._truth_result is None:
            self._truth_result = self._load_eval_result(
                self.truth_leaderboard, self.truth_format, self.truth_has_header, self.on_missing
            )
            # Validate truth result has enough runs for correlation
            num_runs = len(self._truth_result.run_ids)
            if num_runs < 3:
                raise ValueError(
                    f"Truth result has only {num_runs} run(s). "
                    f"Correlation requires at least 3 runs. "
                    f"Did you pass a single file instead of a directory?"
                )
        return self._truth_result

    def _handle_missing(self, msg: str) -> bool:
        """Handle missing data according to on_missing policy.

        Returns True if processing should continue (warn/skip/default),
        raises ValueError if on_missing == "error".
        """
        if self.on_missing == "error":
            raise ValueError(msg)
        elif self.on_missing == "warn":
            print(f"Warning: {msg}", file=sys.stderr)
        return True  # continue for warn/skip/default

    def _load_eval_result(
        self,
        path: Path,
        format: EvalResultFormat,
        has_header: bool = False,
        on_missing: OnMissing = "error",
    ) -> EvalResult:
        """Load EvalResult without filtering. All filtering consolidated in evaluate()."""
        if not path or not Path(path).exists():
            raise ValueError(f"Path does not exist: {path}")

        # Map on_missing: "skip"/"default" -> "ignore"
        eval_on_missing = "ignore" if on_missing in ("skip", "default") else on_missing

        return load_eval_result(
            Path(path),
            format=format,
            has_header=has_header,
            drop_aggregates=False,
            recompute_aggregates=False,
            verify=True,
            on_missing=eval_on_missing,
        )

    def extract_ranking(self, eval_result: EvalResult, measure: str) -> Dict[str, float] | None:
        """Extract run_id -> value mapping for aggregate rows (topic_id == ALL_TOPIC_ID).

        Returns None if measure is missing and on_missing is skip/warn.
        Returns all 0.0 if measure is missing and on_missing is default.
        """
        if measure not in eval_result.measures:
            self._handle_missing(f"Measure '{measure}' not found in result")
            if self.on_missing == "default":
                # Fill all runs with 0.0
                return {run_id: 0.0 for run_id in eval_result.run_ids}
            return None  # skip for warn/skip
        return eval_result.get_aggregate_ranking(measure)

    def get_measure_pairs(
        self, eval_result: EvalResult
    ) -> List[Tuple[str, str]]:
        """Get list of (truth_measure, eval_measure) pairs to analyze."""
        # Determine eval measures to use
        eval_measures = sorted(eval_result.measures)
        if self.eval_measures_filter:
            missing = set(self.eval_measures_filter) - eval_result.measures
            if missing:
                raise ValueError(f"Eval measures not found in eval result: {sorted(missing)}")
            eval_measures = [m for m in eval_measures if m in self.eval_measures_filter]

        # Determine truth measures to use
        truth_measures = sorted(self.truth_result.measures)
        if self.truth_measures_filter:
            missing = set(self.truth_measures_filter) - self.truth_result.measures
            if missing:
                raise ValueError(f"Truth measures not found in truth result: {sorted(missing)}")
            truth_measures = [m for m in truth_measures if m in self.truth_measures_filter]

        # Generate all pairs
        return [(tm, em) for tm in truth_measures for em in eval_measures]

    def evaluate(
        self,
        eval_file: Path,
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Evaluate and return dict keyed by (truth_measure, eval_measure) pairs.

        Args:
            eval_file: Path to the evaluated result file

        Returns:
            Dict mapping (truth_measure, eval_measure) pairs to correlation results.
        """
        # =======================================================================
        # LOAD: Raw data without filtering
        # =======================================================================
        truth_raw = self.truth_result  # Lazily loaded, no filtering
        eval_raw = self._load_eval_result(
            eval_file, self.eval_format, self.eval_has_header, self.on_missing
        )

        # =======================================================================
        # FILTER: Two independent dimensions
        #
        # 1. Topic filtering → requires filter_topics_and_recompute()
        #    - topic_ids specified (--topic-id, --only-shared-topics)
        #    - explicit --recompute-aggregates flag (even without topic filtering)
        #
        # 2. Run filtering → uses filter_runs() (NO recompute)
        #    - explicit run_ids (--run-id)
        #    - common run_ids (--only-shared-runs)
        #    - @k methods implicitly filter to common runs
        # =======================================================================

        # Step 1: Topic filtering / recompute (if needed)
        needs_recompute = (
            self.topic_ids is not None  # --topic-id or --only-shared-topics
            or self.truth_drop_aggregate  # --recompute-aggregates for truth
        )
        if needs_recompute:
            truth_filtered = truth_raw.filter_topics_and_recompute(topic_ids=self.topic_ids)
        else:
            truth_filtered = truth_raw

        needs_eval_recompute = (
            self.topic_ids is not None
            or self.eval_drop_aggregate  # --recompute-aggregates for eval
        )
        if needs_eval_recompute:
            eval_filtered = eval_raw.filter_topics_and_recompute(topic_ids=self.topic_ids)
        else:
            eval_filtered = eval_raw

        # Step 2: Run filtering (no recompute)
        has_top_k_methods = any("@" in m for m in self.correlation_methods)

        # Compute valid_run_ids (intersection of all constraints)
        valid_run_ids: set[str] | None = None

        if self.run_ids is not None:
            valid_run_ids = set(self.run_ids)

        if self.only_shared_runs or has_top_k_methods:
            common_run_ids = set(truth_filtered.run_ids) & set(eval_filtered.run_ids)
            valid_run_ids = common_run_ids if valid_run_ids is None else valid_run_ids & common_run_ids

        # Apply run filtering to EvalResults
        if valid_run_ids is not None:
            truth_filtered = truth_filtered.filter_runs(valid_run_ids)
            eval_filtered = eval_filtered.filter_runs(valid_run_ids)

        # =======================================================================
        # COMPUTE: Correlations for each measure pair and method
        # =======================================================================
        ret = {}

        for truth_m, eval_m in self.get_measure_pairs(eval_raw):
            correlations = {}

            for method in self.correlation_methods:
                base_method, top_k = parse_correlation_method(method)

                # Extract rankings from aggregates (already filtered by topic and run)
                truth_ranking = self.extract_ranking(truth_filtered, truth_m)
                eval_ranking = self.extract_ranking(eval_filtered, eval_m)

                if truth_ranking is None or eval_ranking is None:
                    continue

                # @k filtering: select top k runs by truth ranking
                if top_k is not None:
                    sorted_runs = sorted(truth_ranking.items(), key=lambda x: x[1], reverse=True)
                    top_run_ids = {r for r, _ in sorted_runs[:top_k]}
                    truth_ranking = {r: v for r, v in truth_ranking.items() if r in top_run_ids}
                    eval_ranking = {r: v for r, v in eval_ranking.items() if r in top_run_ids}

                correlations[method] = self._compute_single_correlation(
                    truth_ranking, eval_ranking, base_method
                )

            ret[(truth_m, eval_m)] = correlations

        return ret

    def evaluate_old(
        self,
        eval_file: Path,
    ) -> Dict[Tuple[str, str], Dict]:
        """
        [BACKUP] Old evaluate implementation.

        Evaluate and return dict keyed by (truth_measure, eval_measure) pairs.

        Args:
            eval_file: Path to the evaluated result file

        Returns:
            Dict mapping (truth_measure, eval_measure) pairs to correlation results.
            Each method (e.g., "kendall", "kendall@10") is computed with its own
            top-k filtering based on the @k suffix.

        Filtering order:
            1. Filter truth and eval to common runs (intersection for fair comparison)
            2. If topic_ids specified (--only-shared-topics): filter to those topics, recompute
               Otherwise (--all-topics): use provided aggregates, no topic filtering
            For kendall@k specifically:
            3. Get top k runs from truth's aggregates
            4. Filter eval to top k runs (and topics if specified), recompute
            5. Filter truth to top k runs, recompute
            6. Compare rankings
        """
        # =======================================================================
        # LOAD: Raw data without filtering
        # =======================================================================
        truth_raw = self.truth_result  # Lazily loaded, no filtering
        eval_raw = self._load_eval_result(
            eval_file, self.eval_format, self.eval_has_header, self.on_missing
        )

        # =======================================================================
        # FILTER: All filtering decisions consolidated here
        # =======================================================================

        # Determine what filtering is needed
        has_top_k_methods = any("@" in m for m in self.correlation_methods)
        needs_topic_filter = self.topic_ids is not None
        needs_run_filter = has_top_k_methods  # @k requires filtering to common runs first

        # Apply drop_aggregate flags (CLI flags)
        if self.truth_drop_aggregate:
            truth_base = truth_raw.filter_and_recompute()  # Drop and recompute
        else:
            truth_base = truth_raw

        if self.eval_drop_aggregate:
            eval_base = eval_raw.filter_and_recompute()  # Drop and recompute
        else:
            eval_base = eval_raw

        # For @k methods: filter to common runs upfront
        if needs_run_filter:
            common_run_ids = set(truth_base.run_ids) & set(eval_base.run_ids)
            truth_for_topk = truth_base.filter_and_recompute(
                topic_ids=self.topic_ids,
                run_ids=common_run_ids,
            )
            # Warn about measures lost during filtering
            lost_measures = truth_base.measures - truth_for_topk.measures
            if lost_measures:
                print(
                    f"Warning: {len(lost_measures)} measure(s) unavailable after filtering "
                    f"(no per-topic data): {sorted(lost_measures)}",
                    file=sys.stderr
                )
        else:
            truth_for_topk = None  # Not needed for non-@k methods

        # =======================================================================
        # COMPUTE: Correlations for each measure pair and method
        # =======================================================================
        ret = {}

        for truth_m, eval_m in self.get_measure_pairs(eval_raw):
            correlations = {}

            for method in self.correlation_methods:
                base_method, top_k = parse_correlation_method(method)

                if top_k is not None:
                    # @k method: use pre-filtered truth_for_topk
                    if truth_m not in truth_for_topk.measures:
                        self._handle_missing(f"Measure '{truth_m}' not found for top-k ranking")
                        continue

                    top_run_ids = set(truth_for_topk.top_k_run_ids(truth_m, top_k))

                    # Filter eval to top k runs (and topics if specified)
                    eval_for_method = eval_base.filter_and_recompute(
                        topic_ids=self.topic_ids,
                        run_ids=top_run_ids
                    )

                    # Filter truth to top k runs
                    truth_for_method = truth_for_topk.filter_and_recompute(run_ids=top_run_ids)
                else:
                    # Non-@k method: preserve original aggregates (v1 behavior)
                    truth_for_method = truth_base

                    if needs_topic_filter:
                        eval_for_method = eval_base.filter_and_recompute(
                            topic_ids=self.topic_ids,
                        )
                    else:
                        eval_for_method = eval_base

                truth_ranking = self.extract_ranking(truth_for_method, truth_m)
                eval_ranking = self.extract_ranking(eval_for_method, eval_m)

                if truth_ranking is None or eval_ranking is None:
                    continue

                correlations[method] = self._compute_single_correlation(
                    truth_ranking, eval_ranking, base_method
                )

            ret[(truth_m, eval_m)] = correlations

        return ret

    def _align_rankings(
        self, truth_ranking: Dict[str, float], eval_ranking: Dict[str, float]
    ) -> Tuple[List[float], List[float]]:
        """Align truth and eval rankings, handling missing systems per on_missing policy."""
        a, b = [], []

        truth_systems = set(truth_ranking.keys())
        eval_systems = set(eval_ranking.keys())

        missing_in_eval = truth_systems - eval_systems
        missing_in_truth = eval_systems - truth_systems

        if missing_in_eval or missing_in_truth:
            msg_parts = []
            if missing_in_eval:
                msg_parts.append(f"missing in evaluated: {sorted(missing_in_eval)}")
            if missing_in_truth:
                msg_parts.append(f"missing in ground truth: {sorted(missing_in_truth)}")
            msg = f"Run ID mismatch: {'; '.join(msg_parts)}. \nShared RunIDs: {sorted(truth_systems & eval_systems)}"

            self._handle_missing(msg)

        # Include all systems from ground truth
        for system in truth_systems:
            a.append(float(truth_ranking[system]))
            if system in eval_ranking:
                b.append(float(eval_ranking[system]))
            elif self.on_missing == "default":
                b.append(0.0)
            else:
                # skip: only include common systems
                a.pop()  # remove the truth value we just added

        return a, b

    def _compute_single_correlation(
        self, truth_ranking: Dict[str, float], eval_ranking: Dict[str, float], base_method: str
    ) -> float:
        """Compute a single correlation between aligned rankings."""
        a, b = self._align_rankings(truth_ranking, eval_ranking)

        if base_method == "tauap_b":
            return tauap_b(a, b)
        else:
            return correlation(a, b, base_method)


def _check_input_or_raise(a, b):
    if len(a) < 3:
        raise ValueError(f"Can not calculate correlations on only {len(a)} elements.")
    if len(a) != len(b):
        raise ValueError(f"Can not calculate correlations on unequal elements: {len(a)} != {len(b)}")

def correlation(a, b, method):
    _check_input_or_raise(a, b)

    df = pd.DataFrame([{"a": i, "b": j} for i, j in zip(a, b)])

    return float(df.corr(method).iloc[0]["b"])

def tauap_b(a, b):
    from autojudge_evaluate.pyircore import tauap_b as method

    _check_input_or_raise(a, b)
    return method(a, b)
