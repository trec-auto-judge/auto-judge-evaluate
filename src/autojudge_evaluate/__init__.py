"""
autojudge_evaluate - Evaluation tools for TREC AutoJudge systems.

Provides:
- evaluation.py: LeaderboardEvaluator for correlation analysis
- eval_results/: EvalResult containers and I/O
- nugget_doc_eval.py: Nugget-document evaluation
- analysis/: Correlation tables and plots
- _commands/: CLI commands (meta-evaluate, qrel-evaluate, leaderboard, eval-result)
"""

__version__ = '0.1.0'

from click import group

from ._commands._meta_evaluate import meta_evaluate
from ._commands._leaderboard import leaderboard
from ._commands._eval_result import eval_result
from ._commands._qrel_evaluate import qrel_evaluate


@group()
def main():
    pass


main.command("meta-evaluate")(meta_evaluate)
main.command("leaderboard")(leaderboard)
main.add_command(eval_result)
main.add_command(qrel_evaluate, "qrel-evaluate")
