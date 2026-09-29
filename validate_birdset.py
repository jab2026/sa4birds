"""Command-line front end for BirdSet evaluation.

    python validate_birdset.py --mode=DT --down_task=HSN
    python validate_birdset.py --mode=LT --down_task=ALL

The evaluation itself lives in sa4birds.evaluation; this module only parses
arguments, expands --down_task=ALL, and logs the results.
"""
import argparse
import logging

from registry import DT, MT, LT, LT_SSA
from sa4birds.evaluation import get_device, run_testing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# BirdSet Down tasks
TASKS = ["HSN", "POW", "NES", "NBP", "UHH", "SNE", "SSW", "PER"]

REGIMES = ["DT", "MT", "LT", "LT_SSA"]


def parse_args():
    """Parse the command-line arguments (see --help)."""
    parser = argparse.ArgumentParser(
        description="Run evaluation for a selected downstream task under a specified training regime.")
    parser.add_argument(
        "--mode",
        choices=REGIMES,
        default="DT",
        help="Regime to use: DT, MT, LT, or LT_SSA (default: DT)"
    )
    parser.add_argument(
        "--down_task",
        choices=["ALL"] + TASKS,
        default="HSN",
        help="Specify which downstream task to execute; use ALL to run every task (default: HSN)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Force computation on CPU instead of GPU (default: use GPU if available)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of worker processes used for loading data batches in parallel (default: 6)"
    )

    return parser.parse_args()


def run_tasks(mode, task, device, num_workers):
    """
    Execute evaluation for one or multiple downstream tasks.

    This function runs testing under a specified training regime
    (DT, MT, LT, or LT_SSA) and logs evaluation results for each task.

    Parameters
    ----------
    mode : str
      Training regime: "DT", "MT", "LT" or "LT_SSA" (a dictionary in registry.py).
    task : str
      Downstream task name or "ALL".
      - If "ALL", evaluation runs on every task in TASKS.
      - Otherwise, runs only the specified task.
    device : torch.device
      Device used for computation (CPU or CUDA).
    num_workers : int
      DataLoader workers, passed through to run_testing.

    Returns
    -------
    list[dict]
      List of result dictionaries, one per evaluated task.
      Each result contains:
          {
              <task_name>: {
                  "auroc": float,
                  "cmap": float,
                  "top1_acc": float
              }
          }
    """
    tasks = TASKS if task == "ALL" else [task]
    logger.info("Running evaluation")
    logger.info("Regime: %s | Device: %s | Tasks: %s", mode, device, tasks)
    results = []
    for t in tasks:
        result = run_testing(mode=globals()[mode], down_task=t, device=device,
                             num_workers=num_workers)
        results.append(result)
        logger.info("Regime: %s | Task: %s | AUROC: %s | CMAP: %s | TOP1-ACC: %s ",
                    mode, t, result[t]['auroc'], result[t]['cmap'], result[t]['top1_acc'])
    return results


if __name__ == '__main__':
    args = parse_args()
    device = get_device(args.cpu)
    results = run_tasks(args.mode, args.down_task, device, args.num_workers)
    logger.info(results)
