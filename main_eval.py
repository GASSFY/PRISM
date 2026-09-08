"""
PRISM Phase-1 评估入口：FP16 或伪量化 float checkpoint + lmms-eval。
"""
import argparse
import json
import os
import sys
import warnings
from typing import Any, Union

import numpy as np
import torch
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

from lmms_eval import evaluator, utils
from lmms_eval.models import get_model
from lmms_eval.tasks import TaskManager
from asdq.quantization.eval_load import load_model_for_eval


def _handle_non_serializable(o):
    if isinstance(o, (np.integer, np.int64, np.int32)):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, set):
        return list(o)
    return str(o)


def _append_results_md(md_path: str, args: argparse.Namespace, results: dict) -> None:
    """Append evaluation results to a markdown file."""
    scale_path = getattr(args, "scale_path", None)
    if scale_path and os.path.exists(scale_path):
        theta1 = getattr(args, "asd_theta1", "N/A")
        theta2 = getattr(args, "asd_theta2", "N/A")
        ratio = getattr(args, "asd_high_precision_ratio", "N/A")
        w_bit = getattr(args, "w_bit", getattr(args, "asd_low_w_bit", "N/A"))
        section_title = f"\n## theta1={theta1}, theta2={theta2}, ratio={ratio}, w_bit={w_bit}\n"
    else:
        section_title = "\n## FP16 baseline (no quantization)\n"

    lines: list[str] = []
    lines.append(section_title)

    # Main metrics table from lmms-eval
    table_str = evaluator.make_table(results)
    if table_str:
        lines.append("```")
        lines.append(table_str.strip())
        lines.append("```\n")

    # Group-level table if available
    if "groups" in results:
        groups_table = evaluator.make_table(results, "groups")
        if groups_table:
            lines.append("```")
            lines.append(groups_table.strip())
            lines.append("```\n")

    # Per-subject details from results["results"] or logs
    raw_results = results.get("results", {})
    for task_name, task_metrics in raw_results.items():
        if not isinstance(task_metrics, dict):
            continue
        for metric_key, metric_val in task_metrics.items():
            if "acc" in metric_key and isinstance(metric_val, (int, float)):
                lines.append(f"- **{task_name}** {metric_key} = {metric_val:.5f}")

    # If logs contain per-subject info, extract and write it
    logs = results.get("logs", {})
    for task_name, task_logs in logs.items():
        if not isinstance(task_logs, list) or not task_logs:
            continue
        first = task_logs[0]
        if isinstance(first, dict) and "mmmu_acc" in first:
            from collections import defaultdict
            subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
            for entry in task_logs:
                subj = entry.get("mmmu_acc", {}).get("subject", "Unknown")
                score = entry.get("mmmu_acc", {}).get("score", 0)
                subject_stats[subj]["total"] += 1
                subject_stats[subj]["correct"] += score
            if subject_stats:
                lines.append("\n| Subject | Num | Acc |")
                lines.append("|---------|-----|-----|")
                for subj in sorted(subject_stats.keys()):
                    s = subject_stats[subj]
                    acc = s["correct"] / s["total"] if s["total"] else 0
                    lines.append(f"| {subj} | {s['total']} | {acc:.5f} |")

    lines.append("\n---\n")

    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    with open(md_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[PRISM] Results appended to {md_path}")


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="PRISM Phase-1: evaluate FP16 / pseudo-quant model via lmms-eval.",
    )
    parser.add_argument("--config", default="", help="Path to yaml config")
    parser.add_argument("--model", default="llava_onevision")
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names, e.g. mmmu_val or mme,mmb")
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--batch_size", "-b", type=str, default="1")
    parser.add_argument("--max_batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None, type=str)
    parser.add_argument("--limit", type=float, default=None, help="Limit samples per task (for testing)")
    parser.add_argument("--use_cache", type=str, default=None)
    parser.add_argument("--cache_requests", type=str, default=None, choices=["true", "refresh", "delete"])
    parser.add_argument("--write_out", "-w", action="store_true")
    parser.add_argument("--log_samples", action="store_true")
    parser.add_argument("--gen_kwargs", default="")
    parser.add_argument("--verbosity", type=str, default="INFO")
    parser.add_argument("--include_path", type=str, default=None)
    parser.add_argument("--seed", type=str, default="0,1234,1234,1234")
    parser.add_argument(
        "--scale_path",
        default=None,
        type=str,
        help="Path to pseudo-quant float state_dict from main_quant.py",
    )
    parser.add_argument(
        "--pseudo_quant",
        action="store_true",
        default=True,
        help="Load pseudo-quant float checkpoint (default True when scale_path exists).",
    )
    parser.add_argument("--results_md", default=None, type=str, help="Path to markdown file for appending results")
    args = parser.parse_args()
    return args


def _apply_config(args: argparse.Namespace, config: dict) -> None:
    for k, v in config.items():
        setattr(args, k, v)


def _is_cli_explicit(field: str, argv: list[str]) -> bool:
    """Whether a field is explicitly provided via CLI."""
    flag = f"--{field.replace('_', '-')}"
    return flag in argv


def _merge_config_with_cli_priority(
    base_args: argparse.Namespace,
    config: dict,
    argv: list[str],
) -> argparse.Namespace:
    """
    Merge YAML config into parsed args, while preserving explicit CLI overrides.
    """
    args_copy = argparse.Namespace(**vars(base_args))
    explicit_fields = {"output_path", "tasks", "results_md"}
    preserved_values = {
        k: getattr(base_args, k)
        for k in explicit_fields
        if _is_cli_explicit(k, argv)
    }
    _apply_config(args_copy, config)
    for k, v in preserved_values.items():
        setattr(args_copy, k, v)
    return args_copy


def _parse_seed(seed_str: str) -> tuple:
    parts = seed_str.replace(" ", "").split(",")
    if len(parts) == 1:
        try:
            v = int(parts[0])
            return (v, v, v, v)
        except ValueError:
            return (0, 1234, 1234, 1234)
    out = []
    for p in parts[:4]:
        try:
            out.append(int(p))
        except ValueError:
            out.append(1234)
    while len(out) < 4:
        out.append(1234)
    return tuple(out)


def run_eval(args: argparse.Namespace) -> dict | None:
    if args.tasks is None or args.tasks.strip() == "":
        print("Please specify --tasks (e.g. mmmu_val or mme,mmb). Use --tasks list to list all.")
        if args.tasks == "list":
            task_manager = TaskManager(args.verbosity, include_path=args.include_path, model_name=args.model)
            print("Available tasks:", task_manager.list_all_tasks())
        return None

    task_manager = TaskManager(args.verbosity, include_path=args.include_path, model_name=args.model)
    task_list = [t.strip() for t in args.tasks.split(",")]
    task_names = task_manager.match_tasks(task_list)
    missing = [t for t in task_list if t not in task_names and "*" not in t]
    if missing:
        raise ValueError(f"Tasks not found: {missing}. Try --tasks list.")

    if args.model_args is None:
        args.model_args = ""

    ModelClass = get_model(args.model)
    lm = load_model_for_eval(ModelClass, args)

    seeds = _parse_seed(getattr(args, "seed", "0,1234,1234,1234"))
    import random
    random.seed(seeds[0])
    np.random.seed(seeds[1])
    torch.manual_seed(seeds[2])

    # output_path for submission files (mmbench, ocrbench, etc.) is passed via cli_args=args; simple_evaluate() does not accept output_path.
    results = evaluator.simple_evaluate(
        model=args.model,
        lm=lm,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=False,
        write_out=args.write_out,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity=args.verbosity,
        random_seed=seeds[0],
        numpy_random_seed=seeds[1],
        torch_random_seed=seeds[2],
        fewshot_random_seed=seeds[3],
        cli_args=args,
    )

    if results:
        print(evaluator.make_table(results))
        if "groups" in results:
            print(evaluator.make_table(results, "groups"))
        if args.output_path:
            out_file = args.output_path if args.output_path.endswith(".json") else os.path.join(args.output_path, "results.json")
            os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=_handle_non_serializable)
            print(f"Results saved to {out_file}")
        md_path = getattr(args, "results_md", None)
        if md_path:
            _append_results_md(md_path, args, results)
    return results


def cli_main(args: Union[argparse.Namespace, None] = None) -> None:
    if args is None:
        args = parse_eval_args()
    argv = sys.argv[1:]

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg = [cfg] if not isinstance(cfg, list) else cfg
        for c in cfg:
            args_copy = _merge_config_with_cli_priority(args, c, argv)
            run_eval(args_copy)
    else:
        run_eval(args)


if __name__ == "__main__":
    cli_main()
