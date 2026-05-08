#!/usr/bin/env python3
"""Cross-model comparison runner.

Supports two modes:
  1. Run tests then generate report (--models)
  2. Generate report from existing results (--from-reports)

Usage:
    # --- Mode 1: Run tests and generate comparison report ---

    # Compare two models (default API endpoint):
    python run_comparison.py --models "Qwen3-VL-235B-A22B-Instruct,gpt-4o"

    # With custom API endpoints per model:
    python run_comparison.py \\
        --models "Qwen3-VL-235B-A22B-Instruct,gpt-4o" \\
        --base-urls "http://120.48.75.178:4970/v1,https://api.openai.com/v1"

    # Include stability tests (slower):
    python run_comparison.py --models "Qwen3-VL-235B-A22B-Instruct" --include-stability

    # Quick debug run (1-2 cases per file, verify pipeline end-to-end):
    python run_comparison.py --models "Qwen3-VL-235B-A22B-Instruct" --debug

    # --- Mode 2: Generate report from existing JSON results ---

    # From specific report files:
    python run_comparison.py --from-reports report/ModelA_20260421_120000/report.json report/ModelB_20260421_120000/report.json

    # Auto-discover all report/*/report.json in report/ directory:
    python run_comparison.py --from-reports

Environment variables:
    TEST_BASE_URL     Default LLM API URL (used if --base-urls not specified)
    TEST_API_KEY      API key for authentication
    STABILITY_REPEATS Number of repetitions for stability tests (default 3)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TESTS_DIR = Path(__file__).parent
REPORT_DIR = TESTS_DIR / "report"

_DEFAULT_WEIGHTS = {
    "tool_sequence": 0.35,
    "response_quality": 0.25,
    "param_accuracy": 0.25,
    "efficiency": 0.15,
}
_DIMS = ["tool_sequence", "response_quality", "param_accuracy", "efficiency"]


def _composite_from_scores(scores: dict) -> float | None:
    total = 0.0
    weight_sum = 0.0
    for dim in _DIMS:
        val = scores.get(dim, {}).get("score")
        if isinstance(val, (int, float)):
            total += val * _DEFAULT_WEIGHTS[dim]
            weight_sum += _DEFAULT_WEIGHTS[dim]
    if weight_sum == 0:
        return None
    return round(total / weight_sum, 4)

DEBUG_TESTS = [
    "test_fetch_yellow_cup",
    "test_fold_blanket",
    "test_list_table_items",
    "test_fetch_cup_move_to_coords",
    "test_fetch_cup_quality",
    "test_stability_fetch_cup",
]


def run_model_tests(
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    include_stability: bool = False,
    verbose: bool = False,
    debug: bool = False,
    extra_pytest_args: list[str] | None = None,
) -> tuple[int, Path | None]:
    env = os.environ.copy()
    env["TEST_MODEL"] = model
    if base_url:
        env["TEST_BASE_URL"] = base_url
    if api_key:
        env["TEST_API_KEY"] = api_key
    if verbose:
        env["TEST_VERBOSE"] = "1"
    if debug:
        env["STABILITY_REPEATS"] = "1"

    test_files = [
        "test_task_planning.py",
        "test_function_calling.py",
        "test_response_quality.py",
    ]
    if include_stability:
        test_files.append("test_stability.py")

    cmd = [
        sys.executable, "-m", "pytest",
        *[str(TESTS_DIR / f) for f in test_files],
        "-v",
        "--tb=short",
    ]
    if debug:
        k_expr = " or ".join(DEBUG_TESTS)
        cmd.extend(["-k", k_expr])
    cmd.extend(extra_pytest_args or [])

    label = "  [DEBUG] " if debug else "  "
    print(f"\n{'='*70}")
    print(f"{label}Running tests for: {model}")
    if base_url:
        print(f"{label}API: {base_url}")
    print(f"{label}Tests: {', '.join(test_files)}")
    if debug:
        print(f"{label}Debug 模式: 每个文件仅运行 1~2 个用例, stability_repeats=1")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, env=env, cwd=str(TESTS_DIR))

    json_path = REPORT_DIR / model / "report.json"
    if json_path.exists():
        return result.returncode, json_path
    legacy_json_path = REPORT_DIR / f"eval_report_{model}.json"
    if legacy_json_path.exists():
        return result.returncode, legacy_json_path
    return result.returncode, None


def load_model_report(json_path: Path) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def generate_comparison_report(
    model_reports: dict[str, list[dict]],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_test_names = []
    seen = set()
    for records in model_reports.values():
        for rec in records:
            name = rec["test"]
            if name not in seen:
                all_test_names.append(name)
                seen.add(name)

    comparison_data = {
        "generated_at": datetime.now().isoformat(),
        "models": list(model_reports.keys()),
        "tests": all_test_names,
        "results": {},
    }
    for model, records in model_reports.items():
        comparison_data["results"][model] = {
            "total": len(records),
            "passed": sum(1 for r in records if r.get("outcome") == "passed"),
            "failed": sum(1 for r in records if r.get("outcome") == "failed"),
            "records": records,
        }

    json_path = output_dir / f"comparison_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)

    md_content = _format_comparison_markdown(model_reports, all_test_names)
    md_path = output_dir / f"comparison_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path


def _format_comparison_markdown(
    model_reports: dict[str, list[dict]],
    all_test_names: list[str],
) -> str:
    models = list(model_reports.keys())

    model_stats: dict[str, dict] = {}
    for model, records in model_reports.items():
        total = len(records)
        passed = sum(1 for r in records if r.get("outcome") == "passed")
        model_stats[model] = {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0,
        }

    lines = [
        "# 真实居家场景跨模型对比评估报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**对比模型**: {', '.join(f'`{m}`' for m in models)}",
        "",
        "## 1. 总体对比",
        "",
    ]

    header = "| 指标 | " + " | ".join(f"`{m}`" for m in models) + " |"
    sep = "|------|" + "|".join(["------"] * len(models)) + "|"
    lines.extend([header, sep])

    lines.append("| 总用例数 | " + " | ".join(str(model_stats[m]["total"]) for m in models) + " |")
    lines.append("| 通过数 | " + " | ".join(str(model_stats[m]["passed"]) for m in models) + " |")
    lines.append("| **通过率** | " + " | ".join(f"**{model_stats[m]['pass_rate']:.1%}**" for m in models) + " |")

    dims = ["tool_sequence", "response_quality", "param_accuracy", "efficiency"]
    dim_labels = {"tool_sequence": "工具序列", "response_quality": "回复质量", "param_accuracy": "参数准确", "efficiency": "效率"}

    model_test_lookup: dict[str, dict[str, dict]] = {}
    model_dim_scores: dict[str, dict[str, list[float]]] = {}
    model_composites: dict[str, list[float]] = {}
    for model, records in model_reports.items():
        lookup = {}
        dim_scores: dict[str, list[float]] = {d: [] for d in dims}
        composites: list[float] = []
        for rec in records:
            lookup[rec["test"]] = rec
            if not rec["test"].startswith("stability_"):
                for dim in dims:
                    score_val = rec.get("scores", {}).get(dim, {}).get("score")
                    if isinstance(score_val, (int, float)):
                        dim_scores[dim].append(score_val)
                comp = _composite_from_scores(rec.get("scores", {}))
                if comp is not None:
                    composites.append(comp)
        model_test_lookup[model] = lookup
        model_dim_scores[model] = dim_scores
        model_composites[model] = composites

    comp_vals = []
    for m in models:
        cs = model_composites[m]
        comp_vals.append(f"**{sum(cs)/len(cs):.2%}**" if cs else "-")
    lines.append("| **综合得分** | " + " | ".join(comp_vals) + " |")

    for dim in dims:
        label = dim_labels.get(dim, dim)
        vals = []
        for m in models:
            scores = model_dim_scores[m][dim]
            if scores:
                avg = sum(scores) / len(scores)
                vals.append(f"{avg:.2%}")
            else:
                vals.append("-")
        lines.append(f"| 平均{label} | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("> 综合得分 = 各用例加权得分的平均值（权重: 工具序列 35%, 回复质量 25%, 参数准确 25%, 效率 15%）")

    lines.extend(["", "## 2. 逐用例对比", ""])

    header = "| 测试用例 | " + " | ".join(f"`{m}`" for m in models) + " |"
    sep = "|---------|" + "|".join(["------"] * len(models)) + "|"
    lines.extend([header, sep])

    for test_name in all_test_names:
        if test_name.startswith("stability_"):
            continue
        vals = []
        for m in models:
            rec = model_test_lookup[m].get(test_name)
            if rec:
                outcome = rec.get("outcome", "?")
                symbol = "pass" if outcome == "passed" else "fail" if outcome == "failed" else "-"
                comp = _composite_from_scores(rec.get("scores", {}))
                if comp is not None:
                    vals.append(f"{symbol} {comp:.0%}")
                else:
                    vals.append(symbol)
            else:
                vals.append("-")
        lines.append(f"| {test_name} | " + " | ".join(vals) + " |")

    return "\n".join(lines)


def _discover_report_files() -> list[Path]:
    if not REPORT_DIR.exists():
        return []
    run_reports = sorted(REPORT_DIR.glob("*/report.json"))
    if run_reports:
        return run_reports
    return sorted(REPORT_DIR.glob("eval_report_*.json"))


def _extract_model_name(json_path: Path) -> str:
    try:
        records = load_model_report(json_path)
        if records and isinstance(records[0], dict):
            model = records[0].get("model")
            if model:
                return model
    except Exception:
        pass
    if json_path.name == "report.json" and json_path.parent != REPORT_DIR:
        return json_path.parent.name
    stem = json_path.stem
    if stem.startswith("eval_report_"):
        return stem[len("eval_report_"):]
    return stem


def _run_from_reports(report_paths: list[Path]) -> None:
    model_reports: dict[str, list[dict]] = {}

    for path in report_paths:
        if not path.exists():
            print(f"  警告: 文件不存在，跳过: {path}")
            continue
        model_name = _extract_model_name(path)
        records = load_model_report(path)
        if model_name in model_reports:
            print(f"  警告: 模型 '{model_name}' 重复出现，后者覆盖前者")
        model_reports[model_name] = records
        passed = sum(1 for r in records if r.get("outcome") == "passed")
        total = len(records)
        rate = f"{passed}/{total} ({passed/total:.0%})" if total else "无数据"
        print(f"  已加载: {model_name} — {rate} (来源: {path.name})")

    if not model_reports:
        print("\n错误: 没有找到有效的报告文件")
        sys.exit(1)

    json_path, md_path = generate_comparison_report(model_reports, REPORT_DIR)

    print(f"\n{'='*70}")
    print("  跨模型对比报告已生成:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="真实居家场景跨模型对比测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--models", help="逗号分隔的模型名列表")
    mode_group.add_argument("--from-reports", nargs="*", default=None,
        help="从已有 JSON 报告生成对比")

    parser.add_argument("--base-urls", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--include-stability", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("pytest_args", nargs="*")
    args = parser.parse_args()

    if args.from_reports is not None:
        if args.from_reports:
            report_paths = [Path(p) for p in args.from_reports]
        else:
            report_paths = _discover_report_files()
            if not report_paths:
                print(f"错误: report/ 目录下没有找到可用报告文件")
                sys.exit(1)
            print(f"自动发现 {len(report_paths)} 个报告文件:")
            for p in report_paths:
                print(f"  {p.name}")
        _run_from_reports(report_paths)
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    base_urls = (
        [u.strip() for u in args.base_urls.split(",")]
        if args.base_urls else [None] * len(models)
    )
    if len(base_urls) == 1 and len(models) > 1:
        base_urls = base_urls * len(models)

    if len(base_urls) != len(models):
        print(f"错误: --base-urls 数量 ({len(base_urls)}) 与 --models 数量 ({len(models)}) 不匹配")
        sys.exit(1)

    model_reports: dict[str, list[dict]] = {}
    exit_codes: dict[str, int] = {}

    for model, base_url in zip(models, base_urls):
        exit_code, json_path = run_model_tests(
            model=model,
            base_url=base_url,
            api_key=args.api_key,
            include_stability=args.include_stability,
            verbose=args.verbose,
            debug=args.debug,
            extra_pytest_args=args.pytest_args,
        )
        exit_codes[model] = exit_code
        if json_path and json_path.exists():
            model_reports[model] = load_model_report(json_path)
        else:
            model_reports[model] = []

    json_path, md_path = generate_comparison_report(model_reports, REPORT_DIR)

    print(f"\n{'='*70}")
    print("  跨模型对比报告已生成:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    print(f"{'='*70}")

    any_failed = any(c != 0 for c in exit_codes.values())
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
