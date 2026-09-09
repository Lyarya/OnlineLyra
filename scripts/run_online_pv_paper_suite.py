"""Run the Online Lyra PV paper experiment suite sequentially.

This script is designed for server execution. Each experiment gets its own
output directory containing:
  * run.log
  * online_pv_summary.csv
  * online_pv_metadata.json
  * per-farm/horizon training logs, predictions, and checkpoints
  * generated paper tables and figures
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


BASE_ARGS = {
    "data_dir": "solar_stations",
    "glob": "Solar station site*.xlsx",
    "target_col": "Power (MW)",
    "horizons": [4, 12, 24, 48, 96],
    "lookback": 96,
    "offline_days": 91,
    "year_days": 365,
    "online_steps": 2,
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "replay_capacity": 1000,
    "gate_window": 24,
    "gate_min_history": 8,
    "seed": 2026,
}


@dataclass
class Experiment:
    name: str
    overrides: dict
    save_artifacts: bool = True


@dataclass
class ExperimentRecord:
    name: str
    output_dir: str
    status: str
    returncode: int | None
    started_at: float
    finished_at: float
    duration_sec: float
    command: list[str]


def paper_experiments() -> list[Experiment]:
    return [
        Experiment("main_seed_2026", {"seed": 2026}),
        Experiment("main_seed_2027", {"seed": 2027}),
        Experiment("main_seed_2028", {"seed": 2028}),
        Experiment("gate_window_8", {"gate_window": 8}),
        Experiment("gate_window_48", {"gate_window": 48}),
        Experiment("gate_min_history_4", {"gate_min_history": 4}),
        Experiment("gate_min_history_16", {"gate_min_history": 16}),
        Experiment("online_steps_1", {"online_steps": 1}),
        Experiment("online_steps_3", {"online_steps": 3}),
        Experiment("learning_rate_0p0005", {"learning_rate": 5e-4}),
        Experiment("learning_rate_0p002", {"learning_rate": 2e-3}),
        Experiment("hidden_dim_32", {"hidden_dim": 32}),
        Experiment("hidden_dim_128", {"hidden_dim": 128}),
        Experiment("no_target_clip", {"no_clip_target": True}),
    ]


def build_command(
    python_bin: str,
    project_root: Path,
    output_root: Path,
    device: str,
    exp: Experiment,
) -> tuple[list[str], Path]:
    config = BASE_ARGS | exp.overrides
    output_dir = output_root / exp.name
    cmd = [
        python_bin,
        str(project_root / "scripts" / "run_online_pv.py"),
        "--data_dir",
        str(config["data_dir"]),
        "--glob",
        str(config["glob"]),
        "--target_col",
        str(config["target_col"]),
        "--horizons",
        *[str(h) for h in config["horizons"]],
        "--lookback",
        str(config["lookback"]),
        "--offline_days",
        str(config["offline_days"]),
        "--year_days",
        str(config["year_days"]),
        "--online_steps",
        str(config["online_steps"]),
        "--hidden_dim",
        str(config["hidden_dim"]),
        "--learning_rate",
        str(config["learning_rate"]),
        "--replay_capacity",
        str(config["replay_capacity"]),
        "--gate_window",
        str(config["gate_window"]),
        "--gate_min_history",
        str(config["gate_min_history"]),
        "--seed",
        str(config["seed"]),
        "--device",
        device,
        "--output_dir",
        str(output_dir),
    ]
    if exp.save_artifacts:
        cmd.append("--save_artifacts")
    if config.get("no_clip_target"):
        cmd.append("--no_clip_target")
    return cmd, output_dir


def tee_run(cmd: list[str], cwd: Path, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return proc.wait()


def write_suite_index(output_root: Path, records: list[ExperimentRecord]) -> None:
    import pandas as pd

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "suite_records.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)
    pd.DataFrame([asdict(r) for r in records]).to_csv(output_root / "suite_records.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--project_root", type=Path, default=Path.cwd())
    parser.add_argument("--output_root", type=Path, default=Path("outputs/paper_suite"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    experiments = paper_experiments()
    if args.only:
        selected = set(args.only)
        experiments = [exp for exp in experiments if exp.name in selected]

    records: list[ExperimentRecord] = []
    manifest = {
        "project_root": str(project_root),
        "output_root": str(output_root),
        "device": args.device,
        "python": args.python,
        "experiments": [{"name": exp.name, "overrides": exp.overrides} for exp in experiments],
    }
    (output_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for exp in experiments:
        cmd, output_dir = build_command(args.python, project_root, output_root, args.device, exp)
        log_path = output_dir / "run.log"
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and (output_dir / "online_pv_summary.csv").exists():
            now = time.time()
            records.append(ExperimentRecord(exp.name, str(output_dir), "skipped", None, now, now, 0.0, cmd))
            write_suite_index(output_root, records)
            continue

        print(f"\\n=== Running {exp.name} ===", flush=True)
        print(" ".join(cmd), flush=True)
        started = time.time()
        if args.dry_run:
            returncode = 0
        else:
            returncode = tee_run(cmd, project_root, log_path)
            if returncode == 0:
                fig_cmd = [
                    args.python,
                    str(project_root / "scripts" / "make_online_pv_figures.py"),
                    str(output_dir),
                ]
                tee_run(fig_cmd, project_root, output_dir / "figure_generation.log")
        finished = time.time()
        status = "ok" if returncode == 0 else "failed"
        records.append(
            ExperimentRecord(
                exp.name,
                str(output_dir),
                status,
                returncode,
                started,
                finished,
                finished - started,
                cmd,
            )
        )
        write_suite_index(output_root, records)
        if returncode != 0:
            raise SystemExit(returncode)

    print(f"\\nSuite complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
