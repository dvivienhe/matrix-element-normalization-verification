"""Regenerate every computed table and figure of the article and compare them with expected_outputs.json (tables
and figures byte by byte, JSON results after the run-identity fields are removed). Rebuilds results/; exit code 0
only if every program exits 0 and every compared output equals its expected value.

    python run_all.py"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EXPECTED = ROOT / "expected_outputs.json"
THREADS4 = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}

# (label, program and arguments, extra environment)
STEPS = [
    ("controlled validation", ["code/controlled_validation.py"], THREADS4),
    ("mixed-error set", ["code/mixed_error_methods.py"], {}),
    ("evidence rows", ["code/evidence_rows.py"], {}),
    ("potential convention", ["code/potential_convention.py"], {}),
    ("optical reference", ["code/reference/run_references.py", "--name", "reference"], {}),
    ("circular branches", ["code/reference/circular_branches.py", "--name", "circular_branches"], {}),
    ("Solcore interface analysis", ["code/solcore_interface/analyse_states.py"], {}),
    ("144-case analysis", ["code/case144_analysis.py"], {}),
    ("wall offset", ["code/wall_offset.py"], {}),
    ("text values", ["code/text_values.py"], {}),
    ("figures", ["code/make_figures.py"], {}),
]

FIGURES = [f"results/figures/Figure_{k}.{e}" for k in ("1", "2", "3", "4", "5", "6", "S1", "S2")
           for e in ("pdf", "eps", "png")]
TABLES = [
    "results/mixed_error_set/mixed_error_set_table.csv",
    "results/evidence_rows/evidence_rows_cells.csv",
    "results/potential_convention/potential_convention_readings.csv",
    "results/potential_convention/potential_convention_metrics.csv",
    "results/reference/transfer_grids.csv",
    "results/reference/weak_coupling.csv",
    "results/reference/optical_reference.csv",
    "results/reference/optical_branches.csv",
    "results/reference/zero_reference.csv",
    "results/circular_branches/circular_branches.csv",
    "results/solcore_interface/solcore_cells.csv",
    "results/case144/case144_summary_table.csv",
    "results/case144/case144_sweep_table.csv",
    "results/case144/case144_records_baseline.csv",
    "results/case144/case144_sweep_deltas.csv",
]
JSON_RESULTS = [
    "results/controlled/diagnostic_matrix.json",
    "results/mixed_error_set/mixed_error_set.json",
    "results/evidence_rows/evidence_rows.json",
    "results/potential_convention/potential_convention_results.json",
    "results/reference/valence_basis_checks.json",
    "results/reference/summary.json",
    "results/reference/continuum_reference.json",
    "results/reference/counting_and_occupation.json",
    "results/reference/interface_negative_controls.json",
    "results/circular_branches/circular_checks.json",
    "results/solcore_interface/solcore_results.json",
    "results/case144/case144_results.json",
    "results/wall_offset.json",
    "results/text_values.json",
    "results/derived_values.json",
]
# run-identity fields removed before a JSON result is compared
STRIP_KEYS = {"identity", "timestamp_utc", "wall_seconds", "wall_time_compute_s", "timings_s", "platform",
              "python", "numpy", "scipy", "matplotlib", "head", "dirty", "executable", "argv", "seconds"}
# text_values.json also records the SHA-256 of the result files it reads (they carry time stamps)
STRIP_KEYS_PER_FILE = {"results/text_values.json": {"inputs"}}


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip(o, keys):
    if isinstance(o, dict):
        return {k: _strip(v, keys) for k, v in o.items() if k not in keys}
    if isinstance(o, list):
        return [_strip(v, keys) for v in o]
    return o


def json_digest(path: Path, rel: str) -> str | None:
    if not path.exists():
        return None
    keys = STRIP_KEYS | STRIP_KEYS_PER_FILE.get(rel, set())
    obj = _strip(json.loads(path.read_text(encoding="utf-8")), keys)
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def digests(root: Path = ROOT) -> dict:
    """Current digests of every compared output under root."""
    return {"bytes": {rel: sha256_file(root / rel) for rel in TABLES + FIGURES},
            "json": {rel: json_digest(root / rel, rel) for rel in JSON_RESULTS}}


def main() -> None:
    if RESULTS.exists():
        shutil.rmtree(RESULTS)
    RESULTS.mkdir()
    py = sys.executable
    base_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    base_env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "steps": []}
    for label, argv, extra_env in STEPS:
        t0 = time.time()
        r = subprocess.run([py, *argv], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env={**base_env, **extra_env})
        step = {"label": label, "command": argv, "rc": r.returncode, "wall_s": round(time.time() - t0, 1),
                "stdout_tail": r.stdout[-1500:], "stderr_tail": r.stderr[-1500:]}
        report["steps"].append(step)
        print(f"  {label:28} rc {r.returncode}  {step['wall_s']:6.1f} s", flush=True)
    expected = json.loads(EXPECTED.read_text(encoding="utf-8")) if EXPECTED.exists() else {"bytes": {}, "json": {}}
    current = digests()
    comparisons = {}
    for kind in ("bytes", "json"):
        for rel, value in current[kind].items():
            exp = expected.get(kind, {}).get(rel)
            comparisons[rel] = {"kind": kind, "expected": exp, "observed": value,
                                "identical": exp is not None and exp == value}
    report["comparisons"] = comparisons
    ok_steps = all(s["rc"] == 0 for s in report["steps"])
    ok_cmp = all(c["identical"] for c in comparisons.values())
    report["all_ok"] = bool(ok_steps and ok_cmp)
    report["total_wall_s"] = round(sum(s["wall_s"] for s in report["steps"]), 1)
    (RESULTS / "run_all_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8",
                                                 newline="\n")
    bad = [rel for rel, c in comparisons.items() if not c["identical"]]
    print(f"steps {len(report['steps'])} (all exit 0: {ok_steps}); compared outputs {len(comparisons)}, "
          f"different from the expected value: {bad or 'none'}; total {report['total_wall_s']} s; "
          f"all_ok = {report['all_ok']}")
    sys.exit(0 if report["all_ok"] else 1)


if __name__ == "__main__":
    main()
