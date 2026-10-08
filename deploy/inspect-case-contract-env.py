"""Read-only Conda discovery, runnable with the base environment's standard library."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ENVIRONMENT = "civic-rag-extract-v1"
SERVE_VERSIONS = {
    "openai": "1.75.0", "pydantic": "2.11.4", "torch": "2.6.0",
    "transformers": "4.51.3", "vllm": "0.8.5",
}
PROBE = """
import importlib.metadata as m,json,sys
packages={}
for name in ('pydantic','openai','torch','transformers','vllm','pytest'):
    try: packages[name]=m.version(name)
    except m.PackageNotFoundError: packages[name]=None
try:
    import pydantic,pydantic_core
    imported=True
except Exception:
    imported=False
print('CASE_CONTRACT_PROBE='+json.dumps({'python':list(sys.version_info[:3]),
      'executable':sys.executable,'packages':packages,'pydantic_import':imported}))
"""


def compatibility_errors(probe, purpose):
    errors = []
    version = tuple(probe.get("python", [])[:2])
    packages = probe.get("packages", {})
    supported = {(3, 11), (3, 12)} if purpose == "client" else {(3, 11)}
    if version not in supported:
        errors.append({"component": "python", "actual": probe.get("python"),
                       "expected": "3.11 or 3.12" if purpose == "client" else "3.11"})
    if not probe.get("pydantic_import"):
        errors.append({"component": "pydantic_import", "actual": False, "expected": True})
    required = {"pydantic": "2.11.4"} if purpose == "client" else SERVE_VERSIONS
    for name, wanted in required.items():
        actual = packages.get(name)
        if (actual or "").split("+", 1)[0] != wanted:
            errors.append({"component": name, "actual": actual, "expected": wanted})
    return errors


def compatible(probe, purpose):
    return not compatibility_errors(probe, purpose)


def choose(probes, purpose, *, environment=ENVIRONMENT):
    candidates = [p for p in probes if p["name"] == environment]
    if len(candidates) != 1:
        raise ValueError("required_environment_missing_or_ambiguous")
    if not compatible(candidates[0], purpose):
        raise ValueError(f"required_environment_incompatible_for_{purpose}")
    return candidates[0]


def inspect(conda, *, environment=ENVIRONMENT, run=subprocess.run):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", environment):
        raise ValueError("invalid_environment_name")
    inventory = run([conda, "env", "list", "--json"], capture_output=True, text=True,
                    check=True, timeout=60)
    prefixes = json.loads(inventory.stdout)["envs"]
    probes = []
    for prefix in prefixes:
        name = Path(prefix).name
        if name != environment:
            continue
        item = {"name": name, "prefix": prefix}
        try:
            interpreter = str(Path(prefix) / "bin/python")
            result = run([conda, "run", "--no-capture-output", "-p", prefix,
                          interpreter, "-c", PROBE],
                         capture_output=True, text=True, check=True, timeout=60)
            lines = [line for line in result.stdout.splitlines()
                     if line.startswith("CASE_CONTRACT_PROBE=")]
            item.update(json.loads(lines[-1].split("=", 1)[1]))
        except (subprocess.SubprocessError, ValueError, IndexError):
            item["probe_error"] = "package_probe_failed"
        item["client_compatible"] = compatible(item, "client")
        item["serve_metadata_compatible"] = compatible(item, "serve")
        item["client_mismatches"] = compatibility_errors(item, "client")
        item["serve_mismatches"] = compatibility_errors(item, "serve")
        probes.append(item)
    report = {"status": "failed", "required_environment": environment,
              "available_environments": prefixes, "probes": probes,
              "environment_changed": False, "gpu_inference": "not_run"}
    try:
        client = choose(probes, "client", environment=environment)
        server = choose(probes, "serve", environment=environment)
        report.update({"status": "ready_for_gpu_preflight", "client": client, "server": server})
    except ValueError as exc:
        report["reason"] = str(exc)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda", required=True)
    parser.add_argument("--conda-env", default=ENVIRONMENT)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = inspect(args.conda, environment=args.conda_env)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    summary = {"status": report["status"], "report": str(args.output),
               "required_environment": report["required_environment"]}
    if "client" in report:
        summary.update({"client_environment": report["client"]["name"],
                        "serving_environment": report["server"]["name"]})
    else:
        summary["reason"] = report["reason"]
        summary["mismatches"] = [
            {"prefix": probe["prefix"], "items": probe["serve_mismatches"]}
            for probe in report["probes"]
        ]
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if "client" in report else 2


if __name__ == "__main__":
    raise SystemExit(main())
