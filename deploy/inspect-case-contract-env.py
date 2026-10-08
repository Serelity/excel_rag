"""Read-only Conda discovery, runnable with the base environment's standard library."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ENVIRONMENT = "civic-rag-retrieval"
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


def compatible(probe, purpose):
    version = tuple(probe.get("python", [])[:2])
    packages = probe.get("packages", {})
    if not probe.get("pydantic_import") or packages.get("pydantic") != "2.11.4":
        return False
    if purpose == "client":
        return version in {(3, 11), (3, 12)}
    return version == (3, 11) and all(
        (packages.get(name) or "").split("+", 1)[0] == value
        for name, value in SERVE_VERSIONS.items()
    )


def choose(probes, purpose):
    candidates = [p for p in probes if p["name"] == ENVIRONMENT]
    if len(candidates) != 1:
        raise ValueError("required_environment_missing_or_ambiguous")
    if not compatible(candidates[0], purpose):
        raise ValueError(f"required_environment_incompatible_for_{purpose}")
    return candidates[0]


def inspect(conda, *, run=subprocess.run):
    inventory = run([conda, "env", "list", "--json"], capture_output=True, text=True,
                    check=True, timeout=60)
    prefixes = json.loads(inventory.stdout)["envs"]
    probes = []
    for prefix in prefixes:
        name = Path(prefix).name
        if name != ENVIRONMENT:
            continue
        item = {"name": name, "prefix": prefix}
        try:
            result = run([conda, "run", "--no-capture-output", "-p", prefix, "python", "-c", PROBE],
                         capture_output=True, text=True, check=True, timeout=60)
            lines = [line for line in result.stdout.splitlines()
                     if line.startswith("CASE_CONTRACT_PROBE=")]
            item.update(json.loads(lines[-1].split("=", 1)[1]))
        except (subprocess.SubprocessError, ValueError, IndexError):
            item["probe_error"] = "package_probe_failed"
        item["client_compatible"] = compatible(item, "client")
        item["serve_metadata_compatible"] = compatible(item, "serve")
        probes.append(item)
    report = {"status": "failed", "required_environment": ENVIRONMENT,
              "available_environments": prefixes, "probes": probes,
              "environment_changed": False, "gpu_inference": "not_run"}
    try:
        client = choose(probes, "client")
        server = choose(probes, "serve")
        report.update({"status": "ready_for_gpu_preflight", "client": client, "server": server})
    except ValueError as exc:
        report["reason"] = str(exc)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = inspect(args.conda)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    summary = {"status": report["status"], "report": str(args.output)}
    if "client" in report:
        summary.update({"client_environment": report["client"]["name"],
                        "serving_environment": report["server"]["name"]})
    else:
        summary["reason"] = report["reason"]
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if "client" in report else 2


if __name__ == "__main__":
    raise SystemExit(main())
