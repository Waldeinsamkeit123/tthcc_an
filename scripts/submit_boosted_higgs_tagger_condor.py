#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tthcc_an.config_loader import (
    expand_file_patterns as _expand_file_patterns,
    load_json_maybe_with_comments as _load_json_maybe_with_comments,
    resolve_output_dir as _resolve_output_dir,
)

DEFAULT_LCG_SETUP = "/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh"
DEFAULT_MERGE_REQUEST_MEMORY = "16 GB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and optionally submit an HTCondor workflow on lxplus for the "
            "boosted Higgs tagger study. The default workflow is chunked: many chunk "
            "jobs export slim NPZ payloads, then one merge job produces the final plots and tables."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the study config JSON used by run_boosted_higgs_tagger_study.py.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Final analysis output directory. Defaults to study.outdir from the config.",
    )
    parser.add_argument(
        "--workflow-mode",
        choices=["chunked", "single"],
        default="chunked",
        help="Submit either the recommended chunk+merge DAG workflow or a single batch job.",
    )
    parser.add_argument(
        "--files-per-chunk",
        type=int,
        default=20,
        help="How many ROOT files to include in each chunk job when using chunked mode.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional tag for the condor working directory. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--condor-dir",
        default=None,
        help="Optional directory where condor helper files are written.",
    )
    parser.add_argument(
        "--job-flavour",
        default="workday",
        help="HTCondor job flavour for chunk jobs, for example espresso, microcentury, longlunch, workday.",
    )
    parser.add_argument(
        "--request-memory",
        default="8 GB",
        help="Memory request for chunk jobs, or for the single job in single mode.",
    )
    parser.add_argument(
        "--request-cpus",
        type=int,
        default=1,
        help="CPU request for chunk jobs, or for the single job in single mode.",
    )
    parser.add_argument(
        "--request-disk",
        default="4 GB",
        help="Disk request for chunk jobs, or for the single job in single mode.",
    )
    parser.add_argument(
        "--merge-job-flavour",
        default=None,
        help="Optional override for the merge job flavour in chunked mode.",
    )
    parser.add_argument(
        "--merge-request-memory",
        default=DEFAULT_MERGE_REQUEST_MEMORY,
        help="Memory request for the merge job in chunked mode.",
    )
    parser.add_argument(
        "--merge-request-cpus",
        type=int,
        default=None,
        help="Optional override for merge-job CPU request in chunked mode.",
    )
    parser.add_argument(
        "--merge-request-disk",
        default=None,
        help="Optional override for merge-job disk request in chunked mode.",
    )
    parser.add_argument(
        "--lcg-setup",
        default=DEFAULT_LCG_SETUP,
        help="LCG setup script sourced inside batch jobs.",
    )
    parser.add_argument(
        "--python",
        default="python3",
        help="Python executable to run after sourcing the LCG view.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the generated HTCondor workflow immediately.",
    )
    parser.add_argument(
        "--submit-mode",
        choices=["auto", "share", "eossubmit"],
        default="auto",
        help=(
            "How to submit to HTCondor. 'auto' uses EosSubmit when the relevant "
            "paths are on /eos, otherwise it falls back to plain condor_submit/condor_submit_dag."
        ),
    )
    parser.add_argument(
        "analysis_args",
        nargs=argparse.REMAINDER,
        help=(
            "Extra arguments forwarded to run_boosted_higgs_tagger_study.py. "
            "Prefix forwarded options with '--', for example: "
            "-- --targets hcc hbb --scores gpart_h2cc gpart_h2bb"
        ),
    )
    return parser.parse_args()


def normalize_analysis_args(analysis_args: list[str]) -> list[str]:
    if analysis_args and analysis_args[0] == "--":
        return analysis_args[1:]
    return analysis_args


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def write_text(path: Path, content: str, executable: bool = False) -> None:
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def _strip_hash_comments(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        in_string = False
        escaped = False
        out_chars: list[str] = []
        for char in line:
            if escaped:
                out_chars.append(char)
                escaped = False
                continue
            if char == "\\":
                out_chars.append(char)
                escaped = True
                continue
            if char == '"':
                out_chars.append(char)
                in_string = not in_string
                continue
            if char == "#" and not in_string:
                break
            out_chars.append(char)
        cleaned_lines.append("".join(out_chars))
    return "\n".join(cleaned_lines)


def load_json_maybe_with_comments(path: Path) -> dict[str, Any]:
    return json.loads(_strip_hash_comments(path.read_text(encoding="utf-8")))


def expand_file_patterns(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            files.extend(matches)
            continue
        if any(token in pattern for token in ["*", "?", "["]):
            raise FileNotFoundError(f"No files matched pattern: {pattern}")
        if not Path(pattern).exists():
            raise FileNotFoundError(f"Input file does not exist: {pattern}")
        files.append(pattern)
    return files


def is_eos_path(path: Path) -> bool:
    return str(path.resolve()).startswith("/eos/")


def detect_submit_mode(requested_mode: str, paths: list[Path]) -> str:
    if requested_mode != "auto":
        return requested_mode
    if all(is_eos_path(path) for path in paths):
        return "eossubmit"
    return "share"


def get_submit_command(submit_mode: str, submit_target: Path, use_dag: bool) -> list[str]:
    submit_executable = "condor_submit_dag" if use_dag else "condor_submit"
    if submit_mode == "eossubmit":
        return [
            "bash",
            "-lc",
            f"module load lxbatch/eossubmit && {submit_executable} {shlex.quote(str(submit_target))}",
        ]
    return [submit_executable, str(submit_target)]


def chunked(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("--files-per-chunk must be positive.")
    return [items[index : index + size] for index in range(0, len(items), size)]


def load_expanded_samples(config_path: Path) -> dict[str, Any]:
    payload = _load_json_maybe_with_comments(config_path)
    samples = payload.get("samples", [])
    if not samples:
        raise ValueError(f"No samples found in configuration: {config_path}")

    expanded_samples: list[dict[str, Any]] = []
    for entry in samples:
        expanded_entry = dict(entry)
        expanded_entry["files"] = _expand_file_patterns(list(entry["files"]))
        if not expanded_entry["files"]:
            raise ValueError(f"No files found for sample '{entry.get('name', 'unknown')}'.")
        expanded_samples.append(expanded_entry)

    return {
        "normalization": payload.get("normalization", {}),
        "samples": expanded_samples,
    }


def build_chunk_manifests(
    config_path: Path,
    chunk_config_dir: Path,
    chunk_output_dir: Path,
    files_per_chunk: int,
) -> list[dict[str, str]]:
    config_payload = load_expanded_samples(config_path)
    normalization = config_payload.get("normalization", {})
    jobs: list[dict[str, str]] = []
    chunk_index = 0

    for sample in config_payload["samples"]:
        file_groups = chunked(sample["files"], files_per_chunk)
        for file_group in file_groups:
            manifest_path = chunk_config_dir / f"chunk_{chunk_index:04d}.json"
            chunk_output_path = chunk_output_dir / f"chunk_{chunk_index:04d}.npz"
            manifest_payload = {
                "normalization": normalization,
                "samples": [{**sample, "files": file_group}],
            }
            write_text(manifest_path, json.dumps(manifest_payload, indent=2))
            jobs.append(
                {
                    "chunk_config": str(manifest_path.resolve()),
                    "chunk_output": str(chunk_output_path.resolve()),
                }
            )
            chunk_index += 1

    if not jobs:
        raise ValueError("No chunk jobs were produced from the input configuration.")
    return jobs


def build_single_wrapper_script(
    wrapper_path: Path,
    config_path: Path,
    outdir: Path,
    lcg_setup: str,
    python_executable: str,
    analysis_args: list[str],
) -> None:
    command = [
        python_executable,
        "scripts/run_boosted_higgs_tagger_study.py",
        "--config",
        str(config_path),
        "--outdir",
        str(outdir),
        *analysis_args,
    ]
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -eo pipefail",
            "set +u",
            f"source {shlex.quote(lcg_setup)}",
            "set -u",
            f"cd {shlex.quote(str(REPO_ROOT))}",
            f"mkdir -p {shlex.quote(str(outdir))}",
            shell_join(command),
            "",
        ]
    )
    write_text(wrapper_path, script, executable=True)


def build_chunk_wrapper_script(
    wrapper_path: Path,
    outdir: Path,
    chunk_output_dir: Path,
    lcg_setup: str,
    python_executable: str,
    analysis_args: list[str],
) -> None:
    command = [
        python_executable,
        "scripts/run_boosted_higgs_tagger_study.py",
        "--outdir",
        str(outdir),
        *analysis_args,
    ]
    command_prefix = shell_join(command)
    script_lines = [
        "#!/usr/bin/env bash",
        "set -eo pipefail",
        'chunk_config="$1"',
        'chunk_output="$2"',
        "set +u",
        f"source {shlex.quote(lcg_setup)}",
        "set -u",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"mkdir -p {shlex.quote(str(outdir))}",
        f"mkdir -p {shlex.quote(str(chunk_output_dir))}",
        f'{command_prefix} --config "$chunk_config" --export-chunk "$chunk_output"',
        "",
    ]
    write_text(wrapper_path, "\n".join(script_lines), executable=True)


def build_merge_wrapper_script(
    wrapper_path: Path,
    config_path: Path,
    outdir: Path,
    chunk_glob_pattern: str,
    lcg_setup: str,
    python_executable: str,
    analysis_args: list[str],
) -> None:
    command = [
        python_executable,
        "scripts/run_boosted_higgs_tagger_study.py",
        "--config",
        str(config_path),
        "--outdir",
        str(outdir),
        "--merge-chunks",
        chunk_glob_pattern,
        *analysis_args,
    ]
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -eo pipefail",
            "set +u",
            f"source {shlex.quote(lcg_setup)}",
            "set -u",
            f"cd {shlex.quote(str(REPO_ROOT))}",
            f"mkdir -p {shlex.quote(str(outdir))}",
            shell_join(command),
            "",
        ]
    )
    write_text(wrapper_path, script, executable=True)


def build_submit_description(
    submit_path: Path,
    wrapper_path: Path,
    log_dir: Path,
    request_cpus: int,
    request_memory: str,
    request_disk: str,
    job_flavour: str,
) -> None:
    submit_text = "\n".join(
        [
            "universe = vanilla",
            f"executable = {wrapper_path}",
            f"output = {log_dir / 'job.$(ClusterId).$(ProcId).out'}",
            f"error = {log_dir / 'job.$(ClusterId).$(ProcId).err'}",
            f"log = {log_dir / 'job.$(ClusterId).log'}",
            "should_transfer_files = NO",
            "getenv = True",
            'MY.WantOS = "el9"',
            f'+JobFlavour = "{job_flavour}"',
            f"request_cpus = {request_cpus}",
            f"request_memory = {request_memory}",
            f"request_disk = {request_disk}",
            "",
            "queue 1",
            "",
        ]
    )
    write_text(submit_path, submit_text)


def build_chunk_submit_description(
    submit_path: Path,
    wrapper_path: Path,
    log_dir: Path,
    request_cpus: int,
    request_memory: str,
    request_disk: str,
    job_flavour: str,
    chunk_jobs: list[dict[str, str]],
) -> None:
    queue_lines = ["queue chunk_config, chunk_output from ("]
    queue_lines.extend(f"{job['chunk_config']} {job['chunk_output']}" for job in chunk_jobs)
    queue_lines.append(")")

    submit_text = "\n".join(
        [
            "universe = vanilla",
            f"executable = {wrapper_path}",
            "arguments = $(chunk_config) $(chunk_output)",
            f"output = {log_dir / 'chunk.$(ClusterId).$(ProcId).out'}",
            f"error = {log_dir / 'chunk.$(ClusterId).$(ProcId).err'}",
            f"log = {log_dir / 'chunk.$(ClusterId).log'}",
            "should_transfer_files = NO",
            "getenv = True",
            'MY.WantOS = "el9"',
            f'+JobFlavour = "{job_flavour}"',
            f"request_cpus = {request_cpus}",
            f"request_memory = {request_memory}",
            f"request_disk = {request_disk}",
            "",
            *queue_lines,
            "",
        ]
    )
    write_text(submit_path, submit_text)


def build_workflow_dag(dag_path: Path, chunk_submit_path: Path, merge_submit_path: Path) -> None:
    dag_text = "\n".join(
        [
            f"JOB CHUNKS {chunk_submit_path}",
            f"JOB MERGE {merge_submit_path}",
            "PARENT CHUNKS CHILD MERGE",
            "",
        ]
    )
    write_text(dag_path, dag_text)


def build_metadata(metadata_path: Path, payload: dict[str, Any]) -> None:
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_single_job(
    args: argparse.Namespace,
    config_path: Path,
    outdir: Path,
    condor_dir: Path,
    analysis_args: list[str],
) -> tuple[list[Path], Path, dict[str, Any]]:
    log_dir = condor_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    wrapper_path = condor_dir / "run_job.sh"
    submit_path = condor_dir / "submit.sub"

    build_single_wrapper_script(
        wrapper_path=wrapper_path,
        config_path=config_path,
        outdir=outdir,
        lcg_setup=args.lcg_setup,
        python_executable=args.python,
        analysis_args=analysis_args,
    )
    build_submit_description(
        submit_path=submit_path,
        wrapper_path=wrapper_path,
        log_dir=log_dir,
        request_cpus=args.request_cpus,
        request_memory=args.request_memory,
        request_disk=args.request_disk,
        job_flavour=args.job_flavour,
    )

    metadata = {
        "workflow_mode": "single",
        "config": str(config_path),
        "outdir": str(outdir),
        "wrapper": str(wrapper_path),
        "submit_file": str(submit_path),
        "analysis_args": analysis_args,
        "lcg_setup": args.lcg_setup,
        "python": args.python,
        "job_flavour": args.job_flavour,
        "request_cpus": args.request_cpus,
        "request_memory": args.request_memory,
        "request_disk": args.request_disk,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    return [wrapper_path, submit_path, log_dir], submit_path, metadata


def prepare_chunked_workflow(
    args: argparse.Namespace,
    config_path: Path,
    outdir: Path,
    condor_dir: Path,
    analysis_args: list[str],
) -> tuple[list[Path], Path, dict[str, Any]]:
    chunk_config_dir = condor_dir / "chunk_configs"
    chunk_output_dir = condor_dir / "chunk_outputs"
    chunk_log_dir = condor_dir / "logs" / "chunks"
    merge_log_dir = condor_dir / "logs" / "merge"
    chunk_config_dir.mkdir(parents=True, exist_ok=True)
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    chunk_log_dir.mkdir(parents=True, exist_ok=True)
    merge_log_dir.mkdir(parents=True, exist_ok=True)

    chunk_jobs = build_chunk_manifests(
        config_path=config_path,
        chunk_config_dir=chunk_config_dir,
        chunk_output_dir=chunk_output_dir,
        files_per_chunk=args.files_per_chunk,
    )

    chunk_wrapper_path = condor_dir / "run_chunk.sh"
    chunk_submit_path = condor_dir / "chunks.sub"
    merge_wrapper_path = condor_dir / "run_merge.sh"
    merge_submit_path = condor_dir / "merge.sub"
    dag_path = condor_dir / "workflow.dag"

    build_chunk_wrapper_script(
        wrapper_path=chunk_wrapper_path,
        outdir=outdir,
        chunk_output_dir=chunk_output_dir,
        lcg_setup=args.lcg_setup,
        python_executable=args.python,
        analysis_args=analysis_args,
    )
    build_merge_wrapper_script(
        wrapper_path=merge_wrapper_path,
        config_path=config_path,
        outdir=outdir,
        chunk_glob_pattern=str((chunk_output_dir / "*.npz").resolve()),
        lcg_setup=args.lcg_setup,
        python_executable=args.python,
        analysis_args=analysis_args,
    )
    build_chunk_submit_description(
        submit_path=chunk_submit_path,
        wrapper_path=chunk_wrapper_path,
        log_dir=chunk_log_dir,
        request_cpus=args.request_cpus,
        request_memory=args.request_memory,
        request_disk=args.request_disk,
        job_flavour=args.job_flavour,
        chunk_jobs=chunk_jobs,
    )
    build_submit_description(
        submit_path=merge_submit_path,
        wrapper_path=merge_wrapper_path,
        log_dir=merge_log_dir,
        request_cpus=args.merge_request_cpus or args.request_cpus,
        request_memory=args.merge_request_memory,
        request_disk=args.merge_request_disk or args.request_disk,
        job_flavour=args.merge_job_flavour or args.job_flavour,
    )
    build_workflow_dag(
        dag_path=dag_path,
        chunk_submit_path=chunk_submit_path,
        merge_submit_path=merge_submit_path,
    )

    metadata = {
        "workflow_mode": "chunked",
        "config": str(config_path),
        "outdir": str(outdir),
        "chunk_wrapper": str(chunk_wrapper_path),
        "chunk_submit_file": str(chunk_submit_path),
        "merge_wrapper": str(merge_wrapper_path),
        "merge_submit_file": str(merge_submit_path),
        "dag_file": str(dag_path),
        "chunk_output_dir": str(chunk_output_dir),
        "analysis_args": analysis_args,
        "lcg_setup": args.lcg_setup,
        "python": args.python,
        "files_per_chunk": args.files_per_chunk,
        "n_chunk_jobs": len(chunk_jobs),
        "chunk_request_cpus": args.request_cpus,
        "chunk_request_memory": args.request_memory,
        "chunk_request_disk": args.request_disk,
        "chunk_job_flavour": args.job_flavour,
        "merge_request_cpus": args.merge_request_cpus or args.request_cpus,
        "merge_request_memory": args.merge_request_memory,
        "merge_request_disk": args.merge_request_disk or args.request_disk,
        "merge_job_flavour": args.merge_job_flavour or args.job_flavour,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    relevant_paths = [
        chunk_wrapper_path,
        chunk_submit_path,
        merge_wrapper_path,
        merge_submit_path,
        dag_path,
        chunk_output_dir,
        chunk_log_dir,
        merge_log_dir,
    ]
    return relevant_paths, dag_path, metadata


def main() -> None:
    args = parse_args()
    analysis_args = normalize_analysis_args(args.analysis_args)

    tag = args.tag or datetime.now().strftime("boosted_higgs_tagger_%Y%m%d_%H%M%S")
    condor_dir = Path(args.condor_dir) if args.condor_dir else REPO_ROOT / "condor" / tag
    condor_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config).resolve()
    outdir = Path(_resolve_output_dir(config_path, args.outdir, REPO_ROOT)).resolve()
    metadata_path = condor_dir / "job_metadata.json"

    if args.workflow_mode == "single":
        generated_paths, submit_target, metadata = prepare_single_job(
            args=args,
            config_path=config_path,
            outdir=outdir,
            condor_dir=condor_dir,
            analysis_args=analysis_args,
        )
        use_dag = False
    else:
        generated_paths, submit_target, metadata = prepare_chunked_workflow(
            args=args,
            config_path=config_path,
            outdir=outdir,
            condor_dir=condor_dir,
            analysis_args=analysis_args,
        )
        use_dag = True

    submit_mode = detect_submit_mode(
        requested_mode=args.submit_mode,
        paths=[REPO_ROOT, config_path, outdir, condor_dir, submit_target, *generated_paths],
    )
    metadata["submit_mode"] = submit_mode
    build_metadata(metadata_path, metadata)

    submit_command = get_submit_command(submit_mode, submit_target, use_dag=use_dag)

    print(f"Prepared workflow directory: {condor_dir}")
    print(f"Workflow mode: {args.workflow_mode}")
    print(f"Submit mode: {submit_mode}")
    if args.workflow_mode == "chunked":
        print(f"Chunk jobs: {metadata['n_chunk_jobs']}")
        print(f"Chunk outputs: {metadata['chunk_output_dir']}")
        print(f"Chunk submit file: {metadata['chunk_submit_file']}")
        print(f"Merge submit file: {metadata['merge_submit_file']}")
        print(f"DAG file: {metadata['dag_file']}")
        print(f"Logs: {condor_dir / 'logs'}")
    else:
        print(f"Submit file: {metadata['submit_file']}")
        print(f"Logs: {condor_dir / 'logs'}")
    print("")
    print("Submit with:")
    print(f"  {shell_join(submit_command)}")

    if args.submit:
        subprocess.run(submit_command, check=True)


if __name__ == "__main__":
    main()
