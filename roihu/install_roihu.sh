#!/usr/bin/env bash
#
# CLEAN FROM SCRATCH: Roihu OpenEuroLLM dclm-core-22 + lm-eval 0.4.13 (official compatibility fixes).
#
# This script prepares the host environment, configures the persistent scheduler, generates the evaluation run,
# installs the container package overlay, applies upstream compatibility fixes, and pre-caches datasets.
# It deliberately DEFERS SLURM submission until all preparation is complete.
# The installed upstream template is configured so subsequent
# ordinary oellm-eval schedule commands work on Roihu as well.
#
# Usage:
#   ./install_roihu.sh --project project_2021911 --hf-token-file ~/hf_token
#
# Fully self-contained fresh installation into workspace and cache.
# Assumes the installation directories are initially absent, as requested.
#
set -Eeuo pipefail

# ----------------------------- Configuration -------------------------------
usage() {
  cat <<'HELP'
Usage:
  ./install_roihu.sh --project project_2021911 [options]

Required:
  --project NAME          Project name, e.g. project_2021911.
                          Scratch root and SLURM account are both derived from NAME.
Options:
  --partition NAME        Default: gpumedium
  --hf-token-file PATH    Token file (default: ~/hf_token if present)
  --no-hf-token           Public Hugging Face access only
  --submit                Submit the GPU array after successful validation
  --queue-limit N         Scheduler queue limit (default: 128)
  --model-id ID           Hugging Face model ID (default: openeurollm/prelude)
  --model-revision REV    Model revision (default: pinned prelude snapshot)
  --walltime HH:MM:SS     Per-job limit (default: 24:00:00)
  --no-samples            Disable sample logging
  --job-online            Allow compute jobs to contact Hugging Face
  --help

Installs from scratch into /scratch/NAME/workspace and
/scratch/NAME/cache. Run it in a clean installation directory.
No reset, backup, migration, or preservation logic is included.
HELP
}
PROJECT=''
SLURM_PARTITION=gpumedium
QUEUE_LIMIT=128
GPU_TYPE=gh200
GPUS_PER_NODE=1
SUBMIT=0
JOB_HF_OFFLINE=1
HF_TOKEN_FILE="$HOME/hf_token"
NO_HF_TOKEN=0
MODEL_ID='openeurollm/prelude'
MODEL_REVISION='cfd208a204dad51e1959d81d7a57e1d21b860ce2'
WALLTIME='24:00:00'
LOG_SAMPLES=1
TASK_GROUP='dclm-core-22'
REPO_URL='https://github.com/OpenEuroLLM/oellm-eval.git'
REPO_COMMIT='f52e8fd2e7923eccab8204ae7dac327d6ba10ce8'
SIF_NAME='eval_env-jupiter.sif'
while (( $# )); do
  case "$1" in
    --project|--partition|--hf-token-file|--queue-limit|--model-id|--model-revision|--walltime)
      (( $# >= 2 )) || { echo "Missing argument to $1" >&2; exit 2; }
      case "$1" in
        --project) PROJECT="$2";;
        --partition) SLURM_PARTITION="$2";;
        --hf-token-file) HF_TOKEN_FILE="$2";;
        --queue-limit) QUEUE_LIMIT="$2";;
        --model-id) MODEL_ID="$2";;
        --model-revision) MODEL_REVISION="$2";;
        --walltime) WALLTIME="$2";;
      esac
      shift 2;;
    --no-hf-token) NO_HF_TOKEN=1; shift;;
    --no-samples) LOG_SAMPLES=0; shift;;
    --submit) SUBMIT=1; shift;;
    --job-online) JOB_HF_OFFLINE=0; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done
[[ "$PROJECT" =~ ^project_[0-9]+$ ]] || { usage; echo 'Expected --project project_XXXX' >&2; exit 2; }
PROJECT_ROOT="/scratch/$PROJECT"
SLURM_ACCOUNT="$PROJECT"
[[ -d "$PROJECT_ROOT" ]] || { echo "Project scratch directory does not exist: $PROJECT_ROOT" >&2; exit 2; }
[[ "$SLURM_PARTITION" =~ ^[a-zA-Z0-9_-]+$ && "$QUEUE_LIMIT" =~ ^[1-9][0-9]*$ && "$WALLTIME" =~ ^[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]$ ]] || { echo 'Invalid partition, queue limit, or walltime' >&2; exit 2; }
WORKSPACE="$PROJECT_ROOT/workspace"
CACHE_ROOT="$PROJECT_ROOT/cache"
REPO_DIR="$WORKSPACE/oellm-eval"
CONTAINER_PKGS="$WORKSPACE/container_packages"
HF_HOME="$CACHE_ROOT/huggingface"
HF_HUB_CACHE="$HF_HOME/hub"
# The host scheduler uses datasets 5.x, whereas the container uses datasets 2.19.2.
# Never let the two versions write to the same builder cache.
HOST_HF_DATASETS_CACHE="$HF_HOME/datasets_host_scheduler"
RUNTIME_HF_DATASETS_CACHE="$HF_HOME/datasets_container_runtime"
HF_DATASETS_CACHE="$HOST_HF_DATASETS_CACHE"
UV_CACHE_DIR="$CACHE_ROOT/uv"
UV_PYTHON_INSTALL_DIR="$CACHE_ROOT/uv/python"
UV_TOOL_DIR="$CACHE_ROOT/uv/tools"
UV_TOOL_BIN_DIR="$WORKSPACE/bin"
MODEL_DIR=""
CUSTOM_TASKS="$REPO_DIR/oellm/resources/custom_lm_eval_tasks"
MANIFEST_DIR="$WORKSPACE/reproducibility"
LOG_DIR="$MANIFEST_DIR/logs"
SIF_IMAGE="$WORKSPACE/$SIF_NAME"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"; }

on_error() {
    rc=$?
    printf '\nERROR: setup failed at line %s (exit %s).\n' "${BASH_LINENO[0]:-?}" "$rc" >&2
    printf 'See logs under: %s\n' "$LOG_DIR" >&2
    exit "$rc"
}
trap on_error ERR

# Fresh installation only: no resetting, backups, or migration.
# A non-empty installation directory signals that this was not a clean start.
for target in "$WORKSPACE" "$CACHE_ROOT"; do
    if [[ -e "$target" ]]; then
        if [[ ! -d "$target" ]] || [[ -n "$(ls -A "$target")" ]]; then
            die "Expected an empty fresh installation directory: $target"
        fi
    fi
done
mkdir -p "$WORKSPACE" "$CACHE_ROOT" "$MANIFEST_DIR" "$LOG_DIR" "$UV_TOOL_BIN_DIR" \
         "$HF_HUB_CACHE" "$HOST_HF_DATASETS_CACHE" "$RUNTIME_HF_DATASETS_CACHE"
exec > >(tee -a "$LOG_DIR/setup_$(date +%Y%m%d_%H%M%S).log") 2>&1

log "Checking Roihu tools and architecture"
need git
need curl
need python3
need apptainer
need sbatch
[[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]] ||
    die "Expected aarch64 Roihu login node; detected $(uname -m)."
apptainer --version
python3 --version

export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_TOOL_DIR UV_TOOL_BIN_DIR
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE HOST_HF_DATASETS_CACHE RUNTIME_HF_DATASETS_CACHE
# Read the token contents from ~/hf_token (never treat the path as the token).
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/hf_token}"
if [[ "$NO_HF_TOKEN" == 1 ]]; then
    unset HF_TOKEN || true
    log 'Hugging Face authentication disabled by --no-hf-token'
elif [[ -f "$HF_TOKEN_FILE" ]]; then
    export HF_TOKEN
    HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
    [[ -n "$HF_TOKEN" ]] || die "HF token file is empty: $HF_TOKEN_FILE"
    log "Using Hugging Face credentials from $HF_TOKEN_FILE (token not printed)"
else
    if [[ -n "${HF_TOKEN:-}" ]]; then
        log 'Using token from existing HF_TOKEN environment variable'
    else
        log 'No token file found; public Hugging Face access only'
    fi
fi
export PATH="$UV_TOOL_BIN_DIR:$PATH"
export EVAL_BASE_DIR="$WORKSPACE"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

# ----------------------------- uv and source -------------------------------
log "Installing uv if needed"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    [[ -x "$HOME/.local/bin/uv" ]] && export PATH="$HOME/.local/bin:$PATH"
fi
need uv
uv --version | tee "$MANIFEST_DIR/uv-version.txt"

log "Cloning the pinned oellm-eval revision"
git clone "$REPO_URL" "$REPO_DIR"
git -C "$REPO_DIR" checkout --detach "$REPO_COMMIT"
git -C "$REPO_DIR" rev-parse HEAD | tee "$MANIFEST_DIR/oellm-eval-commit.txt"

log "Creating Python 3.12 environment and installing oellm-eval"
cd "$REPO_DIR"
if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.12 .venv; fi
source .venv/bin/activate
uv pip install -e .
oellm-eval --help >/dev/null
# Keep the venv active for model download and schedule generation.

# ----------------------------- Cluster config ------------------------------
log "Writing the Roihu cluster configuration"
python3 - "$REPO_DIR/oellm/resources/clusters.yaml" "$SIF_NAME" \
    "$SLURM_PARTITION" "$SLURM_ACCOUNT" "$QUEUE_LIMIT" "$WORKSPACE" "$HF_HOME" <<'PY'
from pathlib import Path
import re, sys

path = Path(sys.argv[1])
sif, partition, account, queue_limit, workspace, hf_home = sys.argv[2:]
text = path.read_text()
lines = text.splitlines()
out = []
i = 0
replaced = False
while i < len(lines):
    if re.match(r"^roihu\s*:", lines[i]):
        if not replaced:
            out.extend([
                "roihu: # Roihu GH200",
                '  hostname_pattern: "roihu*"',
                f'  EVAL_BASE_DIR: "{workspace}"',
                f'  HF_HOME: "{hf_home}"',
                f'  PARTITION: "{partition}"',
                f'  ACCOUNT: "{account}"',
                f"  QUEUE_LIMIT: {queue_limit}",
                f'  EVAL_CONTAINER_IMAGE: "{sif}"',
                '  SINGULARITY_ARGS: "--nv"',
            ])
            replaced = True
        i += 1
        while i < len(lines) and (not lines[i].strip() or lines[i][0].isspace() or lines[i].lstrip().startswith("#")):
            i += 1
        continue
    out.append(lines[i])
    i += 1
if not replaced:
    if out and out[-1].strip():
        out.append("")
    out.extend([
        "roihu: # Roihu GH200",
        '  hostname_pattern: "roihu*"',
        f'  EVAL_BASE_DIR: "{workspace}"',
        f'  HF_HOME: "{hf_home}"',
        f'  PARTITION: "{partition}"',
        f'  ACCOUNT: "{account}"',
        f"  QUEUE_LIMIT: {queue_limit}",
        f'  EVAL_CONTAINER_IMAGE: "{sif}"',
        '  SINGULARITY_ARGS: "--nv"',
    ])
path.write_text("\n".join(out) + "\n")
print(path.read_text())
PY

# Roihu scheduler integration: change the SOURCE template, never
# rely on hand-patching a generated job. oellm-eval schedule uses this template
# every time, including future evaluations of other supported models/tasks.
log "Configuring the persistent oellm-eval SLURM template"
python3 - "$REPO_DIR/oellm/resources/template.sbatch" "$GPU_TYPE" "$CONTAINER_PKGS" "$HF_HOME" "$RUNTIME_HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_TOKEN_FILE" "$NO_HF_TOKEN" <<'PY_TEMPLATE'
from pathlib import Path
import sys
p = Path(sys.argv[1]); gpu, packages, hf_home, runtime_cache, hub_cache, token_file, no_token = sys.argv[2:]
s = p.read_text()

def unique_replace(old, new, description):
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"Upstream {description} changed: expected one occurrence, got {count}; refusing unsafe patch")
    s = s.replace(old, new, 1)

unique_replace('#SBATCH --gres=gpu:$GPUS_PER_NODE',
               f'#SBATCH --gres=gpu:{gpu}:$GPUS_PER_NODE', 'GPU GRES')
unique_replace('export HF_DATASETS_CACHE="$HF_HOME/datasets"',
               f'export HF_DATASETS_CACHE="{runtime_cache}"', 'runtime dataset cache')
unique_replace('BIND_PATHS="$EVAL_BASE_DIR:$EVAL_BASE_DIR,',
               f'BIND_PATHS="{packages}:/packages,$EVAL_BASE_DIR:$EVAL_BASE_DIR,',
               'initial BIND_PATHS')
# Insert only after the SINGLE CSV assignment. Use actual newline characters;
# do not introduce braces that would conflict with upstream Python .format().
marker = 'CSV_PATH="{csv_path}"'
exports = ''.join([
    f'export HF_HOME="{hf_home}"\n',
    # Read optional auth at job runtime, never interpolate credentials into job scripts.
    (f'if [ -r "{token_file}" ]; then export APPTAINERENV_HF_TOKEN="$(head -n 1 "{token_file}" | tr -d "\\r\\n")"; export SINGULARITYENV_HF_TOKEN="$APPTAINERENV_HF_TOKEN"; fi\n' if no_token == '0' else ''),
    f'export HF_HUB_CACHE="{hub_cache}"\n',
    'export APPTAINERENV_PYTHONPATH=/packages\n',
    'export SINGULARITYENV_PYTHONPATH=/packages\n',
    f'export APPTAINERENV_HF_HOME="{hf_home}"\n',
    f'export SINGULARITYENV_HF_HOME="{hf_home}"\n',
    f'export APPTAINERENV_HF_DATASETS_CACHE="{runtime_cache}"\n',
    f'export SINGULARITYENV_HF_DATASETS_CACHE="{runtime_cache}"\n',
    f'export APPTAINERENV_HF_HUB_CACHE="{hub_cache}"\n',
    f'export SINGULARITYENV_HF_HUB_CACHE="{hub_cache}"\n',
    '[ -s "$CSV_PATH" ] || exit 2',
])
unique_replace(marker, marker + '\n' + exports, 'CSV_PATH')
if r'\nexport' in s:
    raise SystemExit('Found literal backslash-n export sequence')
p.write_text(s)
print('Persistent GH200 GPU GRES, overlay, and container dataset cache configured:', p)
PY_TEMPLATE

# ----------------------------- Container download --------------------------
# Download the complete official image from Hugging Face on an empty workspace.
# This must NOT depend on a previously installed SIF or on schedule side effects.
log "Downloading evaluation container from Hugging Face (~13 GB)"
python3 - "$WORKSPACE" <<'PY_CONTAINER_DOWNLOAD'
import sys
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="openeurollm/evaluation_singularity_images",
    repo_type="dataset",
    filename="eval_env-jupiter.sif",
    revision="main",
    local_dir=sys.argv[1],
)
print("Downloaded container:", path)
PY_CONTAINER_DOWNLOAD
[[ -s "$SIF_IMAGE" ]] || die "Container download did not create $SIF_IMAGE"
apptainer inspect "$SIF_IMAGE" >/dev/null
sha256sum "$SIF_IMAGE" | tee "$MANIFEST_DIR/container.sha256"

# ----------------------------- Model snapshot ------------------------------
log "Downloading pinned Hugging Face model snapshot: $MODEL_ID ($MODEL_REVISION)"
export MODEL_ID MODEL_REVISION
MODEL_DIR="$("$REPO_DIR/.venv/bin/python" - <<'PY_MODEL'
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id=os.environ['MODEL_ID'], revision=os.environ['MODEL_REVISION'],
    cache_dir=os.environ['HF_HUB_CACHE'],
)
print(path)
PY_MODEL
)"
[[ -d "$MODEL_DIR" && -f "$MODEL_DIR/config.json" ]] || die "Downloaded model snapshot invalid: $MODEL_DIR"
MODEL_RESOLVED_REVISION="$(basename "$MODEL_DIR")"
log "Model revision resolved: $MODEL_RESOLVED_REVISION"
printf '%s\n' "$MODEL_ID" > "$MANIFEST_DIR/model-id.txt"
printf '%s\n' "$MODEL_RESOLVED_REVISION" > "$MANIFEST_DIR/model-revision.txt"

# ----------------------------- Schedule generation -------------------------
# oellm-eval schedule submits internally. Intercept that one sbatch call so
# the generated array cannot start before the overlay and dataset cache exist.
log "Generating jobs while deferring the actual SLURM submission"
DEFER_BIN="$WORKSPACE/.deferred_sbatch"
mkdir -p "$DEFER_BIN"
REAL_PATH="$PATH"
cat > "$DEFER_BIN/sbatch" <<'SH'
#!/usr/bin/env bash
set -e
target="${@: -1}"
if [[ ! -f "$target" ]]; then
    echo "Deferred sbatch wrapper: expected a generated .sbatch file, got: $target" >&2
    exit 2
fi
echo "Submitted batch job 0 (submission deferred by setup script)"
exit 0
SH
chmod +x "$DEFER_BIN/sbatch"

cd "$REPO_DIR"
export EVAL_CONTAINER_IMAGE="$SIF_NAME"
export PATH="$DEFER_BIN:$REAL_PATH"
oellm-eval schedule \
    --models "$MODEL_DIR" \
    --task_groups "$TASK_GROUP" \
    --tasks_per_job 1 \
    --log_samples "$( ((LOG_SAMPLES)) && echo true || echo false )" \
    --slurm_template_var "{\"PARTITION\":\"$SLURM_PARTITION\",\"ACCOUNT\":\"$SLURM_ACCOUNT\",\"GPUS_PER_NODE\":$GPUS_PER_NODE}"
export PATH="$REAL_PATH"
rm -rf "$DEFER_BIN"

# Locate the newly generated run directory; the cluster config writes to
# $EVAL_BASE_DIR/$USER/<timestamp>, not to a separate outputs/ directory.
JOBS_CSV="$(python3 - "$WORKSPACE" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
candidates = [
    p for p in root.rglob("jobs.csv")
    if p.is_file() and (p.parent / "submit_evals.sbatch").is_file()
    and "reproducibility" not in p.parts
]
if not candidates:
    raise SystemExit("No jobs.csv with a sibling submit_evals.sbatch found.")
p = max(candidates, key=lambda x: x.stat().st_mtime)
print(p)
PY
)"
JOB_DIR="$(dirname "$JOBS_CSV")"
SBATCH_FILE="$JOB_DIR/submit_evals.sbatch"
# Scheduler is authenticated; scan generated artifacts to ensure it did not
# embed the secret in the SLURM script or job CSV. Jobs read the token file at runtime.
if [[ -n "${HF_TOKEN:-}" ]]; then
  python3 - "$SBATCH_FILE" "$JOBS_CSV" <<'PY_SECRET'
from pathlib import Path
import os,sys
secret=os.environ['HF_TOKEN']
for filename in sys.argv[1:]:
    if secret in Path(filename).read_text():
        raise SystemExit(f'Refusing generated artifact containing a credential: {filename}')
print('PASS: no HF_TOKEN value embedded in generated artifacts')
PY_SECRET
fi
[[ -s "$JOBS_CSV" ]] || die "Generated jobs.csv is empty: $JOBS_CSV"
[[ -s "$SBATCH_FILE" ]] || die "Generated SLURM file missing: $SBATCH_FILE"

log "Verifying that the persistent source template generated a valid Roihu SLURM script"
python3 - "$JOBS_CSV" "$SBATCH_FILE" "$GPU_TYPE" "$GPUS_PER_NODE" "$CONTAINER_PKGS" "$RUNTIME_HF_DATASETS_CACHE" <<'PY_VALIDATE'
from pathlib import Path
import csv,sys,re
csv_path, job_path, gpu_type, gpu_count, packages, cache = sys.argv[1:]
with open(csv_path,newline='') as f:
    rows=list(csv.reader(f))
assert len(rows)>1, 'No jobs generated'
s=Path(job_path).read_text()
checks={
  'Roihu GPU GRES': f'#SBATCH --gres=gpu:{gpu_type}:{gpu_count}' in s,
  'one CSV_PATH': len(re.findall(r'(?m)^CSV_PATH="[^"\n]+"$',s))==1,
  'packages bind': f'{packages}:/packages' in s,
  'container PYTHONPATH': 'export APPTAINERENV_PYTHONPATH=/packages' in s,
  'runtime dataset cache': f'export HF_DATASETS_CACHE="{cache}"' in s,
  'container dataset cache': f'export APPTAINERENV_HF_DATASETS_CACHE="{cache}"' in s,
}
for name, success in checks.items(): print(('PASS' if success else 'FAIL')+': '+name)
if not all(checks.values()): raise SystemExit('Generated script does not use persistent Roihu configuration')
print(f'Prepared {len(rows)-1} CSV rows')
PY_VALIDATE
# Remove scheduler-generated %N array concurrency throttle: SLURM policy controls concurrency.
# Set per-job walltime for long-running BIG-bench tasks.
python3 - "$SBATCH_FILE" "$WALLTIME" <<'PY_ARRAY'
from pathlib import Path
import re,sys
path=Path(sys.argv[1]); walltime=sys.argv[2]; src=path.read_text()
src,n=re.subn(r'(?m)^(#SBATCH --array=[^\n%]+)%[0-9]+$',r'\1',src)
if not re.search(r'(?m)^#SBATCH --array=',src):
    raise SystemExit('Generated sbatch has no array directive')
# The compiled template provides a time directive; if absent, add one after shebang.
if re.search(r'(?m)^#SBATCH --time=',src):
    src=re.sub(r'(?m)^#SBATCH --time=.*$', '#SBATCH --time='+walltime,src)
else:
    src=src.replace('#!/usr/bin/env bash\n','#!/usr/bin/env bash\n#SBATCH --time='+walltime+'\n',1)
path.write_text(src)
print('SLURM array unconstrained by script; per-job walltime:',walltime)
PY_ARRAY
bash -n "$SBATCH_FILE"

# ----------------------------- Container verification ----------------------
# Image was downloaded and inspected before scheduling. Recheck that it exists.
[[ -s "$SIF_IMAGE" ]] || die "Downloaded image disappeared: $SIF_IMAGE"

# ----------------------------- Package overlay -----------------------------
# Do not permit the first pip resolver pass to select Hugging Face Hub 2.x:
# transformers 4.45.2 needs huggingface-hub>=0.23.2,<1.0. Pin the other
# NVIDIA image-sensitive packages to avoid replacing compatible base versions.
log "Installing lm-eval 0.4.13 and version-constrained runtime dependencies into the external overlay"
mkdir -p "$CONTAINER_PKGS"
apptainer exec --bind "$CONTAINER_PKGS:/packages" "$SIF_IMAGE" \
    python3 -m pip install --target /packages \
    "numpy==1.26.4" "pyarrow==14.0.2" "datasets==2.19.2" \
    "huggingface-hub==0.35.3" "pandas==2.2.3" "packaging==24.2" \
    jsonlines numexpr pybind11 pytablewriter rouge-score sacrebleu scikit-learn sqlitedict

apptainer exec --bind "$CONTAINER_PKGS:/packages" "$SIF_IMAGE" \
    python3 -m pip install --target /packages --no-deps \
    "transformers==4.51.3" "tokenizers==0.21.1" "accelerate==0.34.2" \
    "lm-eval==0.4.13" "peft==0.15.2" evaluate

# ----------------------------- OpenEuroLLM upstream fixes -----------------
# These are the EXACT fail-fast source substitutions from OpenEuroLLM
# containers/lumi.def (LUMI patch section, not ROCm-specific).
# They were tested on Roihu with lm-eval 0.4.13: AGIEval [4] answer lists,
# and SQuAD v2's incompatible doc_to_text/doc_to_target signatures.
log "Applying OpenEuroLLM's AGIEval and SQuAD v2 lm-eval compatibility fixes"
python3 - "$CONTAINER_PKGS/lm_eval" <<'PY_UPSTREAM_PATCHES'
from pathlib import Path
import hashlib, sys
root=Path(sys.argv[1])
patches={
    'api/task.py': [(
        '                c[a]\n'
        '                if (c and isinstance(a, int))\n'
        '                # TODO: for multiple targets a is a list[str]. Fix this hack\n',
        '                c[a]\n'
        '                if (c and isinstance(a, int))\n'
        '                # Gold given as a list of choice indices (AGIEval: [4]) -> map through choices.\n'
        '                else c[a[0]]\n'
        '                if (c and isinstance(a, list) and len(a) and isinstance(a[0], int))\n'
        '                # TODO: for multiple targets a is a list[str]. Fix this hack\n',
    )],
    'tasks/squadv2/task.py': [
        ('    def doc_to_text(self, doc):\n', '    def doc_to_text(self, doc, doc_to_text=None):\n'),
        ('    def doc_to_target(self, doc):\n', '    def doc_to_target(self, doc, doc_to_target=None):\n'),
    ],
}
for rel, replacements in patches.items():
    f=root/rel; src=f.read_text()
    for old,new in replacements:
        if new in src:
            continue
        n=src.count(old)
        if n != 1:
            raise SystemExit(f'Unsupported lm-eval source for upstream patch {rel}: expected one match, found {n}')
        src=src.replace(old,new,1)
    f.write_text(src)
    print('Patched', rel, hashlib.sha256(src.encode()).hexdigest())
PY_UPSTREAM_PATCHES

# The installed harness has its OWN bundled BIG-bench YAML definitions;
# merely passing include_path does not reliably select OpenEuroLLM's overrides.
# Installing the project's own template at the bundled include target ensures
# the effective definitions use strict-match + remove_whitespace.
log "Installing official OpenEuroLLM BIG-bench whitespace-normalization template"
OFFICIAL_BIGBENCH_TEMPLATE="$CUSTOM_TASKS/bigbench_generate_until/_generate_until_template_yaml"
RUNTIME_BIGBENCH_TEMPLATE="$CONTAINER_PKGS/lm_eval/tasks/bigbench/generate_until_template_yaml"
[[ -f "$OFFICIAL_BIGBENCH_TEMPLATE" && -f "$RUNTIME_BIGBENCH_TEMPLATE" ]] || \
    die 'Missing official or installed BIG-bench template; check repository/harness versions'
cp "$OFFICIAL_BIGBENCH_TEMPLATE" "$RUNTIME_BIGBENCH_TEMPLATE"
grep -q 'function: remove_whitespace' "$RUNTIME_BIGBENCH_TEMPLATE" || die 'BIG-bench fix missing'
sha256sum "$RUNTIME_BIGBENCH_TEMPLATE" | tee "$MANIFEST_DIR/upstream-bigbench-template.sha256"
python3 -m compileall -q "$CONTAINER_PKGS/lm_eval/api/task.py" "$CONTAINER_PKGS/lm_eval/tasks/squadv2/task.py"

log "Testing overlay imports inside the container"
APPTAINERENV_PYTHONPATH=/packages \
APPTAINERENV_HF_TOKEN="${HF_TOKEN:-}" \
APPTAINERENV_HF_HOME="$HF_HOME" \
APPTAINERENV_HF_HUB_CACHE="$HF_HUB_CACHE" \
APPTAINERENV_HF_DATASETS_CACHE="$RUNTIME_HF_DATASETS_CACHE" \
APPTAINERENV_OELLM_CUSTOM_TASK_PATH="$CUSTOM_TASKS" \
apptainer exec \
    --bind "$CONTAINER_PKGS:/packages,$CACHE_ROOT:$CACHE_ROOT,$WORKSPACE:$WORKSPACE" \
    "$SIF_IMAGE" python3 - <<'PY'
import numpy, pyarrow, datasets, transformers, tokenizers, accelerate, lm_eval, peft, evaluate, torch
from packaging.version import Version
print("numpy", numpy.__version__)
print("pyarrow", pyarrow.__version__)
print("datasets", datasets.__version__)
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
import huggingface_hub, pandas, packaging
print("huggingface_hub", huggingface_hub.__version__)
print("pandas", pandas.__version__)
print("packaging", packaging.__version__)
print("accelerate", accelerate.__version__)
print("torch", torch.__version__)
assert Version(numpy.__version__) == Version("1.26.4")
assert Version(pyarrow.__version__) == Version("14.0.2")
assert Version(huggingface_hub.__version__) == Version("0.35.3")
assert Version(pandas.__version__) == Version("2.2.3")
assert Version(packaging.__version__) == Version("24.2")
assert Version(datasets.__version__) < Version("2.20")
assert Version(transformers.__version__) == Version("4.51.3")
assert Version(tokenizers.__version__) == Version("0.21.1")
import importlib.metadata as md
assert md.version("lm-eval") == "0.4.13", md.version("lm-eval")
from pathlib import Path
import lm_eval.api.task as task_source
from lm_eval.tasks.squadv2.task import SQuAD2
import inspect
assert 'isinstance(a, list)' in inspect.getsource(task_source)
assert 'doc_to_text=None' in str(inspect.signature(SQuAD2.doc_to_text))
assert 'doc_to_target=None' in str(inspect.signature(SQuAD2.doc_to_target))
print('AGIEval/SQuAD v2 source patches: PASS')
from transformers import Qwen3Config, AutoModelForCausalLM
assert AutoModelForCausalLM._model_mapping[Qwen3Config].__name__ == "Qwen3ForCausalLM"
from lm_eval.tasks import TaskManager, get_task_dict
from pathlib import Path
import os
custom_tasks=Path(os.environ["OELLM_CUSTOM_TASK_PATH"])
# lm-eval 0.4.13 uses Entry dataclass objects rather than dictionaries.
# Confirm the ACTUAL resolved task config (not just the YAML on disk).
manager=TaskManager(include_path=str(custom_tasks))
for name in ('bigbench_repeat_copy_logic_generate_until','bigbench_operators_generate_until',
             'bigbench_qa_wikidata_generate_until','bigbench_cs_algorithms_generate_until',
             'bigbench_dyck_languages_generate_until'):
    entry=manager.task_index[name]
    cfg=entry.cfg
    filters=cfg.get('filter_list', [])
    assert any(f.get('name')=='strict-match' and
               any(x.get('function')=='remove_whitespace' for x in f.get('filter', []))
               for f in filters), (name, filters)
    print('PASS: OpenEuroLLM BIG-bench filter:',name,'source:',entry.yaml_path)
p=Path('/packages/lm_eval/tasks/bigbench/generate_until_template_yaml')
assert 'remove_whitespace' in p.read_text()
print("Container import/version test: PASS")
PY

apptainer exec --bind "$CONTAINER_PKGS:/packages" "$SIF_IMAGE" \
    python3 -m pip freeze --path /packages | sort > "$MANIFEST_DIR/container-packages-freeze.txt"

# ----------------------------- Dataset pre-cache ----------------------------
# This deliberately uses a different cache from the Python 3.12 host scheduler.
# Otherwise datasets 5.x metadata can be unreadable by datasets 2.19.2.
log "Pre-downloading datasets from jobs.csv before any GPU submission"
APPTAINERENV_PYTHONPATH=/packages \
APPTAINERENV_HF_TOKEN="${HF_TOKEN:-}" \
APPTAINERENV_HF_HOME="$HF_HOME" \
APPTAINERENV_HF_HUB_CACHE="$HF_HUB_CACHE" \
APPTAINERENV_HF_DATASETS_CACHE="$RUNTIME_HF_DATASETS_CACHE" \
APPTAINERENV_HF_HUB_OFFLINE=0 \
apptainer exec \
    --bind "$CONTAINER_PKGS:/packages,$CACHE_ROOT:$CACHE_ROOT,$WORKSPACE:$WORKSPACE" \
    "$SIF_IMAGE" python3 - "$JOBS_CSV" "$CUSTOM_TASKS" <<'PY'
import csv, sys
from lm_eval.tasks import TaskManager, get_task_dict
csv_path, custom_tasks = sys.argv[1:]
tasks = set()
with open(csv_path, newline="") as f:
    for row in list(csv.reader(f))[1:]:
        if len(row) > 1 and row[1]:
            tasks.update(x.strip() for x in row[1].split(",") if x.strip())
if not tasks:
    raise SystemExit("No tasks found in jobs.csv; refusing to submit.")
print(f"Found {len(tasks)} unique tasks: {', '.join(sorted(tasks))}")
manager = TaskManager(include_path=custom_tasks)
import os, datasets
print("Container datasets version:", datasets.__version__, flush=True)
print("Container datasets cache:", datasets.config.HF_DATASETS_CACHE, flush=True)
print("Expected container cache:", os.environ.get("HF_DATASETS_CACHE"), flush=True)
if str(datasets.config.HF_DATASETS_CACHE) != os.environ.get("HF_DATASETS_CACHE"):
    raise SystemExit("Dataset cache environment mismatch: refusing to mix formats")
failures = []
# get_task_dict constructs task objects and *already downloads* their datasets.
# Load each independently so the first failing task is identifiable.
for name in sorted(tasks):
    print(f"Loading/caching task: {name}", flush=True)
    try:
        get_task_dict([name], manager)
        print(f"PASS: {name}", flush=True)
    except Exception as exc:
        print(f"FAILED: {name}: {type(exc).__name__}: {exc}", flush=True)
        failures.append(name)
if failures:
    raise SystemExit("Dataset pre-cache failures: " + ", ".join(failures))
print("All task datasets cached successfully.")
PY

# Validate the exact offline mode used by default in GPU jobs. This catches
# missing Hub metadata (e.g., allenai/ai2_arc) on the LOGIN node before sbatch.
if [[ "$JOB_HF_OFFLINE" == 1 ]]; then
    log 'Validating all task datasets can load with HF_HUB_OFFLINE=1'
    APPTAINERENV_PYTHONPATH=/packages \
    APPTAINERENV_HF_TOKEN="${HF_TOKEN:-}" \
    APPTAINERENV_HF_HOME="$HF_HOME" \
    APPTAINERENV_HF_HUB_CACHE="$HF_HUB_CACHE" \
    APPTAINERENV_HF_DATASETS_CACHE="$RUNTIME_HF_DATASETS_CACHE" \
    APPTAINERENV_HF_HUB_OFFLINE=1 \
    apptainer exec --bind "$CONTAINER_PKGS:/packages,$CACHE_ROOT:$CACHE_ROOT,$WORKSPACE:$WORKSPACE" \
      "$SIF_IMAGE" python3 - "$JOBS_CSV" "$CUSTOM_TASKS" <<'PY_OFFLINE'
import csv, sys
from lm_eval.tasks import TaskManager, get_task_dict
csv_path, custom_tasks = sys.argv[1:]
names = set()
with open(csv_path, newline='') as fh:
    for row in list(csv.reader(fh))[1:]:
        if len(row)>1 and row[1]:
            names.update(t.strip() for t in row[1].split(',') if t.strip())
manager=TaskManager(include_path=custom_tasks)
for task in sorted(names):
    print('Offline verify:', task, flush=True)
    get_task_dict([task], manager)
print('Offline dataset verification: PASS')
PY_OFFLINE
else
    log 'Online GPU-job mode selected. Ensure Roihu compute nodes have Hub access.'
fi

# The upstream generator emits HF_HUB_OFFLINE=1 by default for SLURM.
# Override that value only when --job-online was explicitly requested.
python3 - "$SBATCH_FILE" "$JOB_HF_OFFLINE" <<'PY_MODE'
from pathlib import Path
import re,sys
p=Path(sys.argv[1]); value=sys.argv[2]; s=p.read_text()
s,n=re.subn(r'(?m)^export HF_HUB_OFFLINE=[01]$', 'export HF_HUB_OFFLINE='+value,s)
if n != 1: raise SystemExit(f'Expected one HF_HUB_OFFLINE assignment, found {n}')
p.write_text(s)
PY_MODE
bash -n "$SBATCH_FILE"

log 'Writing activation helper for future Roihu shells'
python3 - "$WORKSPACE/activate_roihu.sh" "$REPO_DIR" "$WORKSPACE" "$CACHE_ROOT" "$HF_TOKEN_FILE" "$NO_HF_TOKEN" <<'PY_ACTIVATE'
from pathlib import Path
import shlex,sys
output, repo, workspace, cache, token_file, no_token = sys.argv[1:]
q=shlex.quote
content = f'''#!/usr/bin/env bash
export EVAL_BASE_DIR={q(workspace)}
export HF_HOME={q(cache+'/huggingface')}
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets_host_scheduler"
export PATH={q(repo+'/.venv/bin')}:"$PATH"
'''
if no_token == '1':
    content += 'unset HF_TOKEN\n'
else:
    content += 'if [[ -r ' + q(token_file) + ' ]]; then export HF_TOKEN="$(head -n 1 ' + q(token_file) + ' | tr -d "\\r\\n")"; fi\n'
Path(output).write_text(content)
PY_ACTIVATE
bash -n "$WORKSPACE/activate_roihu.sh"

# ----------------------------- Final validation -----------------------------
log "Writing reproducibility manifest"
{
    echo "date=$(date -Is)"
    echo "lm_eval_harness=0.4.13 + OpenEuroLLM LUMI compatibility patches"
    echo "sample_logging=$LOG_SAMPLES"
    echo "walltime=$WALLTIME"
    echo "hostname=$(hostname)"
    echo "architecture=$(uname -m)"
    echo "workspace=$WORKSPACE"
    echo "cache_root=$CACHE_ROOT"
    echo "host_datasets_cache=$HOST_HF_DATASETS_CACHE"
    echo "container_datasets_cache=$RUNTIME_HF_DATASETS_CACHE"
    echo "repo_url=$REPO_URL"
    echo "repo_commit=$REPO_COMMIT"
    echo "model_id=$MODEL_ID"
    echo "model_revision=$MODEL_RESOLVED_REVISION"
    echo "task_group=$TASK_GROUP"
    echo "job_hf_hub_offline=$JOB_HF_OFFLINE"
    echo "slurm_partition=$SLURM_PARTITION"
    echo "slurm_account=$SLURM_ACCOUNT"
    echo "gpu_type=$GPU_TYPE"
    echo "gpus_per_node=$GPUS_PER_NODE"
    echo "container=$SIF_IMAGE"
    echo "jobs_csv=$JOBS_CSV"
    echo "sbatch_file=$SBATCH_FILE"
    echo "template_persistently_configured=yes"
} | tee "$MANIFEST_DIR/run-manifest.txt"

git -C "$REPO_DIR" diff -- oellm/resources/clusters.yaml oellm/resources/template.sbatch \
    > "$MANIFEST_DIR/oellm-eval-local-patches.diff"

log "Fresh installation completed. Persistent source template configured for future oellm-eval schedule runs."
printf '\nTo use the CLI in a new shell:\n  source "%s/activate_roihu.sh"\n' "$WORKSPACE"
printf '\nEvaluation directory:\n  %s\nSLURM script:\n  %s\nManifest:\n  %s\n' \
    "$JOB_DIR" "$SBATCH_FILE" "$MANIFEST_DIR/run-manifest.txt"

if [[ "$SUBMIT" == "1" ]]; then
    log "SUBMIT=1: submitting the fully prepared evaluation array"
    sbatch "$SBATCH_FILE"
else
    printf '\nReview the generated file, then submit manually:\n  sbatch "%s"\n' "$SBATCH_FILE"
    printf 'Do not rerun the installer merely to submit; use the sbatch command above.\n'
fi
