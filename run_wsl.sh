#!/usr/bin/env bash
# Run the repository inside WSL so CUDA-enabled PyTorch and RAPIDS/cuML
# are imported from the SAME Linux Python environment as Streamlit.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$ROOT_DIR/app.py" ]]; then
  echo "[ERROR] app.py was not found in: $ROOT_DIR"
  echo "Place run.bat and run_wsl.sh beside app.py in the repository root."
  exit 2
fi

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_wsl_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

trap 'status=$?; echo; echo "[ERROR] WSL runner stopped (exit=$status). Log: $LOG_FILE"; exit "$status"' ERR

echo "============================================================"
echo " TSNE CUSTOMER CLUSTERING - WSL2 RUNNER"
echo " Repository: $ROOT_DIR"
echo " Log: $LOG_FILE"
echo "============================================================"

# GPU visibility inside WSL2 is provided by the Windows NVIDIA driver.
HAS_NVIDIA_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAS_NVIDIA_GPU=1
  echo "[OK] NVIDIA GPU is visible inside WSL2."
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
else
  echo "[WARN] NVIDIA GPU is not visible inside this WSL distribution. CPU mode will be used where applicable."
fi

# Initialize Conda if present. The user can optionally select a distro/env
# from Windows by setting TSNE_WSL_DISTRO / TSNE_CONDA_ENV before launching.
CONDA_SH=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] && CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
fi
for candidate in \
  "$HOME/miniforge3/etc/profile.d/conda.sh" \
  "$HOME/mambaforge/etc/profile.d/conda.sh" \
  "$HOME/miniconda3/etc/profile.d/conda.sh" \
  "$HOME/anaconda3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh"; do
  if [[ -z "$CONDA_SH" && -f "$candidate" ]]; then
    CONDA_SH="$candidate"
    break
  fi
done

if [[ -n "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  echo "[OK] Conda initialized: $(conda info --base 2>/dev/null || true)"
else
  echo "[INFO] Conda was not found. Will use an available WSL Python environment."
fi

# Prefer an environment that can import RAPIDS cuML. Otherwise prefer the
# requested environment, then the current/default WSL Python.
PYTHON_BIN=""
CANDIDATE_PYTHONS=()

if [[ -n "${TSNE_CONDA_ENV:-}" && $(command -v conda || true) ]]; then
  ENV_PREFIX="$(conda env list 2>/dev/null | awk -v name="$TSNE_CONDA_ENV" '$1 == name {print $NF; exit}')"
  if [[ -n "$ENV_PREFIX" && -x "$ENV_PREFIX/bin/python" ]]; then
    CANDIDATE_PYTHONS+=("$ENV_PREFIX/bin/python")
  fi
fi

if command -v conda >/dev/null 2>&1; then
  while IFS= read -r prefix; do
    [[ -x "$prefix/bin/python" ]] && CANDIDATE_PYTHONS+=("$prefix/bin/python")
  done < <(conda env list 2>/dev/null | awk 'NF && $1 !~ /^#/ {print $NF}')
fi

command -v python3 >/dev/null 2>&1 && CANDIDATE_PYTHONS+=("$(command -v python3)")
command -v python >/dev/null 2>&1 && CANDIDATE_PYTHONS+=("$(command -v python)")

# Remove duplicate interpreter paths while preserving order.
UNIQUE_PYTHONS=()
declare -A SEEN_PYTHONS=()
for candidate in "${CANDIDATE_PYTHONS[@]}"; do
  resolved="$(readlink -f "$candidate" 2>/dev/null || printf '%s' "$candidate")"
  if [[ -z "${SEEN_PYTHONS[$resolved]+x}" ]]; then
    UNIQUE_PYTHONS+=("$resolved")
    SEEN_PYTHONS[$resolved]=1
  fi
done

for candidate in "${UNIQUE_PYTHONS[@]}"; do
  if "$candidate" -c 'from cuml.manifold import UMAP' >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    echo "[OK] Selected Python with cuML: $PYTHON_BIN"
    break
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ ${#UNIQUE_PYTHONS[@]} -gt 0 ]]; then
    PYTHON_BIN="${UNIQUE_PYTHONS[0]}"
  fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "[ERROR] No Python 3 interpreter was found inside WSL2."
  echo "Install Python/Conda inside Ubuntu, or configure the WSL distribution first."
  exit 3
fi

export PYTHON_BIN
printf '[*] Selected WSL Python: '
"$PYTHON_BIN" --version
"$PYTHON_BIN" -c 'import sys; print("[*] Interpreter:", sys.executable)'

# Ensure PyTorch is CUDA-enabled inside WSL. A CUDA-enabled Windows torch
# installation is separate and is intentionally not used by this runner.
TORCH_CUDA_OK=0
if "$PYTHON_BIN" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
  TORCH_CUDA_OK=1
fi

if [[ "$HAS_NVIDIA_GPU" == "1" && "$TORCH_CUDA_OK" == "0" ]]; then
  echo "[INFO] NVIDIA GPU is visible but this WSL Python does not have CUDA-enabled PyTorch."
  echo "[*] Installing PyTorch CUDA 12.8 into the selected WSL environment..."
  if ! "$PYTHON_BIN" -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128; then
    echo "[WARN] CUDA 12.8 wheel install failed; trying CUDA 12.6..."
    if ! "$PYTHON_BIN" -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126; then
      echo "[WARN] Could not install CUDA PyTorch. The app will continue; GPU dispatch may choose CPU."
    fi
  fi
fi

# Scan repository imports from inside the selected WSL interpreter and install
# missing Python distributions there. Import-name -> pip-name mappings prevent
# the old sklearn/scikit-learn false-positive.
SCANNER="$(mktemp /tmp/tsne_dependency_scan_XXXXXX.py)"
cat > "$SCANNER" <<'PY'
import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
skip_dirs = {
    ".git", ".venv", "venv", "env", "__pycache__", ".idea", ".vscode",
    "node_modules", "build", "dist", ".mypy_cache", ".pytest_cache",
}
stdlib = set(getattr(sys, "stdlib_module_names", ())) | set(sys.builtin_module_names) | {"__future__"}
module_to_distribution = {
    "sklearn": "scikit-learn",
    "umap": "umap-learn",
    "openTSNE": "openTSNE",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "dotenv": "python-dotenv",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "fitz": "pymupdf",
    "gi": "PyGObject",
}
required = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "umap": "umap-learn",
    "hdbscan": "hdbscan",
    "openTSNE": "openTSNE",
    "streamlit": "streamlit",
}
optional_rapids = {"cuml", "cudf", "cupy"}

py_files = [
    p for p in root.rglob("*.py")
    if not any(part in skip_dirs for part in p.relative_to(root).parts)
]
print(f"[*] Scanned {len(py_files)} Python files.")
imports = set()
for path in py_files:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
    except (SyntaxError, UnicodeError) as exc:
        print(f"[WARN] Skipped unparsable file {path}: {exc}")
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module.split(".")[0])

missing = {}
for mod, dist in required.items():
    try:
        found = importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        found = False
    if not found:
        missing[mod] = dist

for mod in sorted(imports):
    if not mod or mod in stdlib or mod in optional_rapids or mod in missing:
        continue
    try:
        found = importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        found = False
    if found:
        continue
    # Local project packages/modules are not pip dependencies.
    local_candidates = [root / f"{mod}.py", root / mod / "__init__.py"]
    if any(p.exists() for p in local_candidates):
        continue
    distribution = module_to_distribution.get(mod, mod)
    missing[mod] = distribution

if not missing:
    print("[OK] All detected Python imports are available.")
    raise SystemExit(0)

packages = sorted(set(missing.values()))
print("[*] Missing imports:", ", ".join(f"{m} -> {d}" for m, d in sorted(missing.items())))
print("[*] Installing missing distributions into:", sys.executable)
result = subprocess.run([sys.executable, "-m", "pip", "install", *packages])
raise SystemExit(result.returncode)
PY

SCAN_STATUS=0
"$PYTHON_BIN" "$SCANNER" || SCAN_STATUS=$?
rm -f "$SCANNER"
if [[ "$SCAN_STATUS" -ne 0 ]]; then
  echo "[ERROR] Dependency installation failed in the selected WSL Python environment."
  exit "$SCAN_STATUS"
fi

# Report the actual runtime backends before starting Streamlit.
"$PYTHON_BIN" - <<'PY'
import sys
print("[*] Runtime Python:", sys.executable)
try:
    import torch
    print("[*] PyTorch:", torch.__version__)
    print("[*] torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[*] CUDA device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("[WARN] PyTorch status unavailable:", repr(exc))
try:
    from cuml.manifold import UMAP
    print("[OK] cuML UMAP is importable in this runtime.")
except Exception as exc:
    print("[WARN] cuML UMAP unavailable in this runtime; UMAP will use the CPU fallback.")
    print("       Reason:", repr(exc))
PY

echo
echo "[*] Starting Streamlit inside WSL2..."
echo "[*] Browser URL: http://localhost:${TSNE_STREAMLIT_PORT:-8502}"
exec "$PYTHON_BIN" -m streamlit run "$ROOT_DIR/app.py" \
  --server.port "${TSNE_STREAMLIT_PORT:-8502}" \
  --server.address "0.0.0.0" \
  --server.headless true
