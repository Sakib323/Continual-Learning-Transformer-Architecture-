#!/usr/bin/env bash
# Provision a fresh vast.ai instance and verify the stack before spending money.
#
#   ssh -p <PORT> root@<HOST>
#   git clone https://github.com/Sakib323/Continual-Learning-Transformer-Architecture-.git
#   cd Continual-Learning-Transformer-Architecture-
#   bash scripts/vast_setup.sh
#
# Ends with a pilot timing run, so the last thing you see is what the sweep will
# actually cost on the card you just rented.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "==> host"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || \
  echo "    no nvidia-smi — CPU-only instance?"
python3 --version

echo
echo "==> environment"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e .
pip install --quiet pytest

echo
echo "==> torch can actually use this GPU"
python - <<'PY'
import sys, torch

print(f"    torch {torch.__version__}  cuda_build={torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("    no CUDA device — stop here rather than paying for a CPU run")

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"    device     : {name}")
print(f"    capability : sm_{cap[0]}{cap[1]}")
print(f"    torch archs: {' '.join(torch.cuda.get_arch_list())}")

# is_available() is NOT enough. On a card newer than the wheel — Blackwell
# (sm_120: RTX 50xx) on a pre-2.7 / pre-cu128 torch — it returns True and then
# every kernel dies with "no kernel image is available for execution on the
# device". Launch a real kernel to find out now rather than 40 runs in.
try:
    x = torch.randn(256, 256, device="cuda")
    torch.mm(x, x).sum().item()
    torch.cuda.synchronize()
except RuntimeError as e:
    sys.exit(
        f"    CUDA kernel launch FAILED: {e}\n"
        f"    torch {torch.__version__} has no kernel for sm_{cap[0]}{cap[1]} ({name}).\n"
        f"    Fix: pip install --upgrade --index-url "
        f"https://download.pytorch.org/whl/cu128 torch\n"
        f"    (Blackwell/RTX 50xx needs torch >= 2.7 built against CUDA 12.8.)"
    )
print("    kernel launch ok")
PY

echo
echo "==> contract tests"
python -m pytest tests/ -q

echo
echo "==> registered mechanisms"
python train.py --list

echo
echo "==> smoke run (nano, 2 tasks) — proves the loop end to end"
python train.py --preset control_sequential --name smoke \
  --set model.size_preset=nano --set stream.steps_per_task=30 \
  --set stream.tasks=copy,reverse --set stream.eval_batches=2 \
  --set optim.warmup_steps=5 --set run.out_dir=/tmp/clms_smoke >/dev/null
echo "    ok"

echo
echo "==> pilot timing on this card"
python scripts/pilot.py --tune \
  --set model.size_preset="${SIZE:-small}" \
  --set stream.steps_per_task="${STEPS:-400}"

cat <<'EOF'

Ready. Next:

  # the full tuned ablation
  python scripts/sweep.py --preset ablation --tune --seeds 0,1,2 \
      --set model.size_preset=small

  # resumable: re-running the same command picks up from checkpoint.pt,
  # mechanism state included, so an interrupted instance costs one task
  # rather than the whole run

Record the card in your notes. Different GPUs mean different numerics, and a
mid-sweep hardware change shows up in results looking exactly like a mechanism
effect — every result.json stores the device, but only you can decide to re-run.
EOF
