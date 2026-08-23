#!/usr/bin/env bash
# Provision a fresh vast.ai instance and verify the stack before spending money.
#
#   ssh -p <PORT> root@<HOST>
#   git clone <your-repo> && cd "Continual learning architecture"
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
echo "==> torch sees the GPU"
python - <<'PY'
import torch
print(f"    torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    device: {torch.cuda.get_device_name(0)}")
    print(f"    capability: {torch.cuda.get_device_capability(0)}")
else:
    raise SystemExit("    no CUDA device — stop here rather than paying for a CPU run")
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
