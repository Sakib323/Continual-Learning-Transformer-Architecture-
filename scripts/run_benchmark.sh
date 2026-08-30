#!/usr/bin/env bash
# Unattended two-track benchmark. Start it and walk away.
#
#   bash scripts/run_benchmark.sh
#   AUTO_STOP=1 bash scripts/run_benchmark.sh      # power off when finished
#
# Runs track A, then track B, then writes both reports and a tarball, without
# needing anyone at the keyboard between stages.
#
# Scenario is class_il throughout. Under task_il, XdG is handed the task id at
# inference while nothing else is, which is not a comparison — and class_il is
# the setting a continuously-updated model actually operates in. The harness
# refuses to run XdG here at all rather than emit a meaningless number, so it is
# absent from the mechanism list below by design, not by oversight.

set -uo pipefail    # deliberately not -e: one failing preset must not kill the run

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate

TASKS="copy,modadd7,reverse,sort,modadd13,induction6,copy12,sortdesc,modadd23,reverse12,induction8,modadd31"
SEEDS="${SEEDS:-0,1,2}"
COMMON="--seeds $SEEDS --set stream.scenario=class_il --set stream.tasks=$TASKS --set stream.eval_batches=2"

# Track A — retention under interference, at full capacity.
A_PRESETS=(ewc si gpm lwf lora olora l2p memory_layer memory_sparse sparse_update kwta cbp shrink_perturb)
# Track B — plasticity, at a capacity that actually saturates. Measured: at nano
# over 12 tasks, modadd accuracy collapses 1.00 -> 0.04 by position, while at
# `small` there is no decay at all. Scoring the plasticity family on a stream
# with no plasticity loss was the unfair part.
B_PRESETS=(cbp shrink_perturb kwta gpm ewc si sparse_update)

started=$(date -u +%s)
banner() { printf '\n%s\n== %s\n%s\n' "$(printf '=%.0s' {1..70})" "$1" "$(printf '=%.0s' {1..70})"; }

banner "TRACK A — forgetting (small, 12 tasks, class_il)"
for P in "${A_PRESETS[@]}"; do
  echo "[track A] $P"
  python scripts/sweep.py --preset "$P" --tune $COMMON \
      --set model.size_preset=small --set run.out_dir=runs_A 2>&1 | tee -a sweep_A.log
done

banner "TRACK B — plasticity (nano, 12 tasks, class_il)"
for P in "${B_PRESETS[@]}"; do
  echo "[track B] $P"
  python scripts/sweep.py --preset "$P" --tune $COMMON \
      --set model.size_preset=nano --set run.out_dir=runs_B 2>&1 | tee -a sweep_B.log
done

banner "REPORTS"
python scripts/mechanism_report.py runs_A --all-points 2>&1 | tee report_A.txt
python scripts/mechanism_report.py runs_B --all-points 2>&1 | tee report_B.txt

# One small artefact to copy off, so a half-awake scp cannot miss half the run.
tar czf results.tar.gz runs_A runs_B report_A.txt report_B.txt \
    sweep_A.log sweep_B.log 2>/dev/null

elapsed=$(( $(date -u +%s) - started ))
banner "DONE in $((elapsed / 3600))h $(((elapsed % 3600) / 60))m — results.tar.gz ready"
echo "Retrieve with:"
echo "  scp -P <PORT> root@<HOST>:$REPO/results.tar.gz ~/Downloads/"

if [ "${AUTO_STOP:-0}" = "1" ]; then
  echo
  echo "AUTO_STOP set — powering off in 120s. Results are on disk and survive a"
  echo "vast.ai *stop*; they do NOT survive a *destroy*. Ctrl-C to cancel."
  sleep 120
  poweroff 2>/dev/null || shutdown -h now 2>/dev/null || \
    echo "could not power off from inside the container — stop it from the vast.ai UI"
fi
