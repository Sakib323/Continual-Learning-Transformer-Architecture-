Raw output of scripts/gpm_diagnostics.py, 2026-09-13, CPU, torch 2.13.0.
sections_A-E.log  eval leakage, PAD dilution, LR schedule, AdamW step leak, rank rule
section_F.log     protected-subspace accuracy
section_G.log     current vs corrected GPM, 6 tasks, 2 seeds

tiny12_class_il.log           12-task Class-IL, tiny, 1 seed, 300 steps/task, harness fixes on: control x2, gpm, gpm_v2 x2
runs_tiny12/                  result.json + config.yaml for each of those five runs
leak_trace_before_fix_cpu.log per-layer basis orthonormality and step leak, Gram-SVD extension (D7 visible)
leak_trace_after_fix_cpu.log  same after the fix
