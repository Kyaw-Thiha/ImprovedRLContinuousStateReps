# Handoff: round 6 state, how to resume the SSP stage, how to close without it

Written 2026-11-06 by the agent running the round. Authoritative records: `../02-decisions.md` (d126-d156) and
`../../REPORT_006.md`. Runs live on R2 under `mjlab_ssp_centering/<campaign>/runs/<job>/`; nothing in this round depends
on the workstation being awake except the keepers below.

## 1. What is finished

| piece | campaign | files | status |
|---|---|---|---|
| G1 probe (6 jobs x 3000 its) | `r6a-ad66ecf2` | `runs/r6probe_*` | done: flat at 2075 -> claim length 2500 |
| G1 claim set (60 jobs x 2500 its) | `r6g1queue-0ede5636` | `runs/r6g1_*` | **done, 60/60**, $8.41 |
| H2 CartPole (3 arms, 100 trials + filter + 20-seed final) | `stage0_cartpole/refilter_v2/dqn_<arm>_none_v4_hparam_search/` | summaries + per-seed runs | done, ~$1.0 |

Read them with `analysis/g1_slope.py --probe`, `analysis/g1_claim_report.py`, `analysis/h2_report.py`;
`analysis/report6_checks.py` re-derives every number REPORT_006 quotes and exits non-zero if one moved.

## 2. What is queued and blocked

`r6sspwidthqueue-31a9570f`: 30 jobs = 12 width (raw reused from round 4; ssp 2x2, 4x4 and their matched RFF) + 18
length-scale sweep jobs (2x2 and 4x4, ls 0.03/0.1/1.0; 0.3 is the width cell). 60k iterations, ~$2 expected.

Blocked on community GPU capacity since 2026-11-05 15:24 UTC: 5,400+ refused creates and five captures, all lost in
boot to host faults (`boot_*.log` under the campaign's `logs/`). Two of them failed at torch initialisation with a
working `nvidia-smi` (driver 580.178, CUDA 13.0) on both wheel channels.

The chain is three systemd user units with linger enabled; each restarts on failure:

- `r6-width-after.service` - waits for 60 `DONE` in the claim campaign, then runs the width+sweep campaign itself
  (`--launch 1`, single build).
- `r6-pilot-after.service` - restores the `r6sspwidthqueue-` prefix from R2, applies the gate, and if it passes launches
  the 20-job pilot (`--launch 1`).
- `r6g1.service` - the claim campaign's hunter; dormant, the queue is drained.

Manual check: `python analysis/count_done.py --campaign r6sspwidthqueue-31a9570f`; `systemctl --user is-active r6-width-after`.

## 3. Reading the stage when it lands

1. `python analysis/restore_runs.py --out results/merged_r2 --prefix r6sspwidthqueue-`
2. `python analysis/ssp_report_table.py` - prints the table REPORT_006 section 2 needs, plus the gate verdict.
3. If the gate passes, the pilot has already been launched by the keeper; read it with
   `python analysis/ssp_readout.py pilot <n> <ls>` (n and ls come from the gate line).
4. Fill REPORT_006 section 2 and the section 5 line; update `report6_checks.py` with the new numbers so the check keeps
   covering the report.

## 4. Closing the round without the SSP stage

Everything needed is already written: REPORT_006 sections 0, 1, 3, 4, 4b and 5 are final, and section 2 carries an
explicit "run status" paragraph saying the stage did not run, why, and what was tried. To close:

1. Stop the keepers: `systemctl --user disable --now r6-width-after r6-pilot-after` (leave `r6g1` alone, it is dormant).
2. Delete any pod named `mjc-r6sspwidthqueue-*` if one appears.
3. In REPORT_006, replace section 2's TBD block with "not run: no community GPU was obtainable in 25+ h; see the status
   paragraph" and remove the section-5 TBD line.
4. Nothing else needs doing: the G1 and H2 results stand on their own, and the round's spend is about $11 of the $20 cap.

## 5. Traps found this round (do not repeat)

- A pod runs the manifest it downloaded; you cannot hand it more work. Merge stages that are not gated on another
  stage's result into one campaign (d145).
- A completion marker is not a claim: without a per-job R2 claim, two pods ran the same six jobs (d140).
- Check the host's CUDA before installing anything, install torch alone and smoke-test it, then the rest: a bad host
  should cost a minute, not 45 (d150, d151).
- A host that reports CUDA 13 can still fail the cu130 wheel; retry the same host with cu128 before discarding it (d153),
  and check `CUDA_VISIBLE_DEVICES` / `/dev/nvidia*` when `nvidia-smi` works but torch cannot see the device (d155).
- Keep one torch build per campaign: `--launch 1` for any stage whose cells are compared with each other (d152, d156).
- A keeper that reports "0" is a suspected failure, not an empty queue (d148).
