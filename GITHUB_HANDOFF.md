# GitHub Handoff — atli-inspection repository

One-page orientation for anyone (new lab member, future session, collaborator) working with
this repo. The binding rules live in `GIT.md`; this document is the map.

## What this repo is

Private repo `eduardoloz/atli-inspection`: code, experiment records, figures, and model cards
for the ATLI transmission-line defect detection project (YOLO-based, UAV imagery, UNLV).
Trained weights are NOT in git — they live on the training server (`$ATLI_SERVER`, real
address in the untracked `.env`) under `~/atli/runs/`, and are distributed via GitHub
Releases when needed.

## Branch map

| branch | contents | status |
|---|---|---|
| `main` | code, experiment log (CLAUDE.md), results, figures, model cards | active; the record of validated work |
| `opt/orchestrator` | edge-deployment campaign master log (`README-OPTIMIZATION.md`: final recipe, gate scoreboard, verification ledger) | campaign complete (2026-07-09) |
| `opt/jetson-nano` | Jetson research, export/pruning scripts, decision memo, prune-grid results | campaign complete |
| `opt/thesis-enhancements` | thesis digest, transferability analysis, final synthesis | campaign complete |
| `backup-*` | pre-history-rewrite snapshots kept locally in worktrees | do not push |
| `worktree-agent-*` | transient session worktree branches | never push |

The three `opt/*` branches are finished research campaigns: read them, cite them, merge
selected docs into `main` if desired — but new work goes on new branches (`opt/<topic>` or
`exp/<topic>`, see GIT.md).

## Non-negotiable rules (details in GIT.md)

1. **Commit identity**: only `57463941+eduardoloz@users.noreply.github.com`. Check with
   `git config user.email` before your first commit in any clone/worktree.
2. **Privacy**: no advisor/lab personal names, no personal emails, no server login in any
   committed file. Use "the PI", "et al.", `$ATLI_SERVER`. Pre-push gate:
   `git diff origin/<branch>..HEAD | grep -iE "yang|zhai|soum|@gmail|@unlv|LozanoE"`
   must return nothing (sole exception: the unrelated bibtex author "Yang, Xianjun").
3. **Never commit**: `.env`, API keys, PDFs, email drafts (`results/email_to_PI*`,
   `results/LOCAL_*` — all gitignored). Never share TensorRT `.engine` files.
4. **No force-pushes.** History was deliberately rewritten twice on 2026-07-08 (see below);
   that was a one-time sanctioned scrub, not a precedent.
5. **Every result quoted anywhere must be seed-averaged (≥3 seeds) or cross-validated**, and
   logged in CLAUDE.md's experiment log + `results/config_benchmark.csv` in the same push.

## The 2026-07-08 history rewrite (context you may need)

All branches were rewritten twice: (1) commit author emails → GitHub noreply; (2) markdown
content redacted (personal names → "the PI"/"et al.", server login → `$ATLI_SERVER`,
`results/data_outreach.md` deleted from all history). Consequences:
- Any clone/checkout from before 2026-07-08 has dead SHAs — re-clone or hard-reset to
  `origin/<branch>`; cherry-pick only commits whose messages are absent from the remote.
- Old commits may remain fetchable by SHA on GitHub's servers until garbage collection; a
  GitHub Support request can hard-purge them if ever needed.
- Recommended account settings: GitHub → Settings → Emails → enable "Keep my email addresses
  private" AND "Block command line pushes that expose my email".

## Model cards (`models/`)

One folder per trained model; the card is `models/<name>/README.md` (GitHub renders it on
folder view). Mandatory fields incl. exact reproduction commands, dataset split with
per-class annotation counts, full augmentation config, per-class metrics ± std, provenance,
limitations, and **whether eduardos-annotated-photos was included** (all current models: NOT
included). Template in `GIT.md`. Registry + augmentation-mechanics explainer:
`models/README.md`. No card, no share.

## Common tasks

- **Log a new experiment**: append the seed-averaged condition to CLAUDE.md's experiment log
  + a row in `results/config_benchmark.csv`, commit both together, push.
- **Add a trained model**: create `models/<name>/README.md` from the GIT.md template, verify
  every number against the run's own logs/args.yaml (not memory), run the privacy gate,
  commit.
- **Share weights**: GitHub Release tagged `vX.Y-<name>`, attach `.pt`/`.onnx` + point to the
  card. Engines are built on-device (`optimization/jetson/build_engine_nano.sh`).
- **Regenerate figures**: `python3 results/figures/generate_clean_vs_prior.py` (data is
  inline in the script; commit script + PNGs together).

## Key documents index

- `CLAUDE.md` — project state + full experiment log (start here)
- `GIT.md` — binding git/privacy/model-card conventions
- `results/pi_summary_2026-07-08.md` — data-only summary of the CPLID decontamination +
  deployment study
- `results/cplid_before_after.md` — clean benchmark, removed-image composition, per-class
  before/after
- `results/cplid_purge_roboflow.md` — Roboflow purge audit trail + API gotchas
- `README-OPTIMIZATION.md` (on `opt/orchestrator`) — final deployment recipe + gates
- `optimization/jetson/decision_memo.md` (on `opt/jetson-nano`) — Nano vs Orin decision
