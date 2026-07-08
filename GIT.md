# GIT.md — repository conventions

Read this before committing, branching, or sharing models. CLAUDE.md points here; these rules
apply to every branch.

## Identity & privacy

- Commit identity is ALWAYS the GitHub noreply address:
  `git config user.email "57463941+eduardoloz@users.noreply.github.com"`
  Never commit with a personal or institutional email. History was scrubbed on 2026-07-08;
  do not reintroduce.
- Before any push, sanity-check: `git log --format=%ae origin/main..HEAD | sort -u`
  must show only the noreply address.
- Never commit: `.env`, API keys, tokens, personal email addresses, or credentials of any
  kind. `.env` is gitignored; keep it that way. Server hostnames/usernames are tolerated
  while the repo is private; scrub them before any visibility change to public.

## Branches

- `main`: the record of validated work. Every experiment conclusion that lands here must be
  seed-averaged (>=3 seeds) or cross-validated, logged in CLAUDE.md's experiment log and
  `results/config_benchmark.csv`.
- `opt/<topic>`: optimization/research campaign branches (e.g. `opt/jetson-nano`). Merge or
  archive when the campaign concludes; do not let them rot.
- `exp/<topic>`: short-lived experiment branches; delete after merging or abandoning.
- `worktree-agent-*`: transient local worktree branches; never push them.
- Force-pushes to `main` are exceptional (history scrubs only) and require coordinating every
  active worktree first.

## Commits

- Summary line: imperative, specific, with the headline number when experimental
  ("clean-data benchmark: champ 0.784 vs base 0.736 (3 seeds)").
- Body: what changed, why, and where the evidence lives (results file, server run dirs).
- AI-assisted commits end with the trailer:
  `Co-Authored-By: Claude <noreply@anthropic.com>` (model name optional).
- One logical change per commit; keep generated artifacts (figures) in the same commit as
  the code that generated them.

## Model sharing & training details (standard)

Trained models are shared via **GitHub Releases** (preferred; tag `vX.Y-<name>`, weights
attached as assets) or a `models/` folder for small `.pt`/`.onnx` files. Rules:

- Share `.pt` and/or `.onnx`. NEVER share TensorRT `.engine` files (device+version specific;
  deployers build their own with `optimization/jetson/` scripts).
- Every shared model REQUIRES a model card `models/<name>.md` committed in the same
  release/commit. No card, no share.

### Model card template (copy verbatim, fill every field)

```markdown
# <model name, e.g. champ_v11n_768>

- **Architecture:** YOLOv11n (2.6M params) | task: detect | obb
- **Dataset:** <local dir on server + Roboflow project/version, e.g.
  ATLI_target_tightNI_noCPLID (797 imgs, 561/116/120) = atli_target-* post-purge versions>
- **Recipe (exact reproduction command):**
  `MODEL=yolo11n.pt EXTRA="scale=0.9 seed=<s>" DATA=<yaml> bash run_config_ext.sh <NAME> <gpu> 150 100 768 16`
  (2-stage TL: SGD lr0 0.01 -> 0.00334, lrf 0.1535; list every non-default arg)
- **Oversampling/augmentation:** <e.g. DD x3 image-level, scale=0.9; defaults otherwise>
- **Seeds trained:** <e.g. 0,1,2>; this file's weights = seed <s>
- **Test metrics (seed-mean ± std, and this checkpoint's own):**
  mAP@0.5 <x.xxx ± x.xxx>, Defective_Damper AP <x.xxx ± x.xxx>; per-class table or link
- **Test split:** <which split, image count, known caveats (e.g. 12 DD instances)>
- **Export lineage:** best.pt -> ONNX opset 12 (-0.00x mAP) [-> engine built on-device]
- **Provenance:** server run dir `~/atli/runs/<name>`, trained <date>, scripts at commit `<sha>`
- **Known limitations:** <e.g. blur-fragile: -0.25 mAP under 7px motion blur; not comparable
  to pre-purge benchmarks>
```

## Experiment hygiene (summary; details in CLAUDE.md)

- Never quote single-run numbers; seed-mean ± std minimum, CV for paper claims.
- New conditions get logged in CLAUDE.md's experiment log + `results/config_benchmark.csv`,
  then pushed, so GitHub always reflects the full record.
- External data requires the pHash+filename leakage gate before it touches training.

## Names & private content in markdown (added 2026-07-08)

- Committed markdown must not contain: personal names of advisors/lab members (use "the PI",
  "a co-PI", "the lab", "et al." for citations), personal email addresses, or the server
  login. The server is referenced as `$ATLI_SERVER` in docs; the real `user@host` lives only
  in the untracked `.env` (`ATLI_SERVER=...`).
- Email drafts and outreach text live in `results/email_to_PI*.md` / `results/LOCAL_*` and
  are gitignored. History was scrubbed of both on 2026-07-08 (two rewrites; final heads
  main 9eb9d1e).
- Pre-commit check before any push:
  `git diff --cached | grep -iE "yang|zhai|@gmail|@unlv|LozanoE" ` must return nothing
  (the unrelated bibtex author "Yang, Xianjun" in damper_dataset_vetting.md is the one
  allowed exception).
