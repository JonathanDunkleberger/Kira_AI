# PUSH_CHECKLIST — copy-paste kit for the public push (JONNY'S HANDS)

**Purpose:** the mission (secrets scan + hygiene + fact pack) is REVIEW-READY and cleared **GO** — see
`SECRETS_SCAN.md` and `README_FACTS.md`. This file is decision-support only: exact commands for each of
your ranked pre-push calls, both an **ACCEPT** path (do nothing) and a **REMEDIATE** path (if you'd rather
clean it). **Nothing here has been run.** The push itself is your hands. Pick per item, then push.

Remote: `origin  https://github.com/JonathanDunkleberger/Kira.git`
Branch reality: you are on `feature/pokemon-agent`, which is **1055 ahead / 1 behind** `main`. Decide the
target (see §0) BEFORE running any remediation, because a history rewrite (call #1) must happen on whatever
branch you actually publish.

---

## §0 — WHICH BRANCH GOES PUBLIC?  (decide first)

- **Option A — publish `feature/pokemon-agent` as the trunk.** Simplest. Rename it to `main` locally and
  push that. Everything below assumes you operate on the branch you'll publish.
- **Option B — merge into `main` first, publish `main`.** `git checkout main && git merge feature/pokemon-agent`
  (fast-forward won't apply — it's diverged 1/1055; expect a merge commit or `git reset --hard feature/pokemon-agent`
  if `main` has nothing you want).

> If you're going to do call #1 (author rewrite), do §0 FIRST, then rewrite the published branch, then push.

---

## §1 — Author metadata  [LOW · decision]

Every commit carries personal Gmail; real name "Jonny Dunkleberger" is in the author name; 2 early commits
carry a Mac hostname. See `SECRETS_SCAN.md` for the exact scope.

- **ACCEPT (recommended default — most solo OSS repos ship this way):** nothing to do. Optionally set a
  masked identity for *future* commits only:
  ```bash
  git config user.name "JonathanDunkleberger"
  git config user.email "JonathanDunkleberger@users.noreply.github.com"
  ```
- **REMEDIATE (destructive — rewrites ALL 1,344 commit SHAs; do on the branch you publish, and only if you
  accept that forks/clones break):** use `git-filter-repo` (install: `pip install git-filter-repo`). Draft a
  mailmap and run against a FRESH CLONE, never the working repo:
  ```bash
  # 1) fresh clone so the live repo is untouched
  git clone --no-local G:/JonnyD/NeuroAI_Bot /tmp/kira-publish && cd /tmp/kira-publish
  # 2) create mailmap.txt:  New Name <new@email>  Old Name <old@email>
  #    e.g.  JonathanDunkleberger <...@users.noreply.github.com>  Jonny Dunkleberger <jonny.dunk52@gmail.com>
  git filter-repo --mailmap mailmap.txt
  # 3) verify, then push THIS clone to origin
  ```
  ⚠️ This is irreversible and reshapes history — your call, your hands.

## §2 — Internal ops docs tracked  [LOW · KEEP, don't delete]

`CLAUDE.md`, `NIGHT_REPORT.md`, `NEXT_SESSION.md`, `STATE_OF_PROJECT.md` (+ archives) are tracked. They leak
internal process, not secrets. **Keep them on disk.**

- **ACCEPT:** leave them — they're harmless and arguably interesting to readers.
- **REMEDIATE (untrack from future commits, keep the files locally):**
  ```bash
  git rm --cached CLAUDE.md NIGHT_REPORT.md NEXT_SESSION.md STATE_OF_PROJECT.md
  printf '\n# internal ops docs\nCLAUDE.md\nNIGHT_REPORT.md\nNEXT_SESSION.md\nSTATE_OF_PROJECT.md\n' >> .gitignore
  git add .gitignore && git commit -m "chore: untrack internal ops docs from public tree"
  ```
  (They stay in *history* unless you also run the §1-style filter-repo path with `--path ... --invert-paths`.)

## §3 — Scratchpad-path recon scripts  [LOW · delete before push]

Exactly **12** `recon_*.py` hardcode a `G:\temp\claude\…scratchpad` path — throwaway probes, not product:
```
pokemon_agent/recon_bill_eyes.py      pokemon_agent/recon_bill_track.py
pokemon_agent/recon_captain_door.py   pokemon_agent/recon_captain_door2.py
pokemon_agent/recon_console_eyes.py   pokemon_agent/recon_gauntlet_fight.py
pokemon_agent/recon_itemuse.py        pokemon_agent/recon_mart.py
pokemon_agent/recon_route6_wedge.py   pokemon_agent/recon_ship_2f.py
pokemon_agent/recon_teach_derive.py   pokemon_agent/recon_vermilion_map.py
```
- **REMEDIATE (delete from tree):**
  ```bash
  git rm pokemon_agent/recon_bill_eyes.py pokemon_agent/recon_bill_track.py \
    pokemon_agent/recon_captain_door.py pokemon_agent/recon_captain_door2.py \
    pokemon_agent/recon_console_eyes.py pokemon_agent/recon_gauntlet_fight.py \
    pokemon_agent/recon_itemuse.py pokemon_agent/recon_mart.py \
    pokemon_agent/recon_route6_wedge.py pokemon_agent/recon_ship_2f.py \
    pokemon_agent/recon_teach_derive.py pokemon_agent/recon_vermilion_map.py
  git commit -m "chore: remove scratchpad-path recon probes"
  ```
  > Note: the other ~215 `recon_*.py` have no hardcoded scratchpad path. Broader "should the ~230 probes
  > ship at all?" is a taste call, not a blocker — SKIP unless you want a tidier tree.

## §4 — Repo bloat / winget .db  [V.LOW · optional]

16 KB winget `.db` is history-only (already gone from tree). ~30× 1 MB `campaign.py` revisions + an 11.5 MB
demo gif inflate the pack. None block a push. Shrinking requires the same filter-repo dance as §1 — **SKIP**
unless clone size bugs you.

## §5 — Canonical run-6 `banked_CREDITS` promotion  [PARKED · your call]

Purest post-credits candidate lives in `G:\temp\longrun\` (NOT in `states/campaign/`, which is UNTOUCHED).
Independent of the push — do it whenever you want a canonical post-credits save. Not required to publish.

---

## §6 — THE PUSH (after you've settled §1–§3)

```bash
git status                      # confirm clean / expected
git log --oneline -5            # eyeball the tip
# publish the branch you chose in §0:
git push -u origin main         # (or: git push -u origin feature/pokemon-agent)
```
Then set the repo public in the GitHub UI (Settings → Danger Zone → Change visibility) if it isn't already.

**Reminder:** if you did §1 REMEDIATE on a fresh clone, push from THAT clone, and a plain `git push` from the
live repo will be rejected (histories diverge) — that's expected; push the rewritten clone with `--force` to
the intended branch, deliberately.
