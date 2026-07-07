# NEXT_SESSION — resume prompt (write date 2026-07-07, night shift #2 close)

Paste this to the fresh session:

---

RESUME — fresh session. Read STATE §0 (newest block first). Never trust this file over
STATE §0 + NIGHT_REPORT.md if they disagree.

**CANONICAL = surf_taught: Fuchsia City (3,7)@(33,32), badges 6, sanctity VALID, round-trip
verified** (backup pre_surf_taught_backup_20260707_102403). Party: Venusaur L57 (Razor
Leaf/STRENGTH/Sleep Powder/Secret Power) / Persian 37 / Fearow 35 / Raticate 31 / Ekans 15
/ **LAPRAS L25 (SURF/Body Slam/Confuse Ray/Perish Song)**. Mankey L10 in the box. ~$71k.

🏅 **BANKED NIGHT SHIFT 2 (commits 0c53e3c → c52e096):**
1. **safari_hms PROMOTED** — GOLD TEETH + HM03 SURF + HM04 STRENGTH in one 50s entry
   (recon_safari strike 20). The tour chain: Center → EAST (43,15-17) → Area 1 (NW doors
   (8,9-11)) → Area 2 (S doors **(10-12,34) — the (20-22,34) group lands in the WRONG
   component of West**) → Area 3 (teeth (28,14), Secret House (12,7)) → reverse chain out.
2. **surf_taught PROMOTED** — Surf→Lapras (over Mist), Strength→Venusaur (over
   PoisonPowder). hm_teach._SLOT_TOPS gained the 24px-spacing anchor set (the 6-mon teach
   chooser; the old anchors missed slots 4/5 — Lapras was unreachable).
3. **General engine kills:** Grid.walkable excludes water (shore-treadmill class); the
   Grid guard CONTENT-VERIFIES vs ROM (dims-equal stale-backup class: Fuchsia vs Safari
   Center); safari_bfs = per-EDGE elevation law (equal or either 0/0xF; plateau/stair
   truth); nudge-free safari_step (ledge hops, tap-turn, no step-burn); movement never
   consumes a walk try; sea_ok requires collision-0 water (REEF class).

⚔️ **LIVE OBJECTIVE: THE SEAFOAM CROSSING → CINNABAR → BLAINE (badge 7).**
recon_cinnabar.py is PROVEN to Route 20 (mount actuation, surf glide, offset-aware
crossings — read its docstring STATUS first). **THE WALL: R20's surface is SEVERED at
Seafoam** (east sea x52..120, west sea x0..79, zero adjacency — dual-flood proven). The
crossing is the SEAFOAM INTERIOR (multi-floor ladder maze = the pad_plan class):
- R20 east door **(60,8)** → 1F arrive (6,21) → east-UPPER room → ladder **(30,8)** →
  B1F (29,8) → B2F+ (B1F's flood from (29,8) is 66 tiles, does NOT reach the return
  ladder (28,19) — the chain dives) → eventually 1F east-LOWER → exit **(32,21)** →
  R20 **(72,14)** = the WEST sea → surf west → **CINNABAR** (west connection, offset 0).
- **NEXT BUILD: port recon_sabrina.pad_plan** (warps-as-edges meta-BFS, zero hardcoded
  rooms) over the Seafoam floors — ladders are pads. Warp tables + layout bins for 1F/B1F
  cached in G:\temp\longrun\pret\ (fetch deeper floors the same way). Wilds inside are
  Zubat/Golbat/Slowpoke-class; BattleAgent handles. Watch for STRENGTH boulders (she HAS
  Strength now; field-move actuation via ht.TeachFlow.use_field_move — unverified live).
- After Cinnabar: heal → **MANSION (Secret Key, item ball in the basement)** → gym →
  **BLAINE = badge 7** → then Giovanni (Viridian, badge 8; spin_nav.py exists unwired) →
  Route 22/23 → Victory Road (Strength!) → E4 → **CREDITS**.

**KNOWN GAPS (owed):** Venusaur "AAAAAAAAAA" (Name Rater, Lavender); bench dead weight
(Ekans/Mankey); spin_nav unwired; _step_to's move-verify window too short for grass
(filed, campaign-shared); safari catches were silent RAM catches (no narration).
**SOUL-DEBT:** Lapras's first Surf = her first fielded moment — the roster-bond beat is
owed when play-live next runs. Dex +~9 species from the safari (nidoran♀♂, nidorino,
nidorina, exeggcute, venonat, parasect, rhyhorn, scyther attempts).

**WORKING-TREE LAW:** kira/* changes = Jonny's Gemini-vision WIP — NEVER commit or sweep.

Rules in force: EMPLOYMENT TERMS (two-wall shift ends, bank-and-continue), tripwire,
arsenal, single-run law, ground-truth-only, frontier-first NEXT_SESSION rewrites. The
diagnosis playbook that worked all night: instrument coords → deterministic replay →
probe the staged save offline (flood + frontier classify) → screenshot → pret. GO.

---

WATCH STATUS: canonical bank is CLEAN (surf_taught — badge 6, Gold Teeth delivered,
Surf on Lapras, Strength on Venusaur; she's outside the Warden's house in Fuchsia with
the whole southern sea ahead); press GO and you'll see her set out for the Seafoam
Islands on Lapras's back — the badge-7 leg; pop-in =
`.venv\Scripts\python.exe -u pokemon_agent\play_live.py --resume --free-roam`.
