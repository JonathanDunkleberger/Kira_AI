# resume_marathon.ps1 - the ONE-PASTE resume for the FireRed marathon (Windows PC).
#
# How the PC<->Mac loop works:
#   - This script inventories EVERY watch sandbox + the canonical save + crash logs
#     into docs\soak-reports\<timestamp>\ and pushes it to GitHub (the Mac agent's eyes).
#   - The Mac agent picks the correct save and writes pokemon_agent\PROMOTE_TARGET.txt
#     (a sandbox path, or the word CANONICAL to launch as-is), then pushes.
#   - Rerunning this script pulls that decision, promotes it through the sanctity gate,
#     and launches the marathon (bot + supervised game).
#
# PROMOTE_TARGET.txt contents (one-shot; consumed after use):
#   (absent)                  -> same as CANONICAL: launch the canonical save as-is
#                                (2026-08-01: the old inventory-only fallback killed the
#                                running processes and launched NOTHING -- and since the
#                                file is consumed after every launch, "absent" is the
#                                COMMON state; it stranded the marathon dark repeatedly)
#   CANONICAL                 -> canonical save is already right; just launch
#   <sandbox path>            -> monotonic promote of that sandbox, then launch
#   NEW_CAMPAIGN <path|AUTO>  -> archive current canonical campaign (trophy), then
#                                promote the sandbox as a FRESH campaign. AUTO picks
#                                the newest Squirtle-line sandbox with <=2 badges.
#   MIGRATE_SHOWTIME          -> stage the showtime run (states/kira) as a campaign
#                                bundle via showtime_to_campaign.py, archive the old
#                                campaign (trophy), promote the stage, launch free-roam.
#   RESTORE_WORLD             -> copy world_model.json (her learned mental map) from the
#                                newest campaign_archived_* trophy into the live campaign
#                                (the migration seeded it empty - she forgot the map),
#                                then launch. Caps are re-read from live RAM at boot, so
#                                trophy-era Fly/Surf flags can't leak into this run.
#
# USAGE:
#   powershell -ExecutionPolicy Bypass -File pokemon_agent\resume_marathon.ps1
# Flags:
#   -NoLaunch   do everything except start the bot/game
param([switch]$NoLaunch)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$report = Join-Path $RepoRoot "docs\soak-reports\$ts"
New-Item -ItemType Directory -Force -Path $report | Out-Null
$mainLog = Join-Path $report "resume_marathon.log"

function Say([string]$m) {
    Write-Host $m
    Add-Content -Path $mainLog -Value $m
}
function RunLogged([string]$label, [scriptblock]$cmd) {
    # Quiet by default: huge Format-Table dumps (states/ + logs\debug) were flooding the
    # console as "glitched numbers scrolling forever" and hid whether launch actually ran
    # (2026-08-02 13:43 chalk). Full text still goes to the soak report file.
    Say "== $label =="
    $out = & $cmd 2>&1 | Out-String -Width 300
    Add-Content -Path $mainLog -Value $out
    $lines = @($out -split "`r?`n" | Where-Object { $_ -ne "" })
    if ($lines.Count -le 40) {
        Write-Host $out
    } else {
        Write-Host ($lines[0..19] -join "`n")
        Write-Host ("... ($($lines.Count) lines -> $mainLog; not dumping all to console) ...")
        Write-Host ($lines[($lines.Count - 5)..($lines.Count - 1)] -join "`n")
    }
    return $out
}
# Newest file mtime anywhere inside a directory (folder mtimes lie on Windows).
function NewestFileTime([string]$dir) {
    $f = Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($f) { return $f.LastWriteTime } else { return (Get-Item $dir).LastWriteTime }
}
function HealthSummary([string]$dir) {
    $h = Join-Path $dir "health.json"
    if (-not (Test-Path $h)) { return "(no health.json)" }
    try {
        $j = Get-Content $h -Raw | ConvertFrom-Json
        $mins = [math]::Round($j.playthrough_s / 60)
        return "badges=$($j.badge_count) party=[$($j.party -join ', ')] place=$($j.place) played=${mins}min timeline=$($j.timeline)"
    } catch { return "(health.json unreadable)" }
}
function IsSquirtleRun([string]$dir) {
    $h = Join-Path $dir "health.json"
    if (-not (Test-Path $h)) { return $false }
    try {
        $j = Get-Content $h -Raw | ConvertFrom-Json
        $party = ($j.party -join " ")
        return (($party -match "squirtle|wartortle|blastoise") -and ([int]$j.badge_count -le 2))
    } catch { return $false }
}

Say "resume_marathon $ts  (repo: $RepoRoot)"

# 0) stop any running Kira/game so saves are quiescent and relaunch is clean
Say "== stopping any running Kira python processes =="
taskkill /F /IM python.exe /T 2>&1 | Out-Null
Start-Sleep -Seconds 2

RunLogged "git pull" { git pull } | Out-Null

$activate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $activate) { & $activate } else { Say "!! .venv not found - using system python" }

$envFile = Join-Path $RepoRoot ".env"
if (Test-Path $envFile) {
    if (-not (Select-String -Path $envFile -Pattern "^\s*POKEMON_AGENT_ENABLED\s*=\s*true" -Quiet)) {
        Say "!! WARNING: POKEMON_AGENT_ENABLED=true not found in .env - she will play MUTE."
    }
    if (-not (Select-String -Path $envFile -Pattern "^\s*OPENROUTER_API_KEY\s*=\s*\S+" -Quiet)) {
        # CONFIRMED live 2026-07-29: without this key every Claude call hits Anthropic directly and
        # dies with 'This organization has been disabled' (the ban) -> she silently runs on local
        # Llama all session (generic tone, repetitive phrasing). Fix it HERE, once, into .env.
        Say "!! OPENROUTER_API_KEY is missing from .env - her Claude brain is DOWN (Llama fallback)."
        $orKey = Read-Host "Paste your OpenRouter API key (starts with sk-or-) to fix this now, or press Enter to skip"
        if ($orKey -and $orKey.Trim().StartsWith("sk-or")) {
            Add-Content -Path $envFile -Value "`nOPENROUTER_API_KEY=$($orKey.Trim())"
            Say "OPENROUTER_API_KEY written to .env - Claude via OpenRouter is back for every future launch."
        } elseif ($orKey) {
            Say "!! that didn't look like an OpenRouter key (sk-or-...) - NOT written. She'll run on Llama."
        } else {
            Say "!! skipped - she'll keep running on local Llama until the key is added to .env."
        }
    }
    if (-not (Select-String -Path $envFile -Pattern "^\s*DISCORD_WEBHOOK_URL\s*=\s*\S+" -Quiet)) {
        # SUBATHON AUDIT 2026-07-30: the dead-man's switch (she's abandoned/wedged -> ping Jonny) posts
        # to this webhook. Without it the alert is a SILENT no-op -- you'd find her stopped hours later
        # instead of getting a phone buzz. Discord: Server Settings > Integrations > Webhooks > New.
        Say "!! DISCORD_WEBHOOK_URL is missing from .env - the dead-man's switch CANNOT reach you."
        Say "   (If she gets truly stuck mid-subathon, you will NOT be notified.)"
        $dwUrl = Read-Host "Paste a Discord webhook URL to fix this now, or press Enter to skip"
        if ($dwUrl -and $dwUrl.Trim().StartsWith("https://discord.com/api/webhooks/")) {
            Add-Content -Path $envFile -Value "`nDISCORD_WEBHOOK_URL=$($dwUrl.Trim())"
            Say "DISCORD_WEBHOOK_URL written to .env - abandon/crash-loop alerts will ping your Discord."
        } elseif ($dwUrl) {
            Say "!! that didn't look like a Discord webhook URL - NOT written. No alerts this run."
        } else {
            Say "!! skipped - she can still play, but a hard stop will be SILENT until you check."
        }
    }
} else { Say "!! WARNING: no .env at repo root" }

# 1) crash forensics
Say "== crash forensics =="
$dbg = Join-Path $RepoRoot "logs\debug"
RunLogged "logs\debug listing (newest 25)" {
    Get-ChildItem $dbg -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 25 |
        Select-Object Name, LastWriteTime, Length | Format-Table -AutoSize
} | Out-Null
Copy-Item (Join-Path $dbg "playlive_crash_*.log") $report -ErrorAction SilentlyContinue
Copy-Item (Join-Path $dbg "playlive_faulthandler.log") $report -ErrorAction SilentlyContinue
# tail of the newest debug log of any kind (the actual death note usually lives here)
$newestDbg = Get-ChildItem $dbg -File -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($newestDbg) {
    Get-Content $newestDbg.FullName -Tail 300 -ErrorAction SilentlyContinue |
        Set-Content (Join-Path $report ("tail_" + $newestDbg.Name))
    Say "tailed newest debug log: $($newestDbg.Name)"
}
# supervisor tee logs carry play_live's own stdout - the actual crash TRACEBACK lives here,
# not in the bot console (which only relays the supervisor's resume alert).
$supDir = Join-Path $dbg "supervisor"
Get-ChildItem $supDir -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 2 | ForEach-Object {
        Get-Content $_.FullName -Tail 1500 -ErrorAction SilentlyContinue |
            Set-Content (Join-Path $report ("tail_supervisor_" + $_.Name))
        Say "tailed supervisor log: $($_.Name)"
    }

# 2) FULL sandbox + canonical inventory (by newest FILE time, not folder time)
$watchRoot = Join-Path $env:TEMP "kira_watch"
Say "== sandbox inventory ($watchRoot) =="
$sandboxes = @(Get-ChildItem $watchRoot -Directory -ErrorAction SilentlyContinue |
    ForEach-Object {
        [pscustomobject]@{ Dir = $_; Newest = (NewestFileTime $_.FullName) }
    } | Sort-Object Newest -Descending)
if ($sandboxes.Count -eq 0) { Say "no sandboxes found under $watchRoot" }
$i = 0
foreach ($s in $sandboxes) {
    $i++
    Say ("[$i] " + $s.Dir.FullName)
    Say ("    last activity: " + $s.Newest)
    Say ("    " + (HealthSummary $s.Dir.FullName))
    Copy-Item (Join-Path $s.Dir.FullName "health.json") (Join-Path $report ("sandbox" + $i + "_" + $s.Dir.Name + "_health.json")) -ErrorAction SilentlyContinue
}
$campaign = Join-Path $RepoRoot "pokemon_agent\states\campaign"
Say "== canonical campaign =="
Say ("    last activity: " + (NewestFileTime $campaign))
Say ("    " + (HealthSummary $campaign))
Copy-Item (Join-Path $campaign "health.json") (Join-Path $report "canonical_health.json") -ErrorAction SilentlyContinue
# SNAPSHOT inventory (2026-07-30): every escape-reload / deep-wedge revert BANKS a timestamped .state
# first (pre_reload_* / pre_deepwedge_*). Listing them in the report lets the Mac agent pick a known-
# good position (e.g. "the moment she reached Cerulean") and pin it via the SNAPSHOT directive below.
RunLogged "campaign snapshot inventory (*.state, newest first)" {
    Get-ChildItem $campaign -File -Filter "*.state" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object Name, LastWriteTime, Length | Format-Table -AutoSize
} | Out-Null
# The showtime lineage (states\kira) banks a checkpoint at each SEGMENT SEAM (seg_cerulean.state =
# the moment she arrived in Cerulean, healthy). Same lineage as the migrated campaign, so one of
# these can be promoted via "SNAPSHOT kira/<file>" -- the real teleport-to-a-known-good-place.
RunLogged "showtime segment checkpoints (states\kira\*.state, newest first)" {
    Get-ChildItem (Join-Path $RepoRoot "pokemon_agent\states\kira") -File -Filter "*.state" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object Name, LastWriteTime, Length | Format-Table -AutoSize
} | Out-Null
# DENSE AUTO-CHECKPOINTS (2026-07-31): the campaign banks a labeled bundle every ~12 min into
# states\campaign\checkpoints\<ts>_<place>_<badges>b_<playtime>[_<reason>]. Listing the newest
# lets the Mac agent pin an exact position via "CKPT <name-substring>" -- the hard teleport.
RunLogged "auto-checkpoint inventory (newest 20)" {
    Get-ChildItem (Join-Path $campaign "checkpoints") -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike "*.partial" } |
        Sort-Object Name -Descending | Select-Object -First 20 |
        Select-Object Name, LastWriteTime | Format-Table -AutoSize
} | Out-Null
RunLogged "campaign dir + backups" {
    Get-ChildItem (Join-Path $RepoRoot "pokemon_agent\states") -Directory -Recurse -Depth 1 -ErrorAction SilentlyContinue |
        Select-Object FullName, LastWriteTime | Format-Table -AutoSize
} | Out-Null

# 3) promote ONLY if the Mac agent has pinned a target
$targetFile = Join-Path $RepoRoot "pokemon_agent\PROMOTE_TARGET.txt"
$promoteOk = $false
$launchApproved = $false

function PromoteBank([string]$bankDir) {
    $out = RunLogged "promote_bank (sanctity-gated) -> $bankDir" {
        python (Join-Path $RepoRoot "pokemon_agent\promote_bank.py") $bankDir "marathon_rescue_$ts"
    }
    Add-Content -Path (Join-Path $report "promote.log") -Value $out
    Say "promote exit code: $LASTEXITCODE  (0 = promoted; nonzero = canonical untouched)"
    return ($LASTEXITCODE -eq 0)
}

if (Test-Path $targetFile) {
    # First non-empty, non-# line only — comments after the directive used to poison CKPT
    # substring match (2026-08-02: whole file became the pattern, promote failed, no launch).
    $target = (
        Get-Content $targetFile -ErrorAction SilentlyContinue |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -and ($_ -notmatch '^\s*#') } |
        Select-Object -First 1
    )
    if (-not $target) {
        Say "!! PROMOTE_TARGET.txt has no directive line (only blanks/comments) - nothing changed, NOT launching."
    } elseif ($target -eq "CANONICAL") {
        Say "PROMOTE_TARGET = CANONICAL -> canonical save is already correct; launching as-is."
        $promoteOk = $true; $launchApproved = $true
    } elseif ($target -like "SNAPSHOT *") {
        # SNAPSHOT <file.state> (2026-07-30): promote a banked recovery snapshot from states/campaign
        # to be the living save -- the "teleport" (e.g. back to the moment she reached Cerulean).
        # Current save is backed up first, sidecars (world/soul/strat) untouched -- same campaign.
        $snapName = $target.Substring(9).Trim()
        # "kira/<file>" pulls a showtime segment checkpoint (same save lineage as the migrated
        # campaign); a bare name pulls a banked recovery snapshot from states\campaign.
        if ($snapName -like "kira/*") {
            $snapPath = Join-Path (Join-Path $RepoRoot "pokemon_agent\states\kira") $snapName.Substring(5)
        } else {
            $snapPath = Join-Path $campaign $snapName
        }
        $liveSave = Join-Path $campaign "kira_campaign.state"
        if ((Test-Path $snapPath) -and ($snapName -ne "kira_campaign.state")) {
            Copy-Item $liveSave (Join-Path $campaign "replaced_$ts.state") -ErrorAction SilentlyContinue
            Copy-Item $snapPath $liveSave -Force
            Say "SNAPSHOT promoted: $snapName -> kira_campaign.state (old save backed up as replaced_$ts.state)"
            $promoteOk = $true; $launchApproved = $true
        } else {
            Say "!! SNAPSHOT '$snapName' not found in states/campaign - nothing changed, NOT launching."
            Say "   (check the snapshot inventory in this report and re-pin the exact filename)"
        }
    } elseif ($target -like "CKPT *") {
        # CKPT <name-substring> (2026-07-31, "tp her to somewhere else in cerulean"): promote the
        # NEWEST dense auto-checkpoint whose dir name matches the substring (labels embed the
        # place, e.g. 20260731_134210_cerulean-city_2b_10h05m) -- the hard teleport that does not
        # depend on the game cooperating. Same contract as SNAPSHOT: live save backed up first,
        # sidecars untouched (same campaign, minutes older).
        $ckptPat = $target.Substring(5).Trim().ToLower()
        $ckptRoot = Join-Path $campaign "checkpoints"
        $ckptDir = Get-ChildItem $ckptRoot -Directory -ErrorAction SilentlyContinue |
                   Where-Object { $_.Name -notlike "*.partial" -and $_.Name.ToLower() -like "*$ckptPat*" } |
                   Sort-Object Name -Descending | Select-Object -First 1
        # Anti-Diglett-rewind (2026-08-02): Mac pins to Diglett/Route2/Pewter/Mt.Moon kept
        # forcing a 20-min re-walk. If a Rock Tunnel CKPT exists, prefer THAT instead.
        $westPoison = $ckptPat -match 'diglett|route-2|pewter|mt-moon|route-3|viridian|pallet|route-1'
        $tunnelDir = Get-ChildItem $ckptRoot -Directory -ErrorAction SilentlyContinue |
                     Where-Object { $_.Name -notlike "*.partial" -and $_.Name.ToLower() -like "*rock-tunnel*" } |
                     Sort-Object Name -Descending | Select-Object -First 1
        if ($westPoison -and $tunnelDir) {
            Say "!! CKPT override: '$ckptPat' is Diglett-west poison — promoting Rock Tunnel instead: $($tunnelDir.Name)"
            $ckptDir = $tunnelDir
        }
        $ckptState = if ($ckptDir) { Join-Path $ckptDir.FullName "kira_campaign.state" } else { $null }
        if ($ckptState -and (Test-Path $ckptState)) {
            $liveSave = Join-Path $campaign "kira_campaign.state"
            Copy-Item $liveSave (Join-Path $campaign "replaced_$ts.state") -ErrorAction SilentlyContinue
            Copy-Item $ckptState $liveSave -Force
            Say "CKPT promoted: $($ckptDir.Name) -> kira_campaign.state (old save backed up as replaced_$ts.state)"
            # Drop stuck LAW orders (e.g. catch_now Snorlax) so a TP doesn't re-wedge into the
            # same force-gym / catch thrash that made Jonny ask for the teleport (2026-08-02).
            $ord = Join-Path $campaign "creator_order.json"
            if (Test-Path $ord) {
                Remove-Item $ord -Force -ErrorAction SilentlyContinue
                Say "cleared creator_order.json (stuck LAW order would re-wedge the TP)"
            }
            $promoteOk = $true; $launchApproved = $true
        } else {
            Say "!! CKPT: no checkpoint matching '*$ckptPat*' with a kira_campaign.state - nothing changed, NOT launching."
            Say "   (see the auto-checkpoint inventory in this report and re-pin the exact name)"
        }
    } elseif ($target -eq "RESTORE_WORLD") {
        $troph = Get-ChildItem (Join-Path $RepoRoot "pokemon_agent\states") -Directory -Filter "campaign_archived_*" -ErrorAction SilentlyContinue |
                 Sort-Object Name -Descending | Select-Object -First 1
        $wm = if ($troph) { Join-Path $troph.FullName "world_model.json" } else { $null }
        if ($wm -and (Test-Path $wm)) {
            Copy-Item $wm (Join-Path $campaign "world_model.json") -Force
            Say "restored world model (mental map) from trophy: $($troph.Name)"
            $promoteOk = $true; $launchApproved = $true
        } else {
            Say "!! RESTORE_WORLD: no campaign_archived_*/world_model.json found - nothing changed, not launching."
        }
    } elseif ($target -eq "MIGRATE_SHOWTIME") {
        $stageOut = RunLogged "showtime_to_campaign (stage states/kira as campaign bundle)" {
            python (Join-Path $RepoRoot "pokemon_agent\showtime_to_campaign.py")
        }
        $stageLine = ($stageOut -split "`n" | Where-Object { $_ -match "^STAGED\s+" } | Select-Object -First 1)
        if ($stageLine) {
            $bank = $stageLine -replace "^STAGED\s+", "" -replace "\s+$", ""
            $arch = Join-Path $RepoRoot "pokemon_agent\states\campaign_archived_$ts"
            Say "staged OK: $bank"
            Say "archiving current canonical campaign -> $arch"
            Move-Item $campaign $arch
            New-Item -ItemType Directory -Force -Path $campaign | Out-Null
            $promoteOk = PromoteBank $bank
            if (-not $promoteOk) {
                Say "!! promote failed - restoring archived campaign to canonical (nothing lost)"
                Remove-Item $campaign -Recurse -Force -ErrorAction SilentlyContinue
                Move-Item $arch $campaign
            }
            $launchApproved = $promoteOk
        } else {
            Say "!! showtime_to_campaign did not stage a bundle - see its output above. Nothing touched."
        }
    } elseif ($target -like "NEW_CAMPAIGN*") {
        $spec = $target.Substring("NEW_CAMPAIGN".Length).Trim()
        $bank = $null
        if ($spec -eq "AUTO") {
            $pick = $sandboxes | Where-Object { IsSquirtleRun $_.Dir.FullName } | Select-Object -First 1
            if ($pick) { $bank = $pick.Dir.FullName; Say "AUTO picked Squirtle-run sandbox: $bank" }
            else { Say "!! AUTO found no Squirtle-line sandbox (<=2 badges) - nothing promoted." }
        } elseif ($spec -and (Test-Path $spec)) { $bank = $spec }
        else { Say "!! NEW_CAMPAIGN path doesn't exist: $spec" }
        if ($bank) {
            # archive the finished campaign as a trophy so monotonic doesn't block the new run
            $arch = Join-Path $RepoRoot "pokemon_agent\states\campaign_archived_$ts"
            Say "archiving current canonical campaign -> $arch"
            Move-Item $campaign $arch
            New-Item -ItemType Directory -Force -Path $campaign | Out-Null
            $promoteOk = PromoteBank $bank
            if (-not $promoteOk) {
                Say "!! promote failed - restoring archived campaign to canonical (nothing lost)"
                Remove-Item $campaign -Recurse -Force -ErrorAction SilentlyContinue
                Move-Item $arch $campaign
            }
            $launchApproved = $promoteOk
        }
    } elseif ($target -and (Test-Path $target)) {
        $promoteOk = PromoteBank $target
        $launchApproved = $promoteOk
    } else {
        Say "!! PROMOTE_TARGET.txt points to a path that does not exist: $target"
    }
    # one-shot: consume the decision so future runs go back to inventory mode
    Remove-Item $targetFile -ErrorAction SilentlyContinue
} else {
    # DEFAULT = CANONICAL (2026-08-01): the directive file is consumed one-shot after every
    # launch, so "absent" is the normal steady state -- treating it as inventory-only meant
    # every plain rerun killed Kira's processes and then launched nothing (stranded the
    # marathon repeatedly). Absent now behaves exactly like a CANONICAL directive; the
    # special one-shot directives above (SNAPSHOT/CKPT/NEW_CAMPAIGN/...) are unchanged.
    Say ">> no PROMOTE_TARGET.txt -- defaulting to CANONICAL launch (canonical save as-is)."
    $promoteOk = $true; $launchApproved = $true
}

# 4) push the report (and the consumed target file) to GitHub
# NOTE: separate git add calls - a pathspec that matches nothing must not sink the whole add.
Say "== push soak report (this can take ~30s; console stays quiet on purpose) =="
RunLogged "push soak report" {
    git add -A docs\soak-reports
    git add -A pokemon_agent\PROMOTE_TARGET.txt 2>&1 | Out-Null
    git commit -m "report(soak): $ts inventory/rescue (auto from resume_marathon.ps1)"
    git push origin main
} | Out-Null
Say "push soak report finished (see log if git complained)."

# 5) launch
if (-not $launchApproved) { Say "done (no launch this run)."; exit 0 }
if ($NoLaunch) { Say "done (-NoLaunch)."; exit 0 }

Say "== launching bot (window 1) =="
Say "    expect: a PowerShell titled roughly 'python run.py' + browser dashboard at :8766"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$RepoRoot'; .\.venv\Scripts\Activate.ps1; python run.py"

Say "waiting for dashboard at http://127.0.0.1:8766 (up to 3 min)..."
$botUp = $false
for ($j = 0; $j -lt 60; $j++) {
    Start-Sleep -Seconds 3
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8766/" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $botUp = $true; break }
    } catch { }
    if (($j % 5) -eq 4) { Say "    still waiting for bot... ($([int](($j+1)*3))s)" }
}
if (-not $botUp) {
    Say "!! bot never came up - check window 1 for the error, then rerun this script to send fresh logs."
    exit 1
}
Say "bot is up. == launching supervised marathon (window 2) =="
Say "    expect: a PowerShell for supervisor.py, THEN the FireRed / mGBA game window"
# FREE-ROAM (sherpa timeline) is the full-game engine - the one that finished fresh_go_6 to
# credits. The showtime spine is a scripted opener that ENDS at its last segment (Misty) and
# loop-retries a lost gym; it is not the marathon vehicle. Canonical campaign now carries the
# migrated stream run; the supervisor resumes it on every crash.
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$RepoRoot'; .\.venv\Scripts\Activate.ps1; python pokemon_agent\supervisor.py --timeline sherpa --audio"
Start-Sleep -Seconds 8
Say "She is live on FREE-ROAM: windowed, true speed, crash auto-restart, campaign banking."
Say "If the GAME window never appears: look at window 2 for a Traceback (ROM path / SDL / mgba)."
Say "To stop everything later: just rerun this script (it stops her first), or taskkill /F /IM python.exe /T"
