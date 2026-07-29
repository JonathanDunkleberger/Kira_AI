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
#   (absent)                  -> inventory + push only (safe, never touches saves)
#   CANONICAL                 -> canonical save is already right; just launch
#   <sandbox path>            -> monotonic promote of that sandbox, then launch
#   NEW_CAMPAIGN <path|AUTO>  -> archive current canonical campaign (trophy), then
#                                promote the sandbox as a FRESH campaign. AUTO picks
#                                the newest Squirtle-line sandbox with <=2 badges.
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
    Say "== $label =="
    $out = & $cmd 2>&1 | Out-String -Width 300
    Write-Host $out
    Add-Content -Path $mainLog -Value $out
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
        Say "!! WARNING: OPENROUTER_API_KEY missing in .env - her Claude brain will be offline."
    }
} else { Say "!! WARNING: no .env at repo root" }

# 1) crash forensics
Say "== crash forensics =="
$dbg = Join-Path $RepoRoot "logs\debug"
RunLogged "logs\debug listing (newest first)" {
    Get-ChildItem $dbg -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
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
    $target = (Get-Content $targetFile -Raw).Trim()
    if ($target -eq "CANONICAL") {
        Say "PROMOTE_TARGET = CANONICAL -> canonical save is already correct; launching as-is."
        $promoteOk = $true; $launchApproved = $true
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
        Say "!! PROMOTE_TARGET.txt points to a path that doesn't exist: $target"
    }
    # one-shot: consume the decision so future runs go back to inventory mode
    Remove-Item $targetFile -ErrorAction SilentlyContinue
} else {
    Say ">> No PROMOTE_TARGET.txt - inventory-only run. Nothing was promoted or launched."
    Say ">> Tell the Mac agent 'done' - it will read this report, pick the right save, and push the decision."
}

# 4) push the report (and the consumed target file) to GitHub
# NOTE: separate git add calls - a pathspec that matches nothing must not sink the whole add.
RunLogged "push soak report" {
    git add -A docs\soak-reports
    git add -A pokemon_agent\PROMOTE_TARGET.txt 2>&1 | Out-Null
    git commit -m "report(soak): $ts inventory/rescue (auto from resume_marathon.ps1)"
    git push origin main
} | Out-Null

# 5) launch
if (-not $launchApproved) { Say "done (no launch this run)."; exit 0 }
if ($NoLaunch) { Say "done (-NoLaunch)."; exit 0 }

Say "== launching bot (window 1) =="
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$RepoRoot'; .\.venv\Scripts\Activate.ps1; python run.py"

Say "waiting for dashboard at http://127.0.0.1:8766 (up to 3 min)..."
$botUp = $false
for ($j = 0; $j -lt 60; $j++) {
    Start-Sleep -Seconds 3
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8766/" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $botUp = $true; break }
    } catch { }
}
if (-not $botUp) {
    Say "!! bot never came up - check window 1 for the error, then rerun this script to send fresh logs."
    exit 1
}
Say "bot is up. == launching supervised marathon (window 2) =="
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$RepoRoot'; .\.venv\Scripts\Activate.ps1; python pokemon_agent\supervisor.py --timeline sherpa --audio"
Say "She's live: windowed, true speed, crash auto-restart, canonical banking."
Say "To stop everything later: just rerun this script (it stops her first), or taskkill /F /IM python.exe /T"
