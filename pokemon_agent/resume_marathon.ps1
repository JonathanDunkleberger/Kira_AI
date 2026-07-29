# resume_marathon.ps1 - the ONE-PASTE resume for the FireRed marathon (Windows PC).
#
# What it does, in order:
#   1. git pull (latest fixes arrive)
#   2. Collects crash forensics (logs\debug\playlive_crash_*.log + faulthandler) into
#      docs\soak-reports\<timestamp>\ so the Mac-side agent can read them from GitHub.
#   3. Finds the newest %TEMP%\kira_watch\sandbox_* (last night's REAL progress lives
#      there - watch.py never writes canonical) and promotes it through the sanctity
#      gate (promote_bank.py validates + backs up canonical first; a failed validation
#      changes nothing).
#   4. Commits + pushes the report folder to GitHub (the Mac agent's eyes).
#   5. Launches the marathon: run.py (bot) in one window, waits for the dashboard,
#      then supervisor.py --timeline sherpa --audio (windowed, true speed, crash
#      auto-restart, banks canonical progress continuously).
#
# USAGE (from anywhere):
#   powershell -ExecutionPolicy Bypass -File pokemon_agent\resume_marathon.ps1
# Flags:
#   -ReportOnly   only collect+push logs/sandbox info; no promote, no launch
#   -NoLaunch     do everything except start the bot/game
param([switch]$ReportOnly, [switch]$NoLaunch)

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
    $out = & $cmd 2>&1 | Out-String
    Write-Host $out
    Add-Content -Path $mainLog -Value $out
    return $out
}

Say "resume_marathon $ts  (repo: $RepoRoot)"
RunLogged "git pull" { git pull } | Out-Null

# venv
$activate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $activate) { & $activate } else { Say "!! .venv not found - using system python" }

# .env sanity
$envFile = Join-Path $RepoRoot ".env"
if (Test-Path $envFile) {
    if (-not (Select-String -Path $envFile -Pattern "^POKEMON_AGENT_ENABLED=true" -Quiet)) {
        Say "!! WARNING: POKEMON_AGENT_ENABLED=true not found in .env - she will play MUTE. Add it."
    }
    if (-not (Select-String -Path $envFile -Pattern "^OPENROUTER_API_KEY=sk-or" -Quiet)) {
        Say "!! WARNING: OPENROUTER_API_KEY missing in .env - her Claude brain will be offline."
    }
} else { Say "!! WARNING: no .env at repo root" }

# 1) crash forensics
Say "== collecting crash forensics =="
Copy-Item (Join-Path $RepoRoot "logs\debug\playlive_crash_*.log") $report -ErrorAction SilentlyContinue
Copy-Item (Join-Path $RepoRoot "logs\debug\playlive_faulthandler.log") $report -ErrorAction SilentlyContinue
$crashFiles = Get-ChildItem $report -Filter "playlive_*" -ErrorAction SilentlyContinue
Say ("crash files captured: " + ($(if ($crashFiles) { ($crashFiles | ForEach-Object Name) -join ", " } else { "NONE (native crash leaves none, or logs dir empty)" })))

# 2) newest watch sandbox -> promote
$promoteOk = $false
$watchRoot = Join-Path $env:TEMP "kira_watch"
$sb = Get-ChildItem $watchRoot -Directory -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($sb) {
    Say "newest watch sandbox: $($sb.FullName)  (modified $($sb.LastWriteTime))"
    Copy-Item (Join-Path $sb.FullName "health.json") (Join-Path $report "sandbox_health.json") -ErrorAction SilentlyContinue
    RunLogged "sandbox contents" { Get-ChildItem $sb.FullName -Recurse -Depth 1 | Select-Object FullName, LastWriteTime, Length | Format-Table -AutoSize } | Out-Null
    if (-not $ReportOnly) {
        $promoteOut = RunLogged "promote_bank (sanctity-gated)" { python (Join-Path $RepoRoot "pokemon_agent\promote_bank.py") $sb.FullName "watch_rescue_$ts" }
        $promoteOk = ($LASTEXITCODE -eq 0)
        Add-Content -Path (Join-Path $report "promote.log") -Value $promoteOut
        Say "promote exit code: $LASTEXITCODE  (0 = promoted; nonzero = canonical untouched)"
    }
} else {
    Say "!! no watch sandbox found under $watchRoot - Windows may have cleaned TEMP."
    Say "   Canonical stays wherever it was; the launch below still works, just from the older save."
}

# 3) push the report so the Mac-side agent can read everything
RunLogged "push soak report" {
    git add docs\soak-reports
    git commit -m "report(soak): $ts crash forensics + sandbox rescue (auto from resume_marathon.ps1)"
    git push origin main
} | Out-Null

# 4) launch
if ($ReportOnly -or $NoLaunch) {
    Say "done (no launch: flags). Tell the Mac agent to pull and read docs/soak-reports/$ts"
    exit 0
}
if (-not $promoteOk) {
    Say ">> promote did not succeed - launching anyway would resume from the OLDER canonical save."
    $go = Read-Host "Launch from older save? (y/N)"
    if ($go -ne "y") { Say "stopped. Report pushed - the Mac agent can diagnose."; exit 1 }
}

Say "== launching bot (window 1) =="
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$RepoRoot'; .\.venv\Scripts\Activate.ps1; python run.py"

Say "waiting for dashboard at http://127.0.0.1:8766 (up to 3 min)..."
$botUp = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 3
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8766/" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $botUp = $true; break }
    } catch { }
}
if (-not $botUp) {
    Say "!! bot never came up - check window 1 for the error, then rerun with -ReportOnly to send me the logs."
    exit 1
}
Say "bot is up. == launching supervised marathon (window 2) =="
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$RepoRoot'; .\.venv\Scripts\Activate.ps1; python pokemon_agent\supervisor.py --timeline sherpa --audio"
Say "She's live: windowed, true speed, crash auto-restart, canonical banking."
Say "To stop everything later: taskkill /F /IM python.exe /T"
