# night_shift.ps1 -- THE NIGHT-SHIFT RELAUNCHER, UNLIMITED EDITION (London Mode phase 1, CEO-spec 2026-07-06)
#
# Loops autonomous Claude sessions off NEXT_SESSION.md until ONE of exactly three things:
#   (a) THE CREDITS ROLL  -- a session wrote "CREDITS" at the top of NIGHT_REPORT.md; loop stops.
#   (b) Jonny stops it    -- Ctrl+C this window (or close it).
#   (c) THE ONE BRAKE     -- two CONSECUTIVE shifts with an IDENTICAL NEXT_SESSION.md frontier AND
#                           zero new commits (provable nothing twice). Stuck-but-committing
#                           CONTINUES; slow-but-moving CONTINUES; a fourth angle with new code
#                           CONTINUES. Only the same-wall-for-hours-producing-nothing machine stops.
#
# Sessions run with --dangerously-skip-permissions (Jonny-approved 2026-07-06): fully autonomous,
# no per-action gates. Safety = the sanctity gate on every canonical promotion + git-revertibility
# of everything else.
#
# BEDTIME:   powershell -ExecutionPolicy Bypass -File .\night_shift.ps1
# TEST:      powershell -ExecutionPolicy Bypass -File .\night_shift.ps1 -Test   (one attended cycle,
#            harmless prompt -- proves launch/report/brake plumbing without driving the emulator)
param(
    [int]$SleepBetween = 60,
    [switch]$Test
)
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Report = Join-Path $Root "NIGHT_REPORT.md"
$Frontier = Join-Path $Root "NEXT_SESSION.md"
$LogDir = Join-Path $Root "logs\nightshift"
New-Item -ItemType Directory -Force $LogDir | Out-Null
if (-not (Test-Path $Report)) {
    @"
# NIGHT REPORT -- started $(Get-Date -Format 'yyyy-MM-dd HH:mm')
One line per shift below (newest last). The winning session promotes the magic word to line 1.

"@ | Set-Content $Report
}

function Get-FrontierHash {
    if (Test-Path $Frontier) { (Get-FileHash $Frontier -Algorithm SHA256).Hash } else { "" }
}

$noProgress = 0
$shift = 0
while ($true) {
    # (a) CREDITS check -- the deliverable. LINE 1 ONLY, anchored: the header text
    # below must never contain the token itself (the first attended test false-fired on its
    # own documentation line).
    $top = (Get-Content $Report -TotalCount 1)
    if ($top -match "^\s*CREDITS") {
        Write-Host "== THE CREDITS ROLLED. The mountain is climbed. Read NIGHT_REPORT.md. =="
        break
    }
    if (-not (Test-Path $Frontier)) {
        Add-Content $Report "- $(Get-Date -Format 'HH:mm') NEXT_SESSION.md MISSING -- stopping for human eyes."
        break
    }

    $shift++
    $startHead = (git rev-parse HEAD 2>$null)
    $startFrontier = Get-FrontierHash
    $log = Join-Path $LogDir ("shift_{0:D3}_{1}.log" -f $shift, (Get-Date -Format "MMdd_HHmm"))
    $started = Get-Date
    Write-Host "== SHIFT $shift launching $(Get-Date -Format HH:mm) (log: $log) =="

    if ($Test) {
        $prompt = "NIGHT-SHIFT PLUMBING TEST (attended): read NEXT_SESSION.md, confirm you can see " +
                  "the frontier (name its first objective in one sentence), append exactly one line " +
                  "'- TEST shift: plumbing OK, frontier visible' to NIGHT_REPORT.md, and exit. Do NOT " +
                  "launch any emulator runs or start the actual work -- this is only a relauncher test."
    } else {
        $preface = "NIGHT SHIFT #$shift (unattended -- the night_shift.ps1 loop; CLAUDE.md employment " +
                   "terms in force). NIGHT-SHIFT CONTRACT for THIS session: (1) at close, append ONE " +
                   "line to NIGHT_REPORT.md -- '- shift ${shift} survey: <what banked> | frontier: " +
                   "<next objective> | needs eyes: <blocker or none>'; (2) if THE CREDITS ROLL, write " +
                   "CREDITS as the FIRST line of NIGHT_REPORT.md + the full mountain survey below it " +
                   "(that line stops the loop); (3) keep NEXT_SESSION.md CURRENT CONTINUOUSLY, not " +
                   "only at close -- REWRITE IT BEFORE LAUNCHING ANY LONG STRIKE/RUN and at every " +
                   "bank (frontier-first discipline: a shift that dies at the context wall mid-strike " +
                   "must still leave a TRUE frontier -- shift 8 died mid-Silph-strike and three " +
                   "successors launched off a shift-7 file); the loop feeds it verbatim to your " +
                   "successor, and an unchanged frontier + zero commits across two shifts fires the " +
                   "loop's brake; commit every real fix (commits are your proof of life).`n`n"
        $prompt = $preface + (Get-Content $Frontier -Raw)
    }

    & claude --dangerously-skip-permissions -p $prompt *> $log
    $mins = [int]((Get-Date) - $started).TotalMinutes

    $endHead = (git rev-parse HEAD 2>$null)
    $endFrontier = Get-FrontierHash
    $commits = 0
    if ($startHead -and $endHead -and ($startHead -ne $endHead)) {
        $commits = [int](git rev-list --count "$startHead..$endHead" 2>$null)
    }
    Add-Content $Report ("- shift {0} {1}->{2} ({3}m): {4} commit(s), frontier {5}" -f `
        $shift, $started.ToString("HH:mm"), (Get-Date -Format "HH:mm"), $mins, $commits, `
        $(if ($endFrontier -ne $startFrontier) { "ADVANCED" } else { "unchanged" }))

    # (c) THE ONE BRAKE -- zero-progress detector
    if (($endFrontier -eq $startFrontier) -and ($commits -eq 0)) {
        $noProgress++
    } else {
        $noProgress = 0
    }
    if ($noProgress -ge 2) {
        Add-Content $Report ("- BRAKE $(Get-Date -Format 'HH:mm'): two consecutive shifts of provable " +
            "nothing (identical frontier, zero commits). The wall needs human eyes -- see STATE section 0 " +
            "CURRENT TRUTH + the last shift log: $log")
        Write-Host "== BRAKE FIRED -- two shifts of provable nothing. Stopping. =="
        break
    }

    if ($Test) {
        Write-Host "== TEST cycle complete -- check NIGHT_REPORT.md for the session's test line. =="
        break
    }
    Write-Host "== SHIFT $shift closed ($mins m, $commits commits). Relaunching in $SleepBetween s. =="
    Start-Sleep -Seconds $SleepBetween
}
