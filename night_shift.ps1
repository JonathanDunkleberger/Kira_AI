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
# PS 5.1 encodes pipeline input to native exes with $OutputEncoding (default ASCII) --
# force UTF-8 so the frontier's arrows/symbols survive the stdin pipe to claude.
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
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
        # The test prompt CARRIES the real frontier content below it -- the whole point is proving
        # that arbitrary frontier text (quotes, -u flags, newlines) survives the launch path.
        $prompt = "NIGHT-SHIFT PLUMBING TEST (attended): the REAL NEXT_SESSION.md content is appended " +
                  "below ONLY to prove prompt-passing survives its special characters -- do NOT act on " +
                  "it, do NOT launch any emulator runs. Just: (1) confirm you received it by naming its " +
                  "first objective in one sentence, (2) append exactly one line " +
                  "'- TEST shift: plumbing OK, frontier visible' to NIGHT_REPORT.md, (3) exit.`n`n" +
                  "--- FRONTIER CONTENT (do not act on) ---`n" + (Get-Content $Frontier -Raw -Encoding UTF8)
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
        # -Encoding UTF8: PS 5.1 reads BOM-less UTF-8 as ANSI, mojibaking the frontier's arrows.
        $prompt = $preface + (Get-Content $Frontier -Raw -Encoding UTF8)
    }

    # Prompt goes via STDIN, never argv: PS 5.1 quotes-but-never-escapes native args, so any
    # embedded '"' in NEXT_SESSION.md re-tokenizes the tail and claude parses fragments like
    # '-u' as CLI options ("error: unknown option '-u'" -> instant 0-minute shifts). Piped
    # stdin has no argument-parsing surface at all.
    $prompt | & claude --dangerously-skip-permissions -p *> $log
    $exitCode = $LASTEXITCODE
    $secs = [int]((Get-Date) - $started).TotalSeconds
    $mins = [int]($secs / 60)

    # LAUNCH SELF-TEST: a shift that dies in under 60s never did any work -- surface the real
    # error LOUDLY (console + report) instead of a silent 0-minute close feeding the brake.
    $logBytes = 0
    if (Test-Path $log) { $logBytes = (Get-Item $log).Length }

    # BILLING-EXHAUSTION -> CLEAN TERMINAL STOP (2026-07-08). Auto-recharge is OFF, so the natural
    # ceiling is the loaded balance. When API calls fail for insufficient credits / billing, relaunching
    # just thrashes against a dead wallet -- so detect the billing signature, write a final standup, and
    # exit cleanly. Signatures are the API's own billing-error phrasing (essentially zero false-positive
    # on a Pokemon nav task -- no billing surface). This is ONLY for balance exhaustion; genuine transient
    # errors and game crash-loops keep escalate-don't-quit (the supervisor + the 2-shift brake handle those).
    $billingRe = 'credit balance is too low|purchase more credits|insufficient[_ ]?(quota|credits?|funds)|402 payment required'
    $logText = ''
    if ($logBytes -gt 0) { $logText = (Get-Content $log -Raw) }
    if ($logText -match $billingRe) {
        $hit = $matches[0]
        Write-Host "== SHIFT $shift STOP: API BALANCE EXHAUSTED (matched '$hit'). Halting cleanly -- top up + relaunch. ==" -ForegroundColor Yellow
        Add-Content $Report ("- shift {0} STOPPED {1} ({2}s, exit {3}): balance exhausted (billing/credit failure: '{4}'). Night train halted CLEANLY -- not a crash. Top up the balance and relaunch night_shift.ps1." -f `
            $shift, (Get-Date -Format "HH:mm"), $secs, $exitCode, $hit)
        break
    }

    if (($secs -lt 60) -and (($exitCode -ne 0) -or ($logBytes -eq 0))) {
        $tail = "(log empty -- claude produced no output at all)"
        if ($logBytes -gt 0) { $tail = ((Get-Content $log -Tail 10) -join "`n") }
        Write-Host "== SHIFT $shift FAST-FAIL: exited in ${secs}s (exit code $exitCode, log $logBytes bytes) ==" -ForegroundColor Red
        Write-Host $tail -ForegroundColor Red
        Add-Content $Report ("- shift {0} FAST-FAIL {1} ({2}s, exit {3}): {4}" -f `
            $shift, (Get-Date -Format "HH:mm"), $secs, $exitCode, ($tail -replace "\r?\n", " / "))
    }

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
