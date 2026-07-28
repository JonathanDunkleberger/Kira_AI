# night_shift.ps1 -- THE NIGHT-SHIFT RELAUNCHER, CURSOR-AGENT EDITION (marathon-stream era, 2026-07-28)
#
# Ported from the Claude Code CLI (banned) to the Cursor CLI (`agent`) per docs/marathon-stream-plan.md
# "How We Work Now". Same loop: read frontier doc -> work -> write report -> relaunch. Brakes RESTORED
# (the run-until-credits override retired with the summit -- the marathon era keeps the brake logic).
#
# Loops autonomous Cursor agent sessions off NEXT_SESSION.md until ONE of exactly four things:
#   (a) THE WIN SENTINEL  -- a session wrote "CREDITS" at the top of NIGHT_REPORT.md; loop stops.
#   (b) Jonny stops it    -- Ctrl+C this window (or close it).
#   (c) THE ONE BRAKE     -- two CONSECUTIVE shifts with an IDENTICAL NEXT_SESSION.md frontier AND
#                            zero new commits (provable nothing twice). Stuck-but-committing
#                            CONTINUES; slow-but-moving CONTINUES; a fourth angle with new code
#                            CONTINUES. Only the same-wall-for-hours-producing-nothing machine stops.
#   (d) HALT / NEEDS-JONNY sentinel on line 1 of NIGHT_REPORT.md -- stop for human eyes.
#
# Sessions run with `agent -p --force` (the --dangerously-skip-permissions equivalent; Jonny-approved
# for the night train 2026-07-06): fully autonomous, no per-action gates. Safety = the sanctity gate
# on every canonical promotion + git-revertibility of everything else. The CLI auto-loads AGENTS.md
# and CLAUDE.md at repo root as rules, so the operating doctrine rides every shift for free.
#
# PREREQS (one-time, human): install the CLI  ->  irm 'https://cursor.com/install?win32=true' | iex
#                            authenticate     ->  agent login   (or set CURSOR_API_KEY)
#
# BEDTIME:   powershell -ExecutionPolicy Bypass -File .\night_shift.ps1
# TEST:      powershell -ExecutionPolicy Bypass -File .\night_shift.ps1 -Test   (one attended cycle,
#            harmless prompt -- proves launch/report/brake plumbing without driving the emulator)
param(
    [int]$SleepBetween = 60,
    [switch]$Test,
    [string]$Model = ""        # optional --model override; default = the account's default model
)
$ErrorActionPreference = "Continue"
# Brakes ON (marathon era): the 2-shift no-progress brake and the billing-exhaustion stop are LIVE.
# (The old RUN-UNTIL-CREDITS override from the 2026-07-09 completion push is retired -- the summit
# happened. Flip to $true only for another deliberate completion push.)
$RunUntilCredits = $false
# PS 5.1 encodes pipeline input to native exes with $OutputEncoding (default ASCII) --
# force UTF-8 so the frontier's arrows/symbols survive the stdin pipe to the agent CLI.
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Report = Join-Path $Root "NIGHT_REPORT.md"
$Frontier = Join-Path $Root "NEXT_SESSION.md"
$LogDir = Join-Path $Root "logs\nightshift"
New-Item -ItemType Directory -Force $LogDir | Out-Null

# ── CLI PREFLIGHT (fail LOUD at bedtime, not 50 silent fast-fails at 3am) ──────────────────────────
# Resolve the agent CLI: PATH first, then the installer's known landing spot (a fresh service/console
# session may not have re-read the user PATH the installer appended to).
$AgentCmd = $null
$cmd = Get-Command agent -ErrorAction SilentlyContinue
if ($cmd) { $AgentCmd = $cmd.Source }
if (-not $AgentCmd) {
    $fallback = Join-Path $env:LOCALAPPDATA "cursor-agent\agent.cmd"
    if (Test-Path $fallback) { $AgentCmd = $fallback }
}
if (-not $AgentCmd) {
    Write-Host "== NIGHT TRAIN CANNOT START: the Cursor CLI ('agent') is not installed. ==" -ForegroundColor Red
    Write-Host "   Install:  irm 'https://cursor.com/install?win32=true' | iex" -ForegroundColor Red
    exit 1
}
# Auth check: an unauthenticated CLI produces nothing but fast-fails all night. `agent status`
# prints "Not logged in" (and CURSOR_API_KEY also satisfies it when set).
$authOut = (& $AgentCmd status 2>&1 | Out-String)
if ($authOut -match 'Not logged in' -and -not $env:CURSOR_API_KEY) {
    Write-Host "== NIGHT TRAIN CANNOT START: Cursor CLI is not authenticated. ==" -ForegroundColor Red
    Write-Host "   Run 'agent login' once (browser flow), or set CURSOR_API_KEY. ==" -ForegroundColor Red
    exit 1
}
Write-Host "== agent CLI: $AgentCmd  (auth OK) =="

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
    # (a) WIN sentinel -- the deliverable. LINE 1 ONLY, anchored: the header text below must never
    # contain the token itself (the first attended test false-fired on its own documentation line).
    $top = (Get-Content $Report -TotalCount 1)
    if ($top -match "^\s*CREDITS") {
        Write-Host "== THE WIN SENTINEL. The mountain is climbed. Read NIGHT_REPORT.md. =="
        break
    }
    # HALT / NEEDS-JONNY sentinels: a session that hits a same-root-twice failure, an idle deadline,
    # or anything needing human hands writes one of these as line 1 of NIGHT_REPORT.md with a
    # one-page diagnosis below it. Both stop the loop for human eyes.
    if ($top -match "^\s*(HALT|NEEDS-JONNY)") {
        Write-Host "== '$($matches[1])' sentinel found on line 1 of NIGHT_REPORT.md. Stopping for human eyes. =="
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
        $preface = "NIGHT SHIFT #$shift (unattended -- the night_shift.ps1 loop; CLAUDE.md operating " +
                   "rules + AGENTS.md non-negotiables in force). NIGHT-SHIFT CONTRACT for THIS session: " +
                   "(1) at close, append ONE line to NIGHT_REPORT.md -- '- shift ${shift} survey: " +
                   "<what banked> | frontier: <next objective> | needs eyes: <blocker or none>'; " +
                   "(2) if the mission's WIN CONDITION lands, write CREDITS as the FIRST line of " +
                   "NIGHT_REPORT.md + the full survey below it (that line stops the loop); if you hit " +
                   "a wall only Jonny can clear, write NEEDS-JONNY as the FIRST line + a one-page " +
                   "diagnosis (that also stops the loop); (3) keep NEXT_SESSION.md CURRENT " +
                   "CONTINUOUSLY, not only at close -- REWRITE IT BEFORE LAUNCHING ANY LONG STRIKE/RUN " +
                   "and at every bank (frontier-first discipline: a shift that dies at the context " +
                   "wall mid-strike must still leave a TRUE frontier); the loop feeds it verbatim to " +
                   "your successor, and an unchanged frontier + zero commits across two shifts fires " +
                   "the loop's brake; commit every real fix (commits are your proof of life).`n`n"
        # -Encoding UTF8: PS 5.1 reads BOM-less UTF-8 as ANSI, mojibaking the frontier's arrows.
        $prompt = $preface + (Get-Content $Frontier -Raw -Encoding UTF8)
    }

    # Prompt goes via STDIN, never argv: PS 5.1 quotes-but-never-escapes native args, so any
    # embedded '"' in NEXT_SESSION.md re-tokenizes the tail and the CLI parses fragments like
    # '-u' as CLI options ("error: unknown option '-u'" -> instant 0-minute shifts). Piped
    # stdin has no argument-parsing surface at all. `agent` infers print mode from piped stdin;
    # -p + --force + --trust make it explicit and fully autonomous (headless, no permission gates).
    $agentArgs = @("-p", "--force", "--trust", "--output-format", "text")
    if ($Model) { $agentArgs += @("--model", $Model) }
    $prompt | & $AgentCmd @agentArgs *> $log
    $exitCode = $LASTEXITCODE
    $secs = [int]((Get-Date) - $started).TotalSeconds
    $mins = [int]($secs / 60)

    # LAUNCH SELF-TEST: a shift that dies in under 60s never did any work -- surface the real
    # error LOUDLY (console + report) instead of a silent 0-minute close feeding the brake.
    $logBytes = 0
    if (Test-Path $log) { $logBytes = (Get-Item $log).Length }

    # BILLING/USAGE-EXHAUSTION -> CLEAN TERMINAL STOP. When agent calls fail for exhausted usage/
    # billing, relaunching just thrashes against a dead wallet -- detect the billing signature,
    # write a final standup, and exit cleanly. Signatures cover both the old API phrasing and
    # Cursor's usage-limit phrasing (essentially zero false-positive surface on a dev-loop task).
    # ONLY for balance exhaustion; genuine transient errors keep the retry path (the brake handles
    # real walls).
    $billingRe = 'credit balance is too low|purchase more credits|insufficient[_ ]?(quota|credits?|funds)|402 payment required|usage limits?( reached| exceeded)?|out of credits|upgrade your plan'
    $logText = ''
    if ($logBytes -gt 0) { $logText = (Get-Content $log -Raw) }
    if ((-not $RunUntilCredits) -and ($logText -match $billingRe)) {
        $hit = $matches[0]
        Write-Host "== SHIFT $shift STOP: USAGE/BILLING EXHAUSTED (matched '$hit'). Halting cleanly -- top up + relaunch. ==" -ForegroundColor Yellow
        Add-Content $Report ("- shift {0} STOPPED {1} ({2}s, exit {3}): usage/billing exhausted ('{4}'). Night train halted CLEANLY -- not a crash. Restore usage and relaunch night_shift.ps1." -f `
            $shift, (Get-Date -Format "HH:mm"), $secs, $exitCode, $hit)
        break
    }

    # FAST-FAIL routing: a sub-60s error shift is NOISE, not a survey line -- it must NOT pollute
    # NIGHT_REPORT.md (which the loop reads for the sentinels and Jonny reads for real progress).
    # Route fast-fails to logs/fastfail.log instead.
    $isFastFail = (($secs -lt 60) -and (($exitCode -ne 0) -or ($logBytes -eq 0)))
    if ($isFastFail) {
        $tail = "(log empty -- the agent CLI produced no output at all)"
        if ($logBytes -gt 0) { $tail = ((Get-Content $log -Tail 10) -join "`n") }
        Write-Host "== SHIFT $shift FAST-FAIL: exited in ${secs}s (exit code $exitCode, log $logBytes bytes) ==" -ForegroundColor Red
        Write-Host $tail -ForegroundColor Red
        $FastFail = Join-Path $LogDir "..\fastfail.log"
        Add-Content $FastFail ("- shift {0} FAST-FAIL {1} ({2}s, exit {3}): {4}" -f `
            $shift, (Get-Date -Format "yyyy-MM-dd HH:mm"), $secs, $exitCode, ($tail -replace "\r?\n", " / "))
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
    if ((-not $RunUntilCredits) -and ($noProgress -ge 2)) {
        Add-Content $Report ("- BRAKE $(Get-Date -Format 'HH:mm'): two consecutive shifts of provable " +
            "nothing (identical frontier, zero commits). The wall needs human eyes -- see STATE section 0 " +
            "CURRENT TRUTH + the last shift log: $log")
        Write-Host "== BRAKE FIRED -- two shifts of provable nothing. Stopping. =="
        break
    }
    if ($RunUntilCredits -and ($noProgress -ge 2)) {
        Write-Host "== RUN-UNTIL-CREDITS: $noProgress shift(s) without frontier/commit progress -- brake DISABLED, escalating + continuing. =="
    }

    if ($Test) {
        Write-Host "== TEST cycle complete -- check NIGHT_REPORT.md for the session's test line. =="
        break
    }
    # ADAPTIVE CADENCE: a shift that COMMITTED was actively building -> relaunch fast (60s)
    # so the next build starts. A 0-commit shift just glanced at a healthy in-flight run and exited
    # clean -> there's nothing to do until the run advances, so wait 15 MINUTES (don't burn tokens
    # re-glancing a slow-cooking run every minute). A fast-fail retries at 60s (something's wrong,
    # recover quickly). This replaces the flat $SleepBetween.
    if ($isFastFail) {
        $sleepSecs = 60
    } elseif ($commits -gt 0) {
        $sleepSecs = 60
    } else {
        $sleepSecs = 900
    }
    Write-Host "== SHIFT $shift closed ($mins m, $commits commits). Relaunching in $sleepSecs s. =="
    Start-Sleep -Seconds $sleepSecs
}
