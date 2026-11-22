# Overnight orchestrator for the remaining identity-sweep thresholds.
#
# Per threshold:
#   1. skip if the .pkl already exists (resume)
#   2. wait until free RAM >= MinFreeGB (the run peaks near 7 GB)
#   3. run, capturing stdout to a private temp file first
#   4. accept the run iff a .pkl was written; copy the log into results\
#   5. retry on genuine failure, up to Attempts times
# Finally: collate every completed threshold into results\sweep_summary.md
#
# FILE-WRITE RESILIENCE (learned the hard way): on this machine writes into
# results\ intermittently fail with Access denied -- ACLs are clean and the
# same file accepts an exclusive handle seconds later, so the cause is almost
# certainly real-time AV holding a transient handle. Three consequences:
#   - every status write retries and can never affect control flow
#   - python's output goes to $env:TEMP first, so a denied log write can never
#     stop the run from happening
#   - a .pkl is NEVER deleted on a failed verification. The earlier version
#     derived the fold count from the log and deleted the .pkl when that read
#     failed, so one AV hiccup could destroy a 45-minute result. Verification
#     now only annotates; sweep_collate.py independently re-derives the fold
#     count from the grouping and recomputes ROC from the .pkl itself.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File sweep_overnight.ps1

param(
    [string[]]$Thresholds   = @("1.00","0.95","0.90","0.80"),
    [double]  $MinFreeGB    = 8.0,
    [int]     $MaxWaitMin   = 240,
    [int]     $Attempts     = 3
)

$ErrorActionPreference = "Continue"
$repo = $PSScriptRoot
Set-Location $repo

$py       = if ($env:KM_PYTHON) { $env:KM_PYTHON } else { "python" }
$status   = "$repo\results\sweep_status.txt"
$fallback = "$env:TEMP\km_sweep_status.txt"

$expectedFolds = @{ "1.00"=99; "0.95"=45; "0.90"=38; "0.80"=29;
                    "0.70"=25; "0.60"=24; "0.50"=21; "0.40"=19 }

# Status logging must never throw and never alter control flow.
function Say($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    for ($i = 0; $i -lt 4; $i++) {
        try { Add-Content -Path $status -Value $line -Encoding utf8 -ErrorAction Stop; return }
        catch { Start-Sleep -Milliseconds 700 }
    }
    try { Add-Content -Path $fallback -Value $line -Encoding utf8 -ErrorAction Stop } catch { }
}

function FreeGB {
    [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB, 2)
}

# Copy with retry; returns $true on success. Never fatal.
function CopyWithRetry($from, $to) {
    for ($i = 0; $i -lt 5; $i++) {
        try { Copy-Item -Path $from -Destination $to -Force -ErrorAction Stop; return $true }
        catch { Start-Sleep -Seconds 2 }
    }
    return $false
}

if (-not (Test-Path "$repo\results")) { New-Item -ItemType Directory -Path "$repo\results" | Out-Null }
Say "=== overnight driver start (pid $PID) ==="
Say "thresholds: $($Thresholds -join ' ')   MinFreeGB=$MinFreeGB  Attempts=$Attempts"

Remove-Item Env:\KM_THRESHOLDS -ErrorAction SilentlyContinue

& $py -c "from rapidfuzz.distance import Levenshtein" 2>$null
if ($LASTEXITCODE -ne 0) { Say "ABORT: rapidfuzz did not import in $py"; exit 1 }
Say "rapidfuzz OK"

$completed = @()
$missing   = @()

foreach ($t in $Thresholds) {
    $tag = [int]([double]$t * 100)
    $pkl = "results\2d_phage_holdout_id$tag.pkl"
    $log = "results\sweep_log_id$tag.txt"

    if (Test-Path $pkl) { Say "SKIP  id$tag  (.pkl already present)"; $completed += $tag; continue }

    $done = $false
    for ($a = 1; $a -le $Attempts -and -not $done; $a++) {

        # --- memory gate -------------------------------------------------
        $waited = 0
        while ((FreeGB) -lt $MinFreeGB -and $waited -lt $MaxWaitMin) {
            if ($waited % 15 -eq 0) { Say "WAIT  id$tag  free $(FreeGB)GB < ${MinFreeGB}GB (${waited}m)" }
            Start-Sleep -Seconds 60
            $waited++
        }
        if ((FreeGB) -lt $MinFreeGB) {
            Say "GIVEUP id$tag  only $(FreeGB)GB free after ${MaxWaitMin}m"
            break
        }

        # --- run (output to temp, so a denied write cannot block the run) --
        $tmp = Join-Path $env:TEMP "km_sweep_id${tag}_a${a}.txt"
        Say "START id$tag  attempt $a/$Attempts  free $(FreeGB)GB"
        $t0 = Get-Date

        & $py analysis\02_ablation\2d_phage_holdout.py --cluster-identity $t *>&1 |
            Out-File -FilePath $tmp -Encoding utf8
        $code = $LASTEXITCODE
        $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)

        # --- accept on .pkl existence, nothing else ------------------------
        if (-not (Test-Path $pkl)) {
            Say "FAIL  id$tag  attempt $a  exit=$code  ${mins}m  no .pkl written"
            $tail = try { (Get-Content $tmp -Tail 3 -ErrorAction Stop) -join " | " } catch { "(temp log unreadable)" }
            Say "      tail: $tail"
            continue
        }

        $done = $true
        $completed += $tag
        if (-not (CopyWithRetry $tmp $log)) { Say "WARN  id$tag  could not copy log into results\; kept at $tmp" }

        # --- annotate only; never destructive ------------------------------
        $folds = -1
        try {
            $fm = Select-String -Path $tmp -Pattern "folds:\s*(\d+)" -ErrorAction Stop | Select-Object -First 1
            if ($fm) { $folds = [int]$fm.Matches.Groups[1].Value }
        } catch { }
        $auc = "(unread)"
        try {
            $am = Select-String -Path $tmp -Pattern "PHAGE-HELD-OUT AUC:.*" -ErrorAction Stop | Select-Object -First 1
            if ($am) { $auc = $am.Matches.Value }
        } catch { }

        if ($folds -eq -1) {
            Say "OK    id$tag  ${mins}m  $auc   (fold count UNVERIFIED from log; collation will check)"
        } elseif ($folds -ne $expectedFolds[$t]) {
            Say "OK*   id$tag  ${mins}m  $auc   WARNING folds=$folds expected=$($expectedFolds[$t]) -- .pkl KEPT, check collation"
        } else {
            Say "OK    id$tag  folds=$folds  ${mins}m  $auc"
        }
    }
    if (-not $done) { Say "ABANDON id$tag after $Attempts attempt(s)"; $missing += $tag }
}

Say "completed this run: $($completed -join ' ')"
if ($missing.Count) { Say "STILL MISSING: $($missing -join ' ')  <-- rerun the driver to retry these" }

Say "collating"
& $py -I "$repo\sweep_collate.py" *>&1 | Out-File -FilePath "$env:TEMP\km_collate_log.txt" -Encoding utf8
$cc = $LASTEXITCODE
CopyWithRetry "$env:TEMP\km_collate_log.txt" "results\sweep_collate_log.txt" | Out-Null
if ($cc -eq 0) { Say "collation OK -> results\sweep_summary.md" }
else { Say "collation FAILED exit=$cc (see results\sweep_collate_log.txt)" }

Say "=== overnight driver done ==="
