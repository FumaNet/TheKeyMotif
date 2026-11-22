Set-Location $PSScriptRoot
$py = if ($env:KM_PYTHON) { $env:KM_PYTHON } else { "python" }

New-Item -ItemType Directory -Force -Path results\nodedup_backup | Out-Null
foreach ($f in 'id80','id70','id60','id50','id40') {
    $p = "results\2d_phage_holdout_$f.pkl"
    if (Test-Path $p) { Move-Item $p results\nodedup_backup\ -Force }
}

$thresholds = '0.95','0.90','0.80','0.70','0.60','0.50','0.40'
foreach ($t in $thresholds) {
    $tag = $t.Replace('0.','')
    "=== id$tag starting $(Get-Date -Format 'HH:mm:ss') ===" |
        Tee-Object -Append results\sweep_dedup_progress.txt
    & $py analysis\02_ablation\2d_phage_holdout.py --cluster-identity $t --dedup 2>&1 |
        Out-File -FilePath "results\sweep_dedup_id$tag.log" -Encoding utf8
    "=== id$tag done $(Get-Date -Format 'HH:mm:ss') ===" |
        Tee-Object -Append results\sweep_dedup_progress.txt
}
"ALL DONE $(Get-Date -Format 'HH:mm:ss')" |
    Tee-Object -Append results\sweep_dedup_progress.txt
