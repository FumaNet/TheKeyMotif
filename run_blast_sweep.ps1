Set-Location $PSScriptRoot
$py = if ($env:KM_PYTHON) { $env:KM_PYTHON } else { "python" }

$levels = '90','80','70','60','50'
foreach ($t in $levels) {
    "=== blast$t starting $(Get-Date -Format 'HH:mm:ss') ===" |
        Tee-Object -Append results\blast_sweep_progress.txt
    & $py analysis\02_ablation\2d_phage_holdout.py `
        --groups-file results\groupings\groups_blast$t.csv --dedup 2>&1 |
        Out-File -FilePath "results\blast_sweep_id$t.log" -Encoding utf8
    "=== blast$t done $(Get-Date -Format 'HH:mm:ss') ===" |
        Tee-Object -Append results\blast_sweep_progress.txt
}
"ALL DONE $(Get-Date -Format 'HH:mm:ss')" |
    Tee-Object -Append results\blast_sweep_progress.txt