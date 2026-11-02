$ErrorActionPreference = 'Stop'
$packageRoot = $PSScriptRoot
$manifestPath = Join-Path $packageRoot '文件校验清单.csv'
$manifest = Import-Csv -LiteralPath $manifestPath -Encoding utf8
$failures = @(
    foreach ($entry in $manifest) {
        $filePath = Join-Path $packageRoot $entry.path
        if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) {
            "Missing $($entry.path)"
            continue
        }
        $actualHash = (Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash
        if ($actualHash -ne $entry.sha256) {
            "Changed $($entry.path)"
        }
    }
)
if ($failures.Count -gt 0) {
    $failures | Write-Output
    throw "$($failures.Count) file integrity checks failed."
}
Write-Output "Verified $($manifest.Count) source file copies."
