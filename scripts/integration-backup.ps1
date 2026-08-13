param(
    [string]$Root = 'E:\AI',
    [string]$BackupRoot = 'E:\AI\state\lexicon-mcp-integration'
)

$ErrorActionPreference = 'Stop'
$resolvedRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\')
$resolvedBackupRoot = [IO.Path]::GetFullPath($BackupRoot).TrimEnd('\')
if (-not $resolvedBackupRoot.StartsWith($resolvedRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "BackupRoot must be inside Root"
}

$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$destination = Join-Path $resolvedBackupRoot $stamp
New-Item -ItemType Directory -Path $destination -Force | Out-Null

$targets = @(
    (Join-Path $resolvedRoot 'mcp\config.json'),
    (Join-Path $resolvedRoot 'scripts\start.ps1'),
    (Join-Path $resolvedRoot 'run\open-webui.env'),
    (Join-Path $resolvedRoot 'open-webui\data\webui.db'),
    (Join-Path $resolvedRoot 'open-webui\data\webui.db-wal'),
    (Join-Path $resolvedRoot 'open-webui\data\webui.db-shm')
)

foreach ($target in $targets) {
    if (Test-Path -LiteralPath $target) {
        Copy-Item -LiteralPath $target -Destination (Join-Path $destination ([IO.Path]::GetFileName($target)))
    }
}

$manifest = foreach ($file in Get-ChildItem -File -LiteralPath $destination) {
    [pscustomobject]@{
        name = $file.Name
        bytes = $file.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
    }
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $destination 'backup-manifest.json') -Encoding utf8
Write-Output $destination

