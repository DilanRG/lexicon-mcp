[CmdletBinding()]
param(
    [string]$Project = 'E:\AI\lexicon-mcp',
    [string]$Sources = 'E:\AI\state\lexicon-mcp-build\sources',
    [string]$BuildState = 'E:\AI\state\lexicon-mcp-build',
    [string]$Output = 'E:\AI\state\lexicon-mcp-build\built\data-v1.0.0',
    [string]$DatasetVersion = 'data-v1.0.0'
)

$ErrorActionPreference = 'Stop'
$projectPath = [System.IO.Path]::GetFullPath($Project)
$sourcePath = [System.IO.Path]::GetFullPath($Sources)
$statePath = [System.IO.Path]::GetFullPath($BuildState)
$outputPath = [System.IO.Path]::GetFullPath($Output)
$logDirectory = Join-Path $statePath 'logs'
[System.IO.Directory]::CreateDirectory($logDirectory) | Out-Null

$stamp = Get-Date -Format 'yyyyMMddTHHmmss'
$stdoutPath = Join-Path $logDirectory "full-build-$stamp.stdout.log"
$stderrPath = Join-Path $logDirectory "full-build-$stamp.stderr.log"
$telemetryPath = Join-Path $logDirectory "full-build-$stamp.telemetry.jsonl"
$summaryPath = Join-Path $logDirectory "full-build-$stamp.summary.json"

$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
if (-not [System.IO.File]::Exists($pythonPath)) {
    throw "Frozen project environment is missing: $pythonPath"
}
$venvConfiguration = Join-Path $projectPath '.venv\pyvenv.cfg'
$homeLine = Get-Content -LiteralPath $venvConfiguration | Where-Object {
    $_ -match '^home\s*=\s*(.+)$'
} | Select-Object -First 1
if (-not $homeLine) {
    throw "Frozen environment has no base Python home: $venvConfiguration"
}
$basePythonHome = ([regex]::Match($homeLine, '^home\s*=\s*(.+)$')).Groups[1].Value.Trim()
$basePythonPath = [System.IO.Path]::GetFullPath((Join-Path $basePythonHome 'python.exe'))
if (-not [System.IO.File]::Exists($basePythonPath)) {
    throw "Frozen environment base Python is missing: $basePythonPath"
}
$arguments = @(
    (Join-Path $projectPath 'scripts\build_full_corpus.py'),
    '--oewn', (Join-Path $sourcePath 'oewn-2025.xml.gz'),
    '--wiktextract', (Join-Path $sourcePath 'wiktextract-en-2026-08-12.jsonl.gz'),
    '--conceptnet', (Join-Path $sourcePath 'conceptnet-assertions-5.7.0.csv.gz'),
    '--numberbatch', (Join-Path $sourcePath 'numberbatch-19.08.txt.gz'),
    '--cmudict', (Join-Path $sourcePath 'cmudict.dict'),
    '--source-lock', (Join-Path $projectPath 'sources.lock.json'),
    '--notices-dir', $projectPath,
    '--output', $outputPath,
    '--build-state', $statePath,
    '--dataset-version', $DatasetVersion
)

$driveRoot = [System.IO.Path]::GetPathRoot($outputPath)
$drive = [System.IO.DriveInfo]::new($driveRoot)
$startedAt = [DateTimeOffset]::Now
$baselineFree = $drive.AvailableFreeSpace
$minimumFree = $baselineFree
$peakPrivateBytes = 0
$peakWorkingSetBytes = 0

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $pythonPath
$startInfo.WorkingDirectory = $projectPath
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
# Windows PowerShell 5.1 runs on .NET Framework, whose ProcessStartInfo has no
# ArgumentList property. These controlled paths contain no quotes; quote every
# argument so whitespace in an overridden root remains safe.
$quotedArguments = foreach ($argument in $arguments) {
    if ($argument.Contains('"')) {
        throw "Build arguments cannot contain a double quote: $argument"
    }
    $trimmed = $argument.TrimEnd('\')
    $trailingSlashCount = $argument.Length - $trimmed.Length
    $escapedTrailingSlashes = ('\' * ($trailingSlashCount * 2)) -join ''
    '"' + $trimmed + $escapedTrailingSlashes + '"'
}
$startInfo.Arguments = $quotedArguments -join ' '

$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $startInfo
if (-not $process.Start()) {
    throw 'Could not start the full-corpus build process.'
}
$workerProcessId = $null

function Get-BuildProcesses {
    $tracked = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()
    try {
        $tracked.Add([System.Diagnostics.Process]::GetProcessById($process.Id))
    } catch [System.ArgumentException] {
        # The launcher can exit immediately after its worker; final samples
        # remain valid even when neither process is still available.
    }
    if ($null -eq $script:workerProcessId) {
        $candidate = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
            try {
                [System.IO.Path]::GetFullPath($_.Path) -eq $basePythonPath -and
                    $_.StartTime -ge $startedAt.LocalDateTime.AddSeconds(-2) -and
                    $_.Id -ne $process.Id
            } catch {
                $false
            }
        } | Sort-Object StartTime, Id | Select-Object -First 1
        if ($candidate) {
            $script:workerProcessId = $candidate.Id
        }
    }
    if ($null -ne $script:workerProcessId) {
        try {
            $tracked.Add(
                [System.Diagnostics.Process]::GetProcessById($script:workerProcessId)
            )
        } catch [System.ArgumentException] {
            # The worker has exited between samples.
        }
    }
    return $tracked
}

$stdoutTask = $process.StandardOutput.ReadToEndAsync()
$stderrTask = $process.StandardError.ReadToEndAsync()

while (-not $process.WaitForExit(15000)) {
    $trackedProcesses = @(Get-BuildProcesses)
    foreach ($trackedProcess in $trackedProcesses) {
        $trackedProcess.Refresh()
    }
    $privateBytes = ($trackedProcesses | Measure-Object PrivateMemorySize64 -Sum).Sum
    $workingSetBytes = ($trackedProcesses | Measure-Object WorkingSet64 -Sum).Sum
    $drive = [System.IO.DriveInfo]::new($driveRoot)
    $free = $drive.AvailableFreeSpace
    $minimumFree = [Math]::Min($minimumFree, $free)
    $peakPrivateBytes = [Math]::Max($peakPrivateBytes, $privateBytes)
    $peakWorkingSetBytes = [Math]::Max($peakWorkingSetBytes, $workingSetBytes)
    [pscustomobject]@{
        timestamp = [DateTimeOffset]::Now.ToString('o')
        available_free_bytes = $free
        peak_volume_bytes_consumed = $baselineFree - $minimumFree
        private_bytes = $privateBytes
        working_set_bytes = $workingSetBytes
        launcher_pid = $process.Id
        worker_pid = $workerProcessId
    } | ConvertTo-Json -Compress | Add-Content -LiteralPath $telemetryPath -Encoding utf8
}
$process.WaitForExit()
try {
    $trackedProcesses = @(Get-BuildProcesses)
    foreach ($trackedProcess in $trackedProcesses) {
        $trackedProcess.Refresh()
    }
    $privateBytes = ($trackedProcesses | Measure-Object PrivateMemorySize64 -Sum).Sum
    $workingSetBytes = ($trackedProcesses | Measure-Object WorkingSet64 -Sum).Sum
    $peakPrivateBytes = [Math]::Max($peakPrivateBytes, $privateBytes)
    $peakWorkingSetBytes = [Math]::Max($peakWorkingSetBytes, $workingSetBytes)
} catch [System.InvalidOperationException], [System.ArgumentException] {
    # The final in-loop sample remains authoritative after a fast process exit.
}
[System.IO.File]::WriteAllText($stdoutPath, $stdoutTask.Result)
[System.IO.File]::WriteAllText($stderrPath, $stderrTask.Result)

$drive = [System.IO.DriveInfo]::new($driveRoot)
$minimumFree = [Math]::Min($minimumFree, $drive.AvailableFreeSpace)
$summary = [ordered]@{
    dataset_version = $DatasetVersion
    output = $outputPath
    started_at = $startedAt.ToString('o')
    finished_at = [DateTimeOffset]::Now.ToString('o')
    exit_code = $process.ExitCode
    baseline_free_bytes = $baselineFree
    minimum_free_bytes = $minimumFree
    peak_volume_bytes_consumed = $baselineFree - $minimumFree
    peak_private_bytes = $peakPrivateBytes
    peak_working_set_bytes = $peakWorkingSetBytes
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
    telemetry_log = $telemetryPath
}
$summary | ConvertTo-Json | Set-Content -LiteralPath $summaryPath -Encoding utf8
$summary | ConvertTo-Json

if ($process.ExitCode -ne 0) {
    Write-Error "Full-corpus build failed with exit code $($process.ExitCode). See $stderrPath"
    exit $process.ExitCode
}
