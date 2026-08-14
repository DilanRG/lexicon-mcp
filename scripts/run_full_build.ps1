[CmdletBinding()]
param(
    [string]$Project = 'E:\AI\lexicon-mcp',
    [string]$Sources = 'E:\AI\state\lexicon-mcp-build\sources',
    [string]$BuildState = 'E:\AI\state\lexicon-mcp-build',
    [string]$Output = 'E:\AI\state\lexicon-mcp-build\built\data-v1.0.0',
    [string]$DatasetVersion = 'data-v1.0.0',
    [switch]$RecoverPartial,
    [string]$OriginalBuildCommit = '',
    [string]$RecoveryCommit = '',
    [string]$ExpectedLexiconSha256 = '79a87f063bce33ae1f1ccc68f49c45762a57c9377c7eb4f14181a4305ab9dd3b',
    [string]$ExpectedGlobalIndexSha256 = 'b6596a46e24d86ae6c04b4a48e19adda96faf9f8bff1b151aa4f395c0938f0c4',
    [string]$ExpectedGlobalVectorsSha256 = '211aeebc9c53b1c79156beca25ca9e2622c4a13229f55293c0801185f13b73bb',
    [switch]$MonitoringArithmeticSelfTest
)

$ErrorActionPreference = 'Stop'

function Get-Int64MetricTotal {
    param(
        [object[]]$Processes,
        [string]$PropertyName
    )

    [long]$total = 0
    foreach ($trackedProcess in $Processes) {
        if ($null -ne $trackedProcess) {
            [long]$value = [long]$trackedProcess.$PropertyName
            $total = [long]($total + $value)
        }
    }
    return [long]$total
}

function Get-Int64Minimum {
    param(
        [long]$Left,
        [long]$Right
    )

    return [Math]::Min([long]$Left, [long]$Right)
}

function Get-Int64Maximum {
    param(
        [long]$Left,
        [long]$Right
    )

    return [Math]::Max([long]$Left, [long]$Right)
}

function Set-ChildProcessEnvironmentVariable {
    param(
        [System.Diagnostics.ProcessStartInfo]$StartInfo,
        [string]$Name,
        [string]$Value
    )

    # ProcessStartInfo.Environment is available on modern .NET/PowerShell,
    # while Windows PowerShell 5.1 exposes EnvironmentVariables instead.
    # Probe the actual property rather than indexing a null compatibility shim.
    $environmentProperty = $StartInfo.PSObject.Properties['Environment']
    if ($null -ne $environmentProperty -and $null -ne $environmentProperty.Value) {
        $environmentProperty.Value[$Name] = $Value
        return
    }
    $legacyProperty = $StartInfo.PSObject.Properties['EnvironmentVariables']
    if ($null -ne $legacyProperty -and $null -ne $legacyProperty.Value) {
        $legacyProperty.Value[$Name] = $Value
        return
    }
    throw 'ProcessStartInfo exposes no writable child environment dictionary.'
}

if ($MonitoringArithmeticSelfTest) {
    $syntheticProcesses = @(
        [pscustomobject]@{
            PrivateMemorySize64 = [long]2147483648
            WorkingSet64 = [long]4294967296
        },
        [pscustomobject]@{
            PrivateMemorySize64 = [long]4096
            WorkingSet64 = [long]8192
        }
    )
    [long]$privateBytes = Get-Int64MetricTotal $syntheticProcesses 'PrivateMemorySize64'
    [long]$workingSetBytes = Get-Int64MetricTotal $syntheticProcesses 'WorkingSet64'
    [long]$peakPrivateBytes = Get-Int64Maximum ([long]0) $privateBytes
    [long]$peakWorkingSetBytes = Get-Int64Maximum ([long]0) $workingSetBytes
    [long]$minimumFree = Get-Int64Minimum ([long]12884901888) ([long]8589934592)
    $selfTestStartInfo = [System.Diagnostics.ProcessStartInfo]::new()
    Set-ChildProcessEnvironmentVariable $selfTestStartInfo 'OPENBLAS_NUM_THREADS' '1'
    $selfTestEnvironment = $selfTestStartInfo.PSObject.Properties['Environment']
    if ($null -eq $selfTestEnvironment -or $null -eq $selfTestEnvironment.Value) {
        $selfTestEnvironment = $selfTestStartInfo.PSObject.Properties['EnvironmentVariables']
    }
    [ordered]@{
        build_started = $false
        private_bytes = $privateBytes
        working_set_bytes = $workingSetBytes
        peak_private_bytes = $peakPrivateBytes
        peak_working_set_bytes = $peakWorkingSetBytes
        minimum_free_bytes = $minimumFree
        openblas_num_threads = [string]$selfTestEnvironment.Value['OPENBLAS_NUM_THREADS']
    } | ConvertTo-Json
    exit 0
}

$projectPath = [System.IO.Path]::GetFullPath($Project)
$sourcePath = [System.IO.Path]::GetFullPath($Sources)
$statePath = [System.IO.Path]::GetFullPath($BuildState)
$outputPath = [System.IO.Path]::GetFullPath($Output)
$operation = if ($RecoverPartial) { 'recovery' } else { 'build' }
if ($RecoverPartial) {
    if ($OriginalBuildCommit -notmatch '^[0-9a-f]{40}$') {
        throw 'Recovery requires -OriginalBuildCommit as lowercase 40-hex.'
    }
    if ($RecoveryCommit -notmatch '^[0-9a-f]{40}$') {
        throw 'Recovery requires -RecoveryCommit as lowercase 40-hex.'
    }
    foreach ($shaAnchor in @(
        $ExpectedLexiconSha256,
        $ExpectedGlobalIndexSha256,
        $ExpectedGlobalVectorsSha256
    )) {
        if ($shaAnchor -notmatch '^[0-9a-f]{64}$') {
            throw 'Recovery SHA-256 anchors must be lowercase 64-hex.'
        }
    }
} else {
    $datasetPartial = $outputPath + '.partial'
    $semanticPartial = Join-Path $datasetPartial 'semantic.partial'
    $completedSemantic = Join-Path $datasetPartial 'semantic'
    if (
        [System.IO.Directory]::Exists($semanticPartial) -or
        [System.IO.Directory]::Exists($completedSemantic)
    ) {
        throw (
            'A post-global semantic build is already staged. Refusing the normal ' +
            'build path; rerun this wrapper with -RecoverPartial and exact commit provenance.'
        )
    }
}
$logDirectory = Join-Path $statePath 'logs'
[System.IO.Directory]::CreateDirectory($logDirectory) | Out-Null

$stamp = Get-Date -Format 'yyyyMMddTHHmmss'
$logPrefix = "full-$operation-$stamp"
$stdoutPath = Join-Path $logDirectory "$logPrefix.stdout.log"
$stderrPath = Join-Path $logDirectory "$logPrefix.stderr.log"
$telemetryPath = Join-Path $logDirectory "$logPrefix.telemetry.jsonl"
$summaryPath = Join-Path $logDirectory "$logPrefix.summary.json"

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
$entrypoint = if ($RecoverPartial) {
    Join-Path $projectPath 'scripts\recover_full_corpus.py'
} else {
    Join-Path $projectPath 'scripts\build_full_corpus.py'
}
$arguments = @(
    $entrypoint,
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
if ($RecoverPartial) {
    $arguments += @(
        '--original-build-commit', $OriginalBuildCommit,
        '--recovery-commit', $RecoveryCommit,
        '--expected-lexicon-sha256', $ExpectedLexiconSha256,
        '--expected-global-index-sha256', $ExpectedGlobalIndexSha256,
        '--expected-global-vectors-sha256', $ExpectedGlobalVectorsSha256
    )
}

$driveRoot = [System.IO.Path]::GetPathRoot($outputPath)
$drive = [System.IO.DriveInfo]::new($driveRoot)
$startedAt = [DateTimeOffset]::Now
[long]$baselineFree = [long]$drive.AvailableFreeSpace
[long]$minimumFree = [long]$baselineFree
[long]$peakPrivateBytes = 0
[long]$peakWorkingSetBytes = 0

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $pythonPath
$startInfo.WorkingDirectory = $projectPath
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
# NumPy's OpenBLAS default can reserve hundreds of MiB per Python process on
# Windows. The corpus pipeline only performs single-vector operations, so a
# large BLAS thread pool wastes memory without accelerating this build.
Set-ChildProcessEnvironmentVariable $startInfo 'OPENBLAS_NUM_THREADS' '1'
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
$monitoringError = $null
$childPreserved = $false

function Add-MonitoringError {
    param([string]$Message)

    if ([string]::IsNullOrWhiteSpace($script:monitoringError)) {
        $script:monitoringError = $Message
    } else {
        $script:monitoringError = $script:monitoringError + [Environment]::NewLine + $Message
    }
}

try {
    while (-not $process.WaitForExit(15000)) {
        $trackedProcesses = @(Get-BuildProcesses)
        foreach ($trackedProcess in $trackedProcesses) {
            $trackedProcess.Refresh()
        }
        [long]$privateBytes = Get-Int64MetricTotal $trackedProcesses 'PrivateMemorySize64'
        [long]$workingSetBytes = Get-Int64MetricTotal $trackedProcesses 'WorkingSet64'
        $drive = [System.IO.DriveInfo]::new($driveRoot)
        [long]$free = [long]$drive.AvailableFreeSpace
        $minimumFree = Get-Int64Minimum $minimumFree $free
        $peakPrivateBytes = Get-Int64Maximum $peakPrivateBytes $privateBytes
        $peakWorkingSetBytes = Get-Int64Maximum $peakWorkingSetBytes $workingSetBytes
        [long]$peakVolumeBytesConsumed = [long]($baselineFree - $minimumFree)
        [pscustomobject]@{
            timestamp = [DateTimeOffset]::Now.ToString('o')
            available_free_bytes = $free
            peak_volume_bytes_consumed = $peakVolumeBytesConsumed
            private_bytes = $privateBytes
            working_set_bytes = $workingSetBytes
            launcher_pid = $process.Id
            worker_pid = $workerProcessId
        } | ConvertTo-Json -Compress | Add-Content -LiteralPath $telemetryPath -Encoding utf8
    }
} catch {
    Add-MonitoringError ("Build monitoring failed; preserving and waiting for the child: " + $_.Exception)
} finally {
    try {
        if (-not $process.HasExited) {
            # A telemetry failure must not orphan or discard an otherwise valid
            # multi-hour build. Keep supervising the launcher until completion.
            $process.WaitForExit()
        }
    } catch {
        Add-MonitoringError ("Could not wait for the build launcher: " + $_.Exception)
    }
    try {
        # The Windows venv launcher can hand work to the base interpreter.
        # Discover it even if monitoring failed before the first sample.
        $null = @(Get-BuildProcesses)
    } catch {
        Add-MonitoringError ("Could not discover the build worker: " + $_.Exception)
    }
    if ($null -ne $workerProcessId) {
        try {
            $workerProcess = [System.Diagnostics.Process]::GetProcessById($workerProcessId)
            if (-not $workerProcess.HasExited) {
                $workerProcess.WaitForExit()
            }
        } catch [System.ArgumentException] {
            # The worker exited between discovery and the final wait.
        } catch {
            Add-MonitoringError ("Could not wait for the build worker: " + $_.Exception)
        }
    }
}

try {
    $trackedProcesses = @(Get-BuildProcesses)
    foreach ($trackedProcess in $trackedProcesses) {
        $trackedProcess.Refresh()
    }
    [long]$privateBytes = Get-Int64MetricTotal $trackedProcesses 'PrivateMemorySize64'
    [long]$workingSetBytes = Get-Int64MetricTotal $trackedProcesses 'WorkingSet64'
    $peakPrivateBytes = Get-Int64Maximum $peakPrivateBytes $privateBytes
    $peakWorkingSetBytes = Get-Int64Maximum $peakWorkingSetBytes $workingSetBytes
} catch [System.InvalidOperationException], [System.ArgumentException] {
    # The final in-loop sample remains authoritative after a fast process exit.
} catch {
    Add-MonitoringError ("Could not collect the final process sample: " + $_.Exception)
}

$launcherStillRunning = $false
try {
    $launcherStillRunning = -not $process.HasExited
} catch {
    $launcherStillRunning = $true
    Add-MonitoringError ("Could not determine launcher state: " + $_.Exception)
}
$workerStillRunning = $false
if ($null -ne $workerProcessId) {
    try {
        $workerStillRunning = -not (
            [System.Diagnostics.Process]::GetProcessById($workerProcessId).HasExited
        )
    } catch [System.ArgumentException] {
        $workerStillRunning = $false
    } catch {
        $workerStillRunning = $true
        Add-MonitoringError ("Could not determine worker state: " + $_.Exception)
    }
}
$childPreserved = $launcherStillRunning -or $workerStillRunning

if (-not $childPreserved) {
    try {
        [System.IO.File]::WriteAllText($stdoutPath, $stdoutTask.Result)
        [System.IO.File]::WriteAllText($stderrPath, $stderrTask.Result)
    } catch {
        Add-MonitoringError ("Could not write redirected build output: " + $_.Exception)
    }
}

try {
    $drive = [System.IO.DriveInfo]::new($driveRoot)
    [long]$finalFree = [long]$drive.AvailableFreeSpace
    $minimumFree = Get-Int64Minimum $minimumFree $finalFree
} catch {
    Add-MonitoringError ("Could not collect the final free-space sample: " + $_.Exception)
}
[long]$peakVolumeBytesConsumed = [long]($baselineFree - $minimumFree)
$processExitCode = $null
if (-not $launcherStillRunning) {
    try {
        $processExitCode = [int]$process.ExitCode
    } catch {
        Add-MonitoringError ("Could not read the build exit code: " + $_.Exception)
    }
}
$summary = [ordered]@{
    operation = $operation
    dataset_version = $DatasetVersion
    output = $outputPath
    started_at = $startedAt.ToString('o')
    finished_at = [DateTimeOffset]::Now.ToString('o')
    exit_code = $processExitCode
    baseline_free_bytes = $baselineFree
    minimum_free_bytes = $minimumFree
    peak_volume_bytes_consumed = $peakVolumeBytesConsumed
    peak_private_bytes = $peakPrivateBytes
    peak_working_set_bytes = $peakWorkingSetBytes
    monitoring_error = $monitoringError
    child_preserved = $childPreserved
    launcher_pid = $process.Id
    worker_pid = $workerProcessId
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
    telemetry_log = $telemetryPath
}
$summaryJson = $summary | ConvertTo-Json
try {
    $summaryJson | Set-Content -LiteralPath $summaryPath -Encoding utf8
} catch {
    [Console]::Error.WriteLine("Could not write build summary ${summaryPath}: $($_.Exception)")
}
[Console]::Out.WriteLine($summaryJson)

if ($childPreserved) {
    [Console]::Error.WriteLine(
        "The monitoring wrapper stopped, but the build child was preserved. " +
        "Launcher PID: $($process.Id); worker PID: $workerProcessId"
    )
    exit 1
}
if ($null -ne $monitoringError) {
    [Console]::Error.WriteLine($monitoringError)
    exit 1
}
if ($processExitCode -ne 0) {
    [Console]::Error.WriteLine(
        "Full-corpus $operation failed with exit code $processExitCode. See $stderrPath"
    )
    exit $processExitCode
}
