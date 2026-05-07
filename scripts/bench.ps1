<#
.SYNOPSIS
    Benchmark the music-separator GPU docker image against tests/raw_vid.mp4.

.DESCRIPTION
    Runs the prebuilt `music-separator:gpu` image (Dockerfile.gpu) repeatedly
    and reports wall-clock statistics. Each run is `docker run --rm --gpus all`
    with the test clip bind-mounted at /inputs and the repo's `out/` mounted at
    /out (mirroring the docker-compose setup).

    To benchmark a different `tuning` block, point `-Config` at any host
    `config.json`; the script bind-mounts it over `/app/config.json` for the
    duration of each run, leaving the image untouched.

    Per-run stderr+stdout is captured to a log file under <repo>/out/bench/
    so you can inspect step-level timings printed by the pipeline.

.PARAMETER Iterations
    Timed runs to keep (default 3).

.PARAMETER WarmupIterations
    Runs to discard before timing (default 1). The first run typically pays for
    cuDNN exhaustive search and OS file-cache fill.

.PARAMETER Image
    Image tag to run (default "music-separator:gpu", matching docker-compose).

.PARAMETER Input
    Host path to the input video. Default: <repo>/tests/raw_vid.mp4.

.PARAMETER Config
    Optional host path to a config.json that should override the image's baked
    /app/config.json. Useful for A/B testing different `tuning` blocks.

.PARAMETER LogDir
    Where per-run captured logs go. Default: <repo>/out/bench.

.PARAMETER Build
    Run `docker build -f Dockerfile.gpu -t <Image> .` before benchmarking.

.PARAMETER NoGpu
    Drop the `--gpus all` flag (useful for sanity-checking the harness on a
    machine without an NVIDIA GPU; the run will still execute on CUDA EP if the
    config asks for it, and most likely fail at session-init time).

.EXAMPLE
    pwsh ./scripts/bench.ps1
    Three timed runs, one warmup, against the baked-in config.cuda.json.

.EXAMPLE
    pwsh ./scripts/bench.ps1 -Build -Iterations 5
    Build the image first, then run 5 timed iterations.

.EXAMPLE
    pwsh ./scripts/bench.ps1 -Config ./bench-configs/tf32-on.json
    Compare a host-side config (with `tuning.cuda.tf32 = true`, say) against
    the baseline by re-running with different -Config files.
#>
[CmdletBinding()]
param(
    [int]$Iterations = 3,
    [int]$WarmupIterations = 1,
    [string]$Image = "music-separator:gpu",
    [string]$Input,
    [string]$Config,
    [string]$LogDir,
    [switch]$Build,
    [switch]$NoGpu
)

$ErrorActionPreference = "Stop"

# ── Paths ────────────────────────────────────────────────────────────────────
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Input)  { $Input  = Join-Path $repoRoot "tests/raw_vid.mp4" }
if (-not $LogDir) { $LogDir = Join-Path $repoRoot "out/bench" }

if (-not (Test-Path $Input)) { throw "input not found: $Input" }
$Input     = (Resolve-Path $Input).Path
$inputDir  = Split-Path $Input -Parent
$inputName = Split-Path $Input -Leaf

$outDir = Join-Path $repoRoot "out"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
$outDir = (Resolve-Path $outDir).Path

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$LogDir = (Resolve-Path $LogDir).Path

$configMount = $null
if ($Config) {
    if (-not (Test-Path $Config)) { throw "config not found: $Config" }
    $configMount = (Resolve-Path $Config).Path
}

# ── docker availability ──────────────────────────────────────────────────────
$null = & docker --version
if ($LASTEXITCODE -ne 0) { throw "'docker' not found on PATH" }

# ── Build (optional) ─────────────────────────────────────────────────────────
if ($Build) {
    Write-Host "==> docker build -f Dockerfile.gpu -t $Image ." -ForegroundColor Cyan
    Push-Location $repoRoot
    try {
        & docker build -f Dockerfile.gpu -t $Image .
        if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
    } finally { Pop-Location }
}

# Verify image exists
$imageId = (& docker images -q $Image) -join ""
if (-not $imageId) {
    throw @"
image '$Image' not found locally. Build it first with one of:
    pwsh $PSCommandPath -Build
    docker build -f Dockerfile.gpu -t $Image .
    docker compose build music-separator-gpu
"@
}

# ── Header ───────────────────────────────────────────────────────────────────
$runStamp = Get-Date -Format "yyyyMMdd-HHmmss"
Write-Host ""
Write-Host "image      : $Image"
Write-Host "input      : $Input  (bind-mounted as /inputs/$inputName)"
if ($configMount) {
    Write-Host "config     : $configMount  (bind-mounted as /app/config.json)"
} else {
    Write-Host "config     : built-in /app/config.json (docker/config.cuda.json)"
}
Write-Host "out dir    : $outDir  (bind-mounted as /out)"
Write-Host "log dir    : $LogDir"
Write-Host ("gpu        : {0}" -f ($(if ($NoGpu) { 'OFF (--gpus dropped)' } else { 'ON  (--gpus all)' })))
Write-Host ("iterations : {0} (after {1} warmup)" -f $Iterations, $WarmupIterations)
Write-Host ""

function Invoke-DockerRun {
    param([string]$Tag, [int]$Index, [int]$Total)

    $logFile = Join-Path $LogDir ("$runStamp-$Tag-$Index.log")

    $dockerArgs = @("run", "--rm")
    if (-not $NoGpu)    { $dockerArgs += @("--gpus", "all") }
    $dockerArgs += @("-v", "${inputDir}:/inputs:ro")
    $dockerArgs += @("-v", "${outDir}:/out")
    if ($configMount)  { $dockerArgs += @("-v", "${configMount}:/app/config.json:ro") }
    $dockerArgs += $Image
    $dockerArgs += @("--input", "/inputs/$inputName")

    # Redirect via cmd.exe so PowerShell 5.1 does not wrap docker's stderr
    # lines as NativeCommandError records (which would trip ErrorActionPreference
    # = Stop even when docker exited 0).
    $quoted = $dockerArgs | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }
    $cmdLine = 'docker ' + ($quoted -join ' ') + ' > "' + $logFile + '" 2>&1'

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & cmd.exe /c $cmdLine
    $exit = $LASTEXITCODE
    $sw.Stop()

    if ($exit -ne 0) {
        Write-Host "    last 30 lines of ${logFile}:" -ForegroundColor Red
        Get-Content $logFile -Tail 30 | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
        throw "$Tag $Index/$Total failed (docker exit $exit); see $logFile"
    }

    [PSCustomObject]@{
        Tag     = $Tag
        Index   = $Index
        Seconds = $sw.Elapsed.TotalSeconds
        Log     = $logFile
    }
}

# ── Warmup ───────────────────────────────────────────────────────────────────
for ($i = 1; $i -le $WarmupIterations; $i++) {
    Write-Host ("==> warmup {0}/{1}" -f $i, $WarmupIterations) -ForegroundColor DarkGray
    $r = Invoke-DockerRun -Tag "warmup" -Index $i -Total $WarmupIterations
    Write-Host ("    wall = {0,7:N3}s   log = {1}" -f $r.Seconds, $r.Log) -ForegroundColor DarkGray
}

# ── Timed runs ───────────────────────────────────────────────────────────────
$results = @()
for ($i = 1; $i -le $Iterations; $i++) {
    Write-Host ("==> run {0}/{1}" -f $i, $Iterations) -ForegroundColor Cyan
    $r = Invoke-DockerRun -Tag "run" -Index $i -Total $Iterations
    Write-Host ("    wall = {0,7:N3}s   log = {1}" -f $r.Seconds, $r.Log) -ForegroundColor Gray
    $results += $r
}

# ── Summary ──────────────────────────────────────────────────────────────────
$times  = $results | ForEach-Object { $_.Seconds }
$mean   = ($times | Measure-Object -Average).Average
$min    = ($times | Measure-Object -Minimum).Minimum
$max    = ($times | Measure-Object -Maximum).Maximum
$stddev = if ($times.Count -gt 1) {
    $m = $mean
    [Math]::Sqrt((($times | ForEach-Object { ($_ - $m) * ($_ - $m) }) | Measure-Object -Sum).Sum / ($times.Count - 1))
} else { 0 }

Write-Host ""
Write-Host "── Results ──────────────────────────────────────────────────" -ForegroundColor Green
Write-Host ("  iterations : {0}" -f $Iterations)
Write-Host ("  wall mean  : {0:N3} s" -f $mean)
Write-Host ("  wall min   : {0:N3} s" -f $min)
Write-Host ("  wall max   : {0:N3} s" -f $max)
Write-Host ("  wall stddev: {0:N3} s" -f $stddev)
Write-Host ""
Write-Host ("Per-run logs are in {0}." -f $LogDir) -ForegroundColor DarkGray
Write-Host ("Inspect a log for step-level timings:  Get-Content '{0}'" -f $results[-1].Log) -ForegroundColor DarkGray
