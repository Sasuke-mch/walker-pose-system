param(
    [int]$LeftCamera = 1,
    [int]$RightCamera = 0,
    [Parameter(Mandatory=$true)][string]$Calibration,
    [ValidateSet("yolo26x_pose", "pmpose")][string]$Model = "yolo26x_pose",
    [double]$MaxPairDeltaMs = 25.0,
    [int]$MaxPairs = 0
)

$ErrorActionPreference = "Stop"
$APP = Split-Path -Parent $PSScriptRoot
Set-Location $APP

$arguments = @(
    ".\run_stereo.py",
    "--model", $Model,
    "--left-camera", $LeftCamera,
    "--right-camera", $RightCamera,
    "--max-pair-delta-ms", $MaxPairDeltaMs,
    "--calibration", $Calibration
)

if ($MaxPairs -gt 0) {
    $arguments += @("--max-pairs", $MaxPairs)
}

python @arguments
exit $LASTEXITCODE
