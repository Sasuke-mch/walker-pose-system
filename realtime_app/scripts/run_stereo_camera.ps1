param(
    [int]$LeftCamera = 0,
    [int]$RightCamera = 1,
    [Parameter(Mandatory=$true)][string]$Calibration,
    [ValidateSet("yolo26x_pose", "pmpose")][string]$Model = "yolo26x_pose",
    [int]$MaxPairs = 0
)

$arguments = @(
    ".\run_stereo.py",
    "--model", $Model,
    "--left-camera", $LeftCamera,
    "--right-camera", $RightCamera,
    "--calibration", $Calibration
)
if ($MaxPairs -gt 0) {
    $arguments += @("--max-pairs", $MaxPairs)
}
python @arguments
