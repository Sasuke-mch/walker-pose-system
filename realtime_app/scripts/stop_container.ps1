$ErrorActionPreference = "Continue"

$containers = @(
    "pose-video-yolo26x-pose",
    "pose-video-yolo26x-detector",
    "pose-video-pmpose"
)

foreach ($name in $containers) {
    docker rm -f $name 2>$null | Out-Null
}

Write-Host "Pose service containers removed if they existed."
