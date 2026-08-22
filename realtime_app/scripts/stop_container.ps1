$ErrorActionPreference = "Continue"
docker rm -f pose-video-yolo26x-pose 2>$null | Out-Null
Write-Host "YOLO26x-pose video test container removed."
