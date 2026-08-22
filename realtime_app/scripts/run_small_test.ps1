param(
    [Parameter(Mandatory=$true)][string]$Video,
    [int]$Frames = 30
)
$ErrorActionPreference = "Stop"
$APP = Split-Path -Parent $PSScriptRoot
Set-Location $APP
python .\run.py --video $Video --max-frames $Frames --headless
exit $LASTEXITCODE
