[CmdletBinding()]
param()

# Canonical model-layout migration for walker-pose-system.
# This script is intentionally non-destructive: it copies weights first,
# backs up every edited JSON file, then updates only active model paths.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ModelsRoot = Join-Path $ProjectRoot "models"

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "realtime_app"))) {
    throw "This script must be stored and run as <project-root>\\scripts\\migrate_model_layout.ps1. Current root: $ProjectRoot"
}

function Require-File([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required model file is missing: $Path"
    }
}

function Require-Directory([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Required model directory is missing: $Path"
    }
}

function Copy-ModelFile([string]$Source, [string]$Destination) {
    Require-File $Source
    $destinationDir = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null

    if (Test-Path -LiteralPath $Destination) {
        $sourceHash = (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash
        $destinationHash = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash
        if ($sourceHash -ne $destinationHash) {
            throw "Existing destination differs from its source. Resolve manually before continuing: $Destination"
        }
        return
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -ErrorAction Stop
}

function Copy-ModelTree([string]$Source, [string]$Destination) {
    Require-Directory $Source
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Get-ChildItem -LiteralPath $Source -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $Destination -Recurse -Force -ErrorAction Stop
    }
}

function Get-RelativeProjectPath([string]$AbsolutePath) {
    return $AbsolutePath.Substring($ProjectRoot.Length + 1).Replace("\\", "/")
}

function Backup-And-WriteJson([System.IO.FileInfo]$File, [scriptblock]$Mutation) {
    $backup = "$($File.FullName).before-model-layout.bak"
    if (-not (Test-Path -LiteralPath $backup)) {
        Copy-Item -LiteralPath $File.FullName -Destination $backup -ErrorAction Stop
    }
    $config = Get-Content -LiteralPath $File.FullName -Raw | ConvertFrom-Json
    & $Mutation $config
    $config | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $File.FullName -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path $ModelsRoot | Out-Null

# Existing canonical inputs and their new, single-purpose locations.
$YoloDetector = Join-Path $ModelsRoot "yolo26\\yolo26x.pt"
Copy-ModelFile (Join-Path $ModelsRoot "yolo26x-pose.pt") (Join-Path $ModelsRoot "yolo26\\yolo26x-pose.pt")
Copy-ModelFile (Join-Path $ModelsRoot "PMPose-b-1.0.0.pth") (Join-Path $ModelsRoot "pmpose\\PMPose-b-1.0.0.pth")
Copy-ModelFile (Join-Path $ModelsRoot "SAM-pose2seg_hiera_b+.pt") (Join-Path $ModelsRoot "bboxmaskpose\\SAM-pose2seg_hiera_b+.pt")
Require-File $YoloDetector

Copy-ModelTree (Join-Path $ProjectRoot "third_party\\probpose\\weights\\ProbPose-s") (Join-Path $ModelsRoot "probpose\\ProbPose-s")
Copy-ModelTree (Join-Path $ProjectRoot "third_party\\sapiens2\\weights\\sapiens2-pose-0.4b") (Join-Path $ModelsRoot "sapiens2\\pose-0.4b")

$ProbPoseWeights = @(Get-ChildItem -LiteralPath (Join-Path $ModelsRoot "probpose\\ProbPose-s") -Recurse -File -Filter "ProbPose-s.pth")
if ($ProbPoseWeights.Count -ne 1) {
    throw "Expected exactly one ProbPose-s.pth under models\\probpose\\ProbPose-s; found $($ProbPoseWeights.Count)."
}
$SapiensWeights = @(Get-ChildItem -LiteralPath (Join-Path $ModelsRoot "sapiens2\\pose-0.4b") -Recurse -File -Filter "*.safetensors")
$CanonicalSapiensWeights = @($SapiensWeights | Where-Object { $_.Name -eq "model.safetensors" })
if ($CanonicalSapiensWeights.Count -ne 1) {
    throw "Expected exactly one canonical Sapiens2 checkpoint named model.safetensors under models\\sapiens2\\pose-0.4b; found $($CanonicalSapiensWeights.Count)."
}
$DistinctSapiensHashes = @(
    $SapiensWeights |
        ForEach-Object { (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash } |
        Sort-Object -Unique
)
if ($DistinctSapiensHashes.Count -ne 1) {
    throw "More than one distinct Sapiens2 .safetensors file was found. Do not select a checkpoint by filename alone."
}
$SapiensWeight = $CanonicalSapiensWeights[0]

$ProbPoseWeightPath = Get-RelativeProjectPath $ProbPoseWeights[0].FullName
$SapiensWeightPath = Get-RelativeProjectPath $SapiensWeight.FullName

$RealtimeConfigs = @(Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "realtime_app") -File -Filter "config*.json")
foreach ($file in $RealtimeConfigs) {
    Backup-And-WriteJson $file {
        param($config)
        if ($null -ne $config.model) {
            $config.model.weight = "models/yolo26/yolo26x-pose.pt"
        }
        if ($null -ne $config.detector) {
            $config.detector.repo = "third_party/yolo26x-pose"
            $config.detector.weight = "models/yolo26/yolo26x.pt"
        }
        if ($null -ne $config.pmpose) {
            $config.pmpose.repo = "third_party/bboxmaskpose"
            $config.pmpose.cache = "models"
        }
    }
}

$SequenceConfigsDir = Join-Path $ProjectRoot "sequence_pipeline\\configs"
$SequenceConfigs = @()
if (Test-Path -LiteralPath $SequenceConfigsDir) {
    $SequenceConfigs = @(Get-ChildItem -LiteralPath $SequenceConfigsDir -File -Filter "*.json")
}
foreach ($file in $SequenceConfigs) {
    Backup-And-WriteJson $file {
        param($config)
        $modelsProperty = $config.PSObject.Properties["models"]
        if ($null -eq $modelsProperty) {
            return
        }
        $models = $modelsProperty.Value
        if ($null -ne $models.yolo26_detector) {
            $models.yolo26_detector.weight.host_path = "models/yolo26/yolo26x.pt"
        }
        if ($null -ne $models.yolo26x_pose) {
            $models.yolo26x_pose.weight.host_path = "models/yolo26/yolo26x-pose.pt"
        }
        if ($null -ne $models.pmpose) {
            $models.pmpose.repo.host_path = "third_party/bboxmaskpose"
            $models.pmpose.weight.host_path = "models/pmpose/PMPose-b-1.0.0.pth"
        }
        if ($null -ne $models.bboxmaskpose) {
            $models.bboxmaskpose.repo.host_path = "third_party/bboxmaskpose"
            $models.bboxmaskpose.weight.host_path = "models/pmpose/PMPose-b-1.0.0.pth"
            $models.bboxmaskpose.required_paths[0].path = "third_party/bboxmaskpose/tools/benchmark_from_yolo26_bboxes.py"
            $models.bboxmaskpose.required_paths[1].path = "third_party/bboxmaskpose/bboxmaskpose/configs/bmp_v2.yaml"
        }
        if ($null -ne $models.probpose) {
            $models.probpose.repo.host_path = "third_party/probpose/repo"
            $models.probpose.weight.host_path = $ProbPoseWeightPath
            $models.probpose.required_paths[0].path = "third_party/probpose/repo/tools/run_probpose_from_yolo26_bboxes.py"
        }
        if ($null -ne $models.sapiens2) {
            $models.sapiens2.repo.host_path = "third_party/sapiens2/repo"
            $models.sapiens2.weight.host_path = $SapiensWeightPath
            $models.sapiens2.required_paths[0].path = "third_party/sapiens2/repo/sapiens/pose/tools/vis/run_sapiens2_from_yolo26_bboxes.py"
        }
    }
}

$ModelFiles = @(Get-ChildItem -LiteralPath $ModelsRoot -Recurse -File | Where-Object {
    $_.Extension -in ".pt", ".pth", ".safetensors"
})
$Manifest = foreach ($modelFile in $ModelFiles) {
    $hash = Get-FileHash -LiteralPath $modelFile.FullName -Algorithm SHA256
    [pscustomobject]@{
        path = Get-RelativeProjectPath $modelFile.FullName
        bytes = $modelFile.Length
        sha256 = $hash.Hash
    }
}
$Manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $ModelsRoot "MODEL_MANIFEST.json") -Encoding UTF8

$AllActiveConfigs = @($RealtimeConfigs) + @($SequenceConfigs)
$StaleReferences = @()
foreach ($file in $AllActiveConfigs) {
    $StaleReferences += @(Select-String -LiteralPath $file.FullName -Pattern "YOLO26-test|external_models|\.\./BBoxMaskPose" -AllMatches)
}
if ($StaleReferences.Count -gt 0) {
    $StaleReferences | ForEach-Object { Write-Host "Stale configuration reference: $($_.Path):$($_.LineNumber): $($_.Line.Trim())" }
    throw "A stale model or repository path remains in an active configuration."
}

Write-Host "Migration completed without deleting any legacy copies."
Write-Host "Model manifest: $(Join-Path $ModelsRoot 'MODEL_MANIFEST.json')"
Write-Host "Next verification commands:"
Write-Host "  python .\\realtime_app\\run_tests.py"
Write-Host "  python .\\realtime_app\\check_environment.py"
