$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$BASE = "D:\my_works\助步器项目"
$REPO = "$BASE\walker-pose-system"
$PACKAGE = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path -LiteralPath $REPO -PathType Container)) {
    throw "仓库目录不存在：$REPO"
}

$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$BACKUP = "${REPO}_backup_${STAMP}"

Write-Host "备份当前仓库：$BACKUP" -ForegroundColor Cyan
Copy-Item -LiteralPath $REPO -Destination $BACKUP -Recurse -Force

function Copy-IfExists {
    param(
        [string]$Source,
        [string]$Destination
    )
    if (Test-Path -LiteralPath $Source -PathType Leaf) {
        $parent = Split-Path -Parent $Destination
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
        Write-Host "[COPIED] $Source" -ForegroundColor Green
    }
    else {
        Write-Host "[SKIP] 不存在：$Source" -ForegroundColor Yellow
    }
}

Write-Host "`n===== 1. 创建标准目录 =====" -ForegroundColor Cyan
$DIRS = @(
    "$REPO\sequence_pipeline",
    "$REPO\benchmarks\speed",
    "$REPO\benchmarks\accuracy",
    "$REPO\benchmarks\qualitative",
    "$REPO\docs",
    "$REPO\third_party"
)
foreach ($DIR in $DIRS) {
    New-Item -ItemType Directory -Path $DIR -Force | Out-Null
}

Write-Host "`n===== 2. 安装仓库文档模板 =====" -ForegroundColor Cyan
Copy-Item "$PACKAGE\README.md" "$REPO\README.md" -Force
Copy-Item "$PACKAGE\.gitignore" "$REPO\.gitignore" -Force
Copy-Item "$PACKAGE\CHANGELOG.md" "$REPO\CHANGELOG.md" -Force
Copy-Item "$PACKAGE\PROJECT_STATUS.md" "$REPO\docs\PROJECT_STATUS.md" -Force
Copy-Item "$PACKAGE\THIRD_PARTY.md" "$REPO\third_party\README.md" -Force
Copy-Item "$PACKAGE\config.example.json" "$REPO\realtime_app\config.example.json" -Force

Write-Host "`n===== 3. 修正测试配置 =====" -ForegroundColor Cyan
$TEST_FILE = "$REPO\realtime_app\tests\test_core.py"
if (-not (Test-Path -LiteralPath $TEST_FILE)) {
    throw "测试文件不存在：$TEST_FILE"
}
$TEST_TEXT = Get-Content -LiteralPath $TEST_FILE -Raw -Encoding UTF8
$TEST_TEXT = $TEST_TEXT.Replace('load_config(root / "config.json")', 'load_config(root / "config.example.json")')
Set-Content -LiteralPath $TEST_FILE -Value $TEST_TEXT -Encoding UTF8

Write-Host "`n===== 4. 复制正式序列实验程序 =====" -ForegroundColor Cyan
$SEQ_SRC = "$BASE\pose_sequence_pipeline"
$SEQ_DST = "$REPO\sequence_pipeline"
if (-not (Test-Path -LiteralPath $SEQ_SRC -PathType Container)) {
    throw "序列实验程序不存在：$SEQ_SRC"
}
if (Test-Path -LiteralPath $SEQ_DST) {
    Remove-Item -LiteralPath $SEQ_DST -Recurse -Force
}
New-Item -ItemType Directory -Path $SEQ_DST -Force | Out-Null

robocopy $SEQ_SRC $SEQ_DST /E /R:1 /W:1 `
    /XD "__pycache__" ".git" `
    /XF "*.pyc" "*.bak" "local.json" "static_check_summary.json"

$ROBO_CODE = $LASTEXITCODE
if ($ROBO_CODE -gt 7) {
    throw "复制 sequence_pipeline 失败，robocopy退出码：$ROBO_CODE"
}

Write-Host "`n===== 5. 复制速度汇总 =====" -ForegroundColor Cyan
$SPEED = "$BASE\pose_speed_outputs"

$SPEED_FILES = @(
    "summary\bboxmaskpose_predict_once_summary.csv",
    "summary\pmpose_predict_once_summary.csv",
    "summary\probpose_summary.csv",
    "summary\sapiens2_summary.csv",
    "summary\yolo26x_pose_summary.csv",
    "summary\visualization_instance_counts.csv",
    "yolo26x_detector_summary.csv"
)

foreach ($REL in $SPEED_FILES) {
    $NAME = Split-Path $REL -Leaf
    Copy-IfExists "$SPEED\$REL" "$REPO\benchmarks\speed\$NAME"
}

Write-Host "`n===== 6. 复制精度汇总 =====" -ForegroundColor Cyan
$RESULTS = "$BASE\sequence_results"

$ACCURACY_MAP = @(
    @(
        "ochuman_val2500_bboxmaskpose_conf005_baseline_iter1\bboxmaskpose_coco_keypoint_summary.csv",
        "bboxmaskpose_coco_keypoint_summary.csv"
    ),
    @(
        "ochuman_val2500_bboxmaskpose_conf005_baseline_iter1\bboxmaskpose_final_operating_point.csv",
        "bboxmaskpose_final_operating_point.csv"
    ),
    @(
        "ochuman_val2500_pmpose_conf005_baseline_standard\baseline_coco_keypoint_summary.csv",
        "pmpose_coco_keypoint_summary.csv"
    ),
    @(
        "ochuman_val2500_probpose_conf005_clean_rerun\probpose_coco_keypoint_summary.csv",
        "probpose_coco_keypoint_summary.csv"
    ),
    @(
        "ochuman_val2500_probpose_conf005_clean_rerun\probpose_final_score030_summary.json",
        "probpose_final_score030_summary.json"
    ),
    @(
        "ochuman_val2500_sapiens2_conf005_noflip_b2_full\sapiens2_coco_summary.csv",
        "sapiens2_coco_summary.csv"
    ),
    @(
        "ochuman_val2500_sapiens2_conf005_noflip_b2_full\sapiens2_final_summary.json",
        "sapiens2_final_summary.json"
    )
)

foreach ($PAIR in $ACCURACY_MAP) {
    Copy-IfExists "$RESULTS\$($PAIR[0])" "$REPO\benchmarks\accuracy\$($PAIR[1])"
}

Write-Host "`n===== 7. 复制一组定性对比图 =====" -ForegroundColor Cyan
$MODELS = @("yolo26x_pose", "pmpose", "probpose", "bboxmaskpose", "sapiens2")
foreach ($MODEL in $MODELS) {
    Copy-IfExists `
        "$SPEED\visualizations\$MODEL\input8.jpg" `
        "$REPO\benchmarks\qualitative\${MODEL}_input8.jpg"
}

Write-Host "`n===== 8. 清理缓存 =====" -ForegroundColor Cyan
Get-ChildItem -LiteralPath $REPO -Directory -Filter "__pycache__" -Recurse -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force

Get-ChildItem -LiteralPath $REPO -File -Filter "*.pyc" -Recurse -ErrorAction SilentlyContinue |
    Remove-Item -Force

Get-ChildItem -LiteralPath $REPO -File -Filter "*.bak" -Recurse -ErrorAction SilentlyContinue |
    Remove-Item -Force

Write-Host "`n===== 9. 运行静态检查和测试 =====" -ForegroundColor Cyan
Set-Location "$REPO\realtime_app"

python -m compileall -q .
if ($LASTEXITCODE -ne 0) {
    throw "Python语法检查失败。"
}

python .\run_tests.py
if ($LASTEXITCODE -ne 0) {
    throw "单元测试失败。"
}

Write-Host "`n===== 10. 检查Git候选文件 =====" -ForegroundColor Cyan
Set-Location $REPO

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "未检测到Git，已完成文件整理，但尚未执行Git检查。" -ForegroundColor Yellow
}
else {
    if (-not (Test-Path -LiteralPath "$REPO\.git")) {
        git init | Out-Null
    }

    git status --short
}

Write-Host "`n整理完成。" -ForegroundColor Green
Write-Host "仓库：$REPO"
Write-Host "备份：$BACKUP"
Write-Host "注意：pose_sequence_pipeline_codes 未并入主仓库，它应先作为旧版代码保留在本地。"
