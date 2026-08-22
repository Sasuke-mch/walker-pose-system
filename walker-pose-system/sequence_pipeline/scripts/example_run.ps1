param(
  [Parameter(Mandatory=$true)][string]$InputDir,
  [string]$OutputDir = ""
)
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
  $OutputDir = Join-Path (Resolve-Path "..") "sequence_results"
}
python .\run_sequence.py run --input-dir $InputDir --output-dir $OutputDir --models all --continue-on-error
