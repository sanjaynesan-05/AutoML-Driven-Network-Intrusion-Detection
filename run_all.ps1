param([switch]$Dev,[string]$Stage="all")
$ErrorActionPreference="Stop"
$Root=Split-Path -Parent $MyInvocation.MyCommand.Path
$Python=Join-Path $Root ".venv\Scripts\python.exe"
$Log=Join-Path $Root "logs\run_all.log"
New-Item -ItemType Directory -Force (Join-Path $Root "logs") | Out-Null
if (!(Test-Path $Python)) { throw "Project venv interpreter missing: $Python" }
$Actual=& $Python -c "import sys; print(sys.executable)"
if ([System.IO.Path]::GetFullPath($Actual) -ne [System.IO.Path]::GetFullPath($Python)) { throw "Unexpected interpreter: $Actual" }
function Invoke-Stage($Name,$Module) {
  if ($Stage -ne "all" -and $Stage -ne $Name) { return }
  "[$(Get-Date -Format o)] START $Name" | Tee-Object -FilePath $Log -Append
  & $Python -m $Module 2>&1 | Tee-Object -FilePath $Log -Append
  if ($LASTEXITCODE -ne 0) { "[$(Get-Date -Format o)] FAIL $Name exit=$LASTEXITCODE" | Tee-Object -FilePath $Log -Append; throw "Stage failed: $Name" }
  "[$(Get-Date -Format o)] PASS $Name" | Tee-Object -FilePath $Log -Append
}
Invoke-Stage "profile" "src.exploration"
Invoke-Stage "baseline" "src.baseline"
Invoke-Stage "tuning" "src.tuning"
if ($Dev) { $env:AUTOML_BUDGET="90" }
Invoke-Stage "flaml" "src.automl_flaml"
Invoke-Stage "autogluon" "src.automl_autogluon"
Invoke-Stage "leakage" "src.leakage_experiment"
Invoke-Stage "bias" "src.bias_analysis"
Invoke-Stage "selection" "src.train_final_model"
Invoke-Stage "explainability" "src.explainability"
if ($Stage -eq "all" -or $Stage -eq "report") { & $Python -m src.reporting; if ($LASTEXITCODE -ne 0) { throw "report failed" } }
if ($Stage -eq "all" -or $Stage -eq "tests") { & $Python -m pytest; if ($LASTEXITCODE -ne 0) { throw "pytest failed" } }
Write-Host "Workflow stages completed. See logs/run_all.log; any failed stage stops the workflow."
