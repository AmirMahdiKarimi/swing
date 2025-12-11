$ErrorActionPreference = "Stop"

$data = "SWING-main/SWING-main/Data/ClassI_Model/ClassI_training_210.csv"
$common_flags = "--mhc_grouping two_digit --anchor_strict --add_blosum --kmer_k 3 --no_embed_aaindex --add_protbert"

Write-Host "----------------------------------------------------------------"
Write-Host "Starting Logistic Regression Training..."
Write-Host "----------------------------------------------------------------"
# We write out the full command to ensure arguments are passed correctly
cmd /c "python SWING-main/SWING-main/Scripts/Models/LogReg/train.py --data_set $data $common_flags"
if ($LASTEXITCODE -ne 0) { Write-Error "Logistic Regression failed" }

Write-Host "----------------------------------------------------------------"
Write-Host "Starting XGBoost Training..."
Write-Host "----------------------------------------------------------------"
cmd /c "python SWING-main/SWING-main/Scripts/Models/XGBoost/train.py --data_set $data $common_flags"
if ($LASTEXITCODE -ne 0) { Write-Error "XGBoost failed" }

Write-Host "----------------------------------------------------------------"
Write-Host "Starting LightGBM Training..."
Write-Host "----------------------------------------------------------------"
cmd /c "python SWING-main/SWING-main/Scripts/Models/LightGBM/train.py --data_set $data $common_flags"
if ($LASTEXITCODE -ne 0) { Write-Error "LightGBM failed" }

Write-Host "----------------------------------------------------------------"
Write-Host "Starting CatBoost Training..."
Write-Host "----------------------------------------------------------------"
cmd /c "python SWING-main/SWING-main/Scripts/Models/CatBoost/train.py --data_set $data $common_flags"
if ($LASTEXITCODE -ne 0) { Write-Error "CatBoost failed" }

Write-Host "----------------------------------------------------------------"
Write-Host "All training runs completed."
Write-Host "----------------------------------------------------------------"
