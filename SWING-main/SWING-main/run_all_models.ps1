$ErrorActionPreference = "Stop"

# Define models and their script paths
$models = @(
    @{ Name = "LogReg";   Script = "Scripts/Models/LogReg/train.py" },
    @{ Name = "XGBoost";  Script = "Scripts/Models/XGBoost/train.py" },
    @{ Name = "LightGBM"; Script = "Scripts/Models/LightGBM/train.py" },
    @{ Name = "CatBoost"; Script = "Scripts/Models/CatBoost/train.py" }
)

# Common arguments
$dataSet = "Data/ClassI_Model/ClassI_training_210.csv"
# Explicitly use Epitope (Peptide) as the sequence column and Hit as label
# Enable ProtBert and MHC one-hot encoding
$commonArgs = @(
    "--data_set", $dataSet,
    "--seq_col", "Epitope",
    "--label_col", "Hit",
    "--add_mhc",
    "--add_protbert"
)

foreach ($model in $models) {
    $modelName = $model.Name
    $scriptPath = $model.Script
    $outputDir = "Results/$modelName"
    
    Write-Host "Training $modelName..."
    
    # Create specific args for this model
    $args = @($scriptPath) + $commonArgs + @("--output_dir", $outputDir)
    
    # Run the python script
    python @args
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$modelName training failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    
    Write-Host "$modelName training completed successfully."
}

Write-Host "All models trained successfully."
