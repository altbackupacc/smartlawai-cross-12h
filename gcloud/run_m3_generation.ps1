param(
    [string]$Project = "smartlawai-1",
    [string]$Region = "us-central1",
    [string]$Image = "us-central1-docker.pkg.dev/smartlawai-1/smartlaw/train:latest",
    [string]$GcsBucket = "smartlawai-1-checkpoints",
    [string]$HfToken = $env:HF_TOKEN,
    [string]$Accel = "A100"
)

$ErrorActionPreference = "Stop"

if (-not $HfToken) {
    if ($env:HUGGING_FACE_HUB_TOKEN) {
        $HfToken = $env:HUGGING_FACE_HUB_TOKEN
    } else {
        $HfToken = "not_needed"
    }
}

if ($Accel -eq "A100") {
    $MachineType = "a2-highgpu-1g"
    $AccelType = "NVIDIA_TESLA_A100"
    $AccelCount = 1
} elseif ($Accel -eq "L4") {
    $MachineType = "g2-standard-8"
    $AccelType = "NVIDIA_L4"
    $AccelCount = 1
} else {
    Write-Error "Unknown Accel: $Accel. Expected A100 or L4."
}

$Timestamp = Get-Date -Format "yyyyMMddTHHmmss"
$JobName = "smartlaw-m3-generation-$($Accel.ToLower())-$Timestamp"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "SmartLawAI M3 Structured Generation Job (GCP $Accel)" -ForegroundColor Green
Write-Host "Machine: $MachineType with 1x $AccelType"
Write-Host "Region:  $Region"
Write-Host "Project: $Project"
Write-Host "Job:     $JobName"
Write-Host "============================================================" -ForegroundColor Cyan

$TempConfig = [System.IO.Path]::GetTempFileName()

$YamlConfig = @"
workerPoolSpecs:
  - machineSpec:
      machineType: $MachineType
      acceleratorType: $AccelType
      acceleratorCount: $AccelCount
    diskSpec:
      bootDiskSizeGb: 100
      bootDiskType: pd-ssd
    replicaCount: 1
    containerSpec:
      imageUri: "$Image"
      args:
        - "scripts.run_generation_bench"
        - "--device=cuda"
        - "--model-id=mistralai/Mistral-7B-Instruct-v0.3"
      env:
        - name: HF_TOKEN
          value: "$HfToken"
        - name: HUGGING_FACE_HUB_TOKEN
          value: "$HfToken"
        - name: GCS_BUCKET
          value: "$GcsBucket"
        - name: RUN_ID
          value: "$JobName"
"@

[System.IO.File]::WriteAllText($TempConfig, $YamlConfig)

try {
    & gcloud ai custom-jobs create `
        --project="$Project" `
        --region="$Region" `
        --display-name="$JobName" `
        --config="$TempConfig"

    Write-Host "[+] Successfully submitted: $JobName" -ForegroundColor Green
    Write-Host "    To stream logs, run:" -ForegroundColor Yellow
    Write-Host "    gcloud ai custom-jobs stream-logs $JobName --region=$Region" -ForegroundColor White
} finally {
    if (Test-Path $TempConfig) {
        Remove-Item $TempConfig -Force
    }
}
