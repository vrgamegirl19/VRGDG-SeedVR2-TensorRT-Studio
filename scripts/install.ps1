param(
    [switch]$SkipModels,
    [switch]$SkipTensorRT,
    [switch]$Repair
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$StudioRoot = Split-Path -Parent $PSScriptRoot
$Outputs = Join-Path $StudioRoot 'outputs'
$Log = Join-Path $Outputs 'install.log'
$Venv = Join-Path $StudioRoot '.venv'
$StudioPython = Join-Path $Venv 'Scripts\python.exe'
$Marker = Join-Path $StudioRoot '.seedvr-studio-installed'
Push-Location -LiteralPath $StudioRoot

New-Item -ItemType Directory -Force $Outputs | Out-Null
try { Start-Transcript -Path $Log -Append | Out-Null } catch {}

function Write-Step([string]$Message) {
    Write-Host ''
    Write-Host ('== ' + $Message + ' ==') -ForegroundColor Cyan
}

function Refresh-Path {
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = $machinePath + ';' + $userPath
}

function Find-Python312 {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        $path = & $py.Source -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $path) { return ($path | Select-Object -Last 1).Trim() }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        $version = & $python.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and [version]$version -eq [version]'3.12') { return $python.Source }
    }
    $locations = @(
        (Join-Path $env:LocalAppData 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:ProgramFiles 'Python312\python.exe')
    )
    return $locations | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

function Install-WingetPackage([string]$Id, [string]$Name) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "$Name is missing and Windows Package Manager (winget) is unavailable. Install $Name manually, then run this installer again."
    }
    Write-Step "Installing $Name"
    & $winget.Source install --id $Id --exact --accept-package-agreements --accept-source-agreements --silent
    if ($LASTEXITCODE -ne 0) { throw "winget could not install $Name (exit $LASTEXITCODE)." }
    Refresh-Path
}

function Invoke-Pip([string[]]$Arguments) {
    & $StudioPython -m pip --isolated @Arguments
    if ($LASTEXITCODE -ne 0) { throw "pip failed: $($Arguments -join ' ')" }
}

try {
    Write-Host 'SeedVR Studio complete Windows installer' -ForegroundColor Green
    Write-Host 'Installs the UI, CUDA PyTorch, SeedVR2, SageAttention 2, TensorRT RTX, models, FFmpeg, and GPU-specific engines.'
    Write-Host 'Interrupted downloads and engine builds can be resumed by running this installer again.'

    if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
        throw 'No NVIDIA driver was detected. Install a current NVIDIA driver, restart Windows, and run this installer again.'
    }

    if (-not (Get-Command ffmpeg.exe -ErrorAction SilentlyContinue) -or -not (Get-Command ffprobe.exe -ErrorAction SilentlyContinue)) {
        Install-WingetPackage 'Gyan.FFmpeg' 'FFmpeg'
    }

    $BasePython = Find-Python312
    if (-not $BasePython) {
        Install-WingetPackage 'Python.Python.3.12' 'Python 3.12'
        $BasePython = Find-Python312
    }
    if (-not $BasePython) { throw 'Python 3.12 was installed but could not be located. Restart Windows and run the installer again.' }

    Write-Step 'Creating the private Python environment'
    if ($Repair -and (Test-Path $Venv)) {
        Write-Warning 'Repair mode reuses the existing environment and reinstalls all packages.'
    }
    if (-not (Test-Path $StudioPython)) {
        & $BasePython -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python virtual environment.' }
    }
    Invoke-Pip @('install', '--upgrade', 'pip', 'setuptools', 'wheel', '--index-url', 'https://pypi.org/simple')

    Write-Step 'Installing the tested CUDA 13 PyTorch build'
    $torchArguments = @(
        '-m', 'pip', '--isolated', 'install', '--pre',
        'torch==2.15.0.dev20260824+cu130',
        'torchvision==0.30.0.dev20260824+cu130',
        'torchaudio==2.11.0.dev20260824+cu130',
        '--index-url', 'https://download.pytorch.org/whl/nightly/cu130'
    )
    & $StudioPython @torchArguments
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'The exact tested nightly is unavailable; installing the newest mutually compatible CUDA 13 nightly.'
        Invoke-Pip @('install', '--pre', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/nightly/cu130')
    }

    Write-Step 'Installing SeedVR Studio and SeedVR2 dependencies'
    Invoke-Pip @('install', '--editable', $StudioRoot, '--index-url', 'https://pypi.org/simple')
    Invoke-Pip @('install', '--requirement', (Join-Path $StudioRoot 'vendor\seedvr2\requirements.txt'), '--index-url', 'https://pypi.org/simple')

    Write-Step 'Installing SageAttention 2 and TensorRT RTX'
    Invoke-Pip @('install', '--requirement', (Join-Path $StudioRoot 'requirements-windows-cu130.txt'), '--index-url', 'https://pypi.org/simple')

    Write-Step 'Checking CUDA and attention support'
    & $StudioPython -c "import sys, torch; sys.path.insert(0, 'vendor/seedvr2'); assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0)); from src.optimization.compatibility import SAGE_ATTN_2_AVAILABLE; assert SAGE_ATTN_2_AVAILABLE, 'SageAttention 2 did not load'; print('SageAttention 2: ready')"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA or SageAttention verification failed. See docs\SAGEATTENTION.md.' }

    if (-not $SkipModels) {
        Write-Step 'Downloading the default SeedVR2 3B FP8 model and VAE'
        & $StudioPython (Join-Path $PSScriptRoot 'download_models.py')
        if ($LASTEXITCODE -ne 0) { throw 'The model download failed. Run this installer again to resume it.' }
    } else {
        Write-Warning 'Model download was skipped. Models will download when first selected.'
    }

    if (-not $SkipTensorRT) {
        Write-Step 'Building GPU-specific TensorRT VAE engines'
        Write-Host 'This is a one-time operation and may take a while. Do not copy these engines to a different GPU.'
        & $StudioPython (Join-Path $PSScriptRoot 'prepare_tensorrt.py')
        if ($LASTEXITCODE -ne 0) { throw 'TensorRT engine preparation failed. Run this installer again to resume it.' }
    } else {
        Write-Warning 'TensorRT engine preparation was skipped. Only the Legacy engine will be usable.'
    }

    if (-not $SkipModels -and -not $SkipTensorRT) {
        Write-Step 'Final readiness check'
        Refresh-Path
        & $StudioPython (Join-Path $PSScriptRoot 'verify_install.py')
        if ($LASTEXITCODE -ne 0) { throw 'The final installation check failed.' }
        Set-Content -LiteralPath $Marker -Value (Get-Date -Format o) -Encoding ascii
    }

    Write-Host ''
    Write-Host 'SeedVR Studio is installed and ready.' -ForegroundColor Green
    Write-Host 'Double-click Launch SeedVR Studio Pro.bat to start.'
}
catch {
    Write-Host ''
    Write-Host ('INSTALLATION FAILED: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host ('Detailed log: ' + $Log) -ForegroundColor Yellow
    exit 1
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
    try { Pop-Location } catch {}
}
