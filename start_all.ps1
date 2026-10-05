$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$venvPython = Join-Path $projectRoot 'venv\Scripts\python.exe'
if (Test-Path -LiteralPath $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonExe = (Get-Command python -ErrorAction Stop).Source
}

$services = @(
    @{ Name = 'Collector'; Folder = 'collector-agent'; Port = 8001; Status = '/collector/status' },
    @{ Name = 'Analysis'; Folder = 'iwra\analysis-agent'; Port = 8002; Status = '/analysis/status' },
    @{ Name = 'Guidance'; Folder = 'ir-module'; Port = 8003; Status = '/ir/status' },
    @{ Name = 'Alert'; Folder = 'alert-agent'; Port = 8004; Status = '/alert/status' },
    @{ Name = 'Summarizer'; Folder = 'summarizer-agent'; Port = 8005; Status = '/summarizer/status' },
    @{ Name = 'Router'; Folder = 'router-agent'; Port = 8000; Status = '/' }
)

$logs = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null

foreach ($service in $services) {
    $url = "http://127.0.0.1:$($service.Port)$($service.Status)"
    try {
        $null = Invoke-RestMethod -Uri $url -TimeoutSec 2
        Write-Host "$($service.Name) already running on port $($service.Port)."
        continue
    } catch {}

    $folder = Join-Path $projectRoot $service.Folder
    $outLog = Join-Path $logs "$($service.Name.ToLower()).out.log"
    $errLog = Join-Path $logs "$($service.Name.ToLower()).err.log"
    $process = Start-Process -FilePath $pythonExe -ArgumentList @('-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', [string]$service.Port) -WorkingDirectory $folder -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru

    $ready = $false
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            $null = Invoke-RestMethod -Uri $url -TimeoutSec 2
            $ready = $true
            break
        } catch {
            if ($process.HasExited) { break }
        }
    }

    if ($ready) {
        Write-Host "$($service.Name) ready on port $($service.Port)."
    } else {
        Write-Warning "$($service.Name) did not start. Check $errLog"
    }
}

Write-Host 'Open http://localhost:8000/ in your browser.'
