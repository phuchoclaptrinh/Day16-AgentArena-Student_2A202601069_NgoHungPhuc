param(
    [switch]$Full,
    [string]$Brief
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $repoRoot ".env"

if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Missing $envFile. Create it from .env.example."
}

foreach ($rawLine in Get-Content -LiteralPath $envFile -Encoding utf8) {
    $line = $rawLine.Trim()
    if (-not $line -or $line.StartsWith("#")) {
        continue
    }

    $pair = $line.Split("=", 2)
    if ($pair.Count -ne 2 -or -not $pair[0].Trim()) {
        throw "Invalid .env line: $rawLine"
    }

    $name = $pair[0].Trim()
    $value = $pair[1].Trim()
    if (
        $value.Length -ge 2 -and
        (($value.StartsWith('"') -and $value.EndsWith('"')) -or
         ($value.StartsWith("'") -and $value.EndsWith("'")))
    ) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    [Environment]::SetEnvironmentVariable($name, $value, "Process")
}

$required = "ARENA_API_KEY", "ARENA_BASE_URL", "ARENA_MODEL"
$missing = @($required | Where-Object {
    [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($_, "Process"))
})
if ($missing.Count -gt 0) {
    throw "Missing .env variables: $($missing -join ', ')"
}
if ($env:ARENA_API_KEY -eq "thay_bang_openai_api_key_cua_ban") {
    throw "Replace the ARENA_API_KEY placeholder in .env before running."
}
if ($Full -and $Brief) {
    throw "Use either -Full or -Brief, not both."
}

$env:PYTHONIOENCODING = "utf-8"
$env:ARENA_PROVIDER = "openai"
Push-Location $repoRoot
try {
    $arguments = @(
        "scripts\run_practice.py",
        "--model", "real",
        "--prompt-addendum",
        "--max-tokens", "1200"
    )
    if ($Full) {
        $arguments += @("--out", "runs\openai-full.json")
    } elseif ($Brief) {
        $safeBrief = $Brief -replace "[^A-Za-z0-9_-]", "_"
        $arguments += @(
            "--brief", $Brief,
            "--no-flaky",
            "--out", "runs\openai-$safeBrief.json"
        )
    } else {
        $arguments += @(
            "--brief", "pub-01-sla-hien-hanh",
            "--no-flaky",
            "--out", "runs\openai-smoke.json"
        )
    }

    & python @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
