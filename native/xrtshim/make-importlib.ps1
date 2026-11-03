# Makes xrt_coreutil.lib, the import library MSVC links xrtshim against, from the
# xrt_coreutil.dll AMD's NPU driver installs. Run in a "Developer PowerShell for VS" (it needs
# dumpbin and lib).
#
#   .\make-importlib.ps1 [-Dll C:\Windows\System32\AMD\xrt_coreutil.dll] [-Out xrt_coreutil.lib]
param(
  [string]$Dll = "$env:SystemRoot\System32\AMD\xrt_coreutil.dll",
  [string]$Out = "xrt_coreutil.lib"
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $Dll)) {
  $found = Get-ChildItem "$env:SystemRoot\System32\DriverStore\FileRepository" -Recurse -Filter xrt_coreutil.dll -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $found) { throw "xrt_coreutil.dll not found; pass -Dll" }
  $Dll = $found.FullName
}
Write-Host "exports of $Dll"
$def = [System.IO.Path]::ChangeExtension($Out, ".def")
$lines = & dumpbin /exports $Dll
$names = @()
$inTable = $false
foreach ($l in $lines) {
  if ($l -match '^\s+ordinal\s+hint\s+RVA\s+name') { $inTable = $true; continue }
  if ($inTable -and $l -match '^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+(\S+)') { $names += $Matches[1] }
  elseif ($inTable -and $l -match '^\s*Summary') { break }
}
if ($names.Count -eq 0) { throw "no exports read from $Dll" }
"LIBRARY xrt_coreutil`r`nEXPORTS" | Set-Content $def
$names | ForEach-Object { "    $_" } | Add-Content $def
& lib /nologo /def:$def /machine:x64 /out:$Out
Write-Host "$Out ($($names.Count) exports)"
