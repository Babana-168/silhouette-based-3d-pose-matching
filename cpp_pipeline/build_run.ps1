$ErrorActionPreference = 'Stop'

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found: $vswhere"
}

$vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
if (-not $vsInstall) {
    throw 'Visual Studio with C++ tools not found. Install Desktop development with C++.'
}

$vcvars = Join-Path $vsInstall 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) {
    throw "vcvars64.bat not found: $vcvars"
}

$cmd = "`"$vcvars`" >nul 2>&1 && cl.exe /O2 /fp:fast /openmp /arch:AVX2 /EHsc /std:c++17 /I`"C:\opencv\build\include`" main.cpp /Fe:pose_match.exe /link /LIBPATH:`"C:\opencv\build\x64\vc16\lib`" opencv_world4120.lib"
cmd /c $cmd
