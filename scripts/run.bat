@echo off
rem Whetstone launcher. Ships inside the release package as run.bat
rem
rem   run.bat probe                          what this GPU can do
rem   run.bat inspect  <model_dir>           architecture + roofline
rem   run.bat convert  <model_dir> [out]     build a .wstone
rem   run.bat verify   <file.wstone> [src]   check integrity and fidelity
rem   run.bat run      <file.wstone> --ids   generate, and report tok/s
rem   run.bat ppl      <file.wstone> --tokens <f>   wikitext-2 perplexity
rem   run.bat tune     <file.wstone>         pick the per-shape GEMV kernel
rem   run.bat tokens   [model_dir] [out]     tokenize wikitext-2 for ppl
rem   run.bat chat     <file.wstone>         interactive chat, tok/s per turn
rem   run.bat hfchat   [model_dir]           the HuggingFace chat harness
rem   run.bat bench    [model_dir]           throughput run
rem   run.bat download [dest]                fetch the reference model
rem   run.bat setup                          create the Python venv
rem   run.bat doctor                         diagnose the environment

rem Delayed expansion is deliberately NOT enabled: it would eat "!" in any
rem argument, and prompts routinely contain one. Every %VAR% below is read after
rem it is set, so nothing here needs it.
setlocal

set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
set "BIN=%HERE%\bin\whetstone.exe"
if not exist "%BIN%" set "BIN=%HERE%\target\release\whetstone.exe"
set "BENCH=%HERE%\bench"
set "VENV=%HERE%\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "DEFAULT_MODEL=%HERE%\models\Qwen2.5-0.5B-Instruct"

set "CMD=%~1"
if "%CMD%"=="" goto :usage
if "%CMD%"=="-h" goto :usage
if "%CMD%"=="--help" goto :usage
if "%CMD%"=="help" goto :usage
shift

if "%CMD%"=="probe"    goto :rust
if "%CMD%"=="inspect"  goto :rust
if "%CMD%"=="convert"  goto :convert
if "%CMD%"=="verify"   goto :verify
if "%CMD%"=="run"      goto :rust
if "%CMD%"=="ppl"      goto :rust
if "%CMD%"=="tune"     goto :rust
if "%CMD%"=="tokens"   goto :tokens
if "%CMD%"=="chat"     goto :rust
if "%CMD%"=="hfchat"   goto :chat
if "%CMD%"=="bench"    goto :bench
if "%CMD%"=="baseline" goto :baseline
if "%CMD%"=="download" goto :download
if "%CMD%"=="setup"    goto :setup
if "%CMD%"=="doctor"   goto :doctor

echo error: unknown command: %CMD% 1>&2
echo.
goto :usage

rem ---------------------------------------------------------------- helpers

:need_bin
if not exist "%BIN%" (
    echo error: whetstone.exe not found at %BIN% 1>&2
    echo Build it with:  cargo build --release 1>&2
    exit /b 1
)
exit /b 0

:need_py
if not exist "%PY%" (
    echo error: Python environment not set up. Run:  run.bat setup 1>&2
    exit /b 1
)
exit /b 0

rem ------------------------------------------------------------- rust paths

:rust
rem `shift` does not affect %*, so passing %* here would repeat the subcommand:
rem `run.bat probe --iters 100` became `whetstone probe probe --iters 100`.
rem Collect the remaining arguments one at a time instead.
call :need_bin || exit /b 1
set "ARGS="
:rust_collect
if "%~1"=="" goto :rust_go
set "ARGS=%ARGS% "%~1""
shift
goto :rust_collect
:rust_go
"%BIN%" %CMD%%ARGS%
exit /b %ERRORLEVEL%

:tokens
call :need_py || exit /b 1
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%DEFAULT_MODEL%"
if not "%~1"=="" shift
set "OUT=%~1"
if "%OUT%"=="" set "OUT=%HERE%\tokens.u32"
if not "%~1"=="" shift
set "ARGS="
:tokens_collect
if "%~1"=="" goto :tokens_go
set "ARGS=%ARGS% "%~1""
shift
goto :tokens_collect
:tokens_go
"%PY%" "%BENCH%\prepare_tokens.py" --model "%MODEL%" --out "%OUT%"%ARGS%
exit /b %ERRORLEVEL%

:convert
call :need_bin || exit /b 1
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%DEFAULT_MODEL%"
if not exist "%MODEL%" (
    echo error: no such model directory: %MODEL% 1>&2
    echo Fetch the reference model with:  run.bat download 1>&2
    exit /b 1
)
shift
set "OUT=%~1"
if "%OUT%"=="" (set "OUT=%HERE%\model.wstone") else (shift)
set "ARGS="
:convert_collect
if "%~1"=="" goto :convert_go
set "ARGS=%ARGS% "%~1""
shift
goto :convert_collect
:convert_go
"%BIN%" convert "%MODEL%" -o "%OUT%"%ARGS%
exit /b %ERRORLEVEL%

:verify
call :need_bin || exit /b 1
set "FILE=%~1"
if "%FILE%"=="" (
    echo usage: run.bat verify ^<file.wstone^> [source_model_dir] 1>&2
    exit /b 1
)
shift
set "SRC=%~1"
if "%SRC%"=="" ("%BIN%" verify "%FILE%") else ("%BIN%" verify "%FILE%" --source "%SRC%")
exit /b %ERRORLEVEL%

rem ----------------------------------------------------------- python paths

:chat
call :need_py || exit /b 1
call :resolve_model %1 || exit /b 1
if not "%~1"=="" shift
set "ARGS="
:hfchat_collect
if "%~1"=="" goto :hfchat_go
set "ARGS=%ARGS% "%~1""
shift
goto :hfchat_collect
:hfchat_go
"%PY%" "%BENCH%\chat.py" --model "%MODEL%"%ARGS%
exit /b %ERRORLEVEL%

:bench
call :need_py || exit /b 1
call :resolve_model %1 || exit /b 1
if not "%~1"=="" shift
set "ARGS="
:bench_collect
if "%~1"=="" goto :bench_go
set "ARGS=%ARGS% "%~1""
shift
goto :bench_collect
:bench_go
"%PY%" "%BENCH%\chat.py" --model "%MODEL%" --bench%ARGS%
exit /b %ERRORLEVEL%

:baseline
call :need_py || exit /b 1
call :resolve_model %1 || exit /b 1
if not "%~1"=="" shift
set "ARGS="
:baseline_collect
if "%~1"=="" goto :baseline_go
set "ARGS=%ARGS% "%~1""
shift
goto :baseline_collect
:baseline_go
"%PY%" "%BENCH%\baseline_hf.py" --model "%MODEL%"%ARGS%
exit /b %ERRORLEVEL%

:download
call :need_py || exit /b 1
set "DEST=%~1"
if "%DEST%"=="" set "DEST=%HERE%\models"
"%PY%" "%BENCH%\download_model.py" --out "%DEST%"
exit /b %ERRORLEVEL%

:resolve_model
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%DEFAULT_MODEL%"
if not exist "%MODEL%" (
    echo error: no such model directory: %MODEL% 1>&2
    echo Fetch the reference model with:  run.bat download 1>&2
    exit /b 1
)
exit /b 0

rem ------------------------------------------------------------------ setup

:setup
echo ==^> creating Python environment in %VENV%
where python >nul 2>&1 || (echo error: python not found on PATH 1>&2 & exit /b 1)
python -m venv "%VENV%" || (echo error: could not create venv 1>&2 & exit /b 1)
"%VENV%\Scripts\pip.exe" install --quiet --upgrade pip
echo ==^> installing torch ^(large; this can take a while^)
"%VENV%\Scripts\pip.exe" install --quiet torch || (echo error: torch install failed 1>&2 & exit /b 1)
"%VENV%\Scripts\pip.exe" install --quiet transformers safetensors huggingface_hub regex ^
    || (echo error: dependency install failed 1>&2 & exit /b 1)
echo   ok environment ready
echo.
echo   next:  run.bat download   ^(fetch the reference model^)
echo          run.bat chat       ^(talk to it^)
exit /b 0

rem ----------------------------------------------------------------- doctor

:doctor
echo ==^> environment
where nvidia-smi >nul 2>&1 && (
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
) || echo warn: nvidia-smi not found -- no NVIDIA driver?
if exist "%BIN%" ("%BIN%" --version) else (echo warn: whetstone.exe missing at %BIN%)
if exist "%PY%" (
    "%PY%" --version
    "%PY%" -c "import importlib;[print('  ok  ',m,getattr(importlib.import_module(m),'__version__','?')) for m in ('torch','transformers')]" 2>nul ^
        || echo warn: torch/transformers not importable
) else (
    echo warn: no Python environment -- run run.bat setup
)
if exist "%DEFAULT_MODEL%" (echo   ok model %DEFAULT_MODEL%) else (echo warn: reference model not downloaded -- run run.bat download)
echo.
if exist "%BIN%" "%BIN%" probe --iters 20000
exit /b 0

rem ------------------------------------------------------------------ usage

:usage
echo.
echo Whetstone launcher
echo.
echo   run.bat probe                          what this GPU can do
echo   run.bat inspect  ^<model_dir^>           architecture + roofline
echo   run.bat convert  ^<model_dir^> [out]     build a .wstone
echo   run.bat verify   ^<file.wstone^> [src]   check integrity and fidelity
echo   run.bat chat     [model_dir]           interactive chat, live tok/s
echo   run.bat bench    [model_dir]           throughput run
echo   run.bat baseline [model_dir]           HF baseline: tok/s + perplexity
echo   run.bat download [dest]                fetch the reference model
echo   run.bat setup                          create the Python venv
echo   run.bat doctor                         diagnose the environment
echo.
echo The Rust subcommands need only the binary. The Python ones ^(chat, bench,
echo download^) need a virtualenv, which 'setup' creates in .venv
echo.
exit /b 0
