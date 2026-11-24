@echo off
cd /d "%~dp0"
title TSNE Customer Clustering

set "PY_EXE="

where py >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%i in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
        if not defined PY_EXE set "PY_EXE=%%i"
    )
)

if not defined PY_EXE (
    where python >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do (
            if not defined PY_EXE set "PY_EXE=%%i"
        )
    )
)

if not defined PY_EXE (
    where python3 >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%i in ('python3 -c "import sys; print(sys.executable)" 2^>nul') do (
            if not defined PY_EXE set "PY_EXE=%%i"
        )
    )
)

if not defined PY_EXE (
    echo [LOI] Khong tim thay Python tren may.
    pause
    exit /b 1
)

echo [*] Python: "%PY_EXE%"
"%PY_EXE%" --version

set "RUNNER=%TEMP%\tsne_runner_%RANDOM%.py"

> "%RUNNER%"  echo import ast
>> "%RUNNER%" echo import importlib.util
>> "%RUNNER%" echo from pathlib import Path
>> "%RUNNER%" echo import subprocess
>> "%RUNNER%" echo import sys
>> "%RUNNER%" echo.
>> "%RUNNER%" echo IMPORT_MAP = {
>> "%RUNNER%" echo     "bs4": "beautifulsoup4",
>> "%RUNNER%" echo     "cv2": "opencv-python",
>> "%RUNNER%" echo     "dotenv": "python-dotenv",
>> "%RUNNER%" echo     "fitz": "pymupdf",
>> "%RUNNER%" echo     "PIL": "Pillow",
>> "%RUNNER%" echo     "sklearn": "scikit-learn",
>> "%RUNNER%" echo     "skimage": "scikit-image",
>> "%RUNNER%" echo     "umap": "umap-learn",
>> "%RUNNER%" echo     "yaml": "PyYAML",
>> "%RUNNER%" echo }
>> "%RUNNER%" echo.
>> "%RUNNER%" echo SKIP_DIRS = {
>> "%RUNNER%" echo     ".git", ".hg", ".svn", ".venv", "venv", "env",
>> "%RUNNER%" echo     "__pycache__", ".pytest_cache", ".mypy_cache",
>> "%RUNNER%" echo     ".idea", ".vscode", ".devcontainer", "node_modules", "build", "dist",
>> "%RUNNER%" echo }
>> "%RUNNER%" echo.
>> "%RUNNER%" echo FALLBACK_STDLIB = {
>> "%RUNNER%" echo     "abc", "argparse", "ast", "asyncio", "base64", "collections", "contextlib",
>> "%RUNNER%" echo     "copy", "csv", "dataclasses", "datetime", "decimal", "enum", "functools",
>> "%RUNNER%" echo     "glob", "hashlib", "importlib", "inspect", "io", "itertools", "json",
>> "%RUNNER%" echo     "logging", "math", "multiprocessing", "os", "pathlib", "pickle", "platform",
>> "%RUNNER%" echo     "pprint", "random", "re", "shutil", "signal", "sqlite3", "statistics",
>> "%RUNNER%" echo     "string", "subprocess", "sys", "tempfile", "threading", "time", "typing",
>> "%RUNNER%" echo     "unittest", "urllib", "uuid", "warnings", "zipfile",
>> "%RUNNER%" echo }
>> "%RUNNER%" echo.
>> "%RUNNER%" echo def main():
>> "%RUNNER%" echo     root = Path.cwd()
>> "%RUNNER%" echo     raw_stdlib = getattr(sys, "stdlib_module_names", None)
>> "%RUNNER%" echo     stdlib = set(raw_stdlib) if raw_stdlib else set(FALLBACK_STDLIB)
>> "%RUNNER%" echo     stdlib.add("__future__")
>> "%RUNNER%" echo.
>> "%RUNNER%" echo     local_mods = set()
>> "%RUNNER%" echo     for p in root.iterdir():
>> "%RUNNER%" echo         if p.name in SKIP_DIRS or p.name.startswith("."):
>> "%RUNNER%" echo             continue
>> "%RUNNER%" echo         if p.is_file() and p.suffix == ".py":
>> "%RUNNER%" echo             local_mods.add(p.stem)
>> "%RUNNER%" echo         elif p.is_dir():
>> "%RUNNER%" echo             local_mods.add(p.name)
>> "%RUNNER%" echo.
>> "%RUNNER%" echo     py_files = [
>> "%RUNNER%" echo         p for p in root.rglob("*.py")
>> "%RUNNER%" echo         if not any(part in SKIP_DIRS for part in p.parts)
>> "%RUNNER%" echo     ]
>> "%RUNNER%" echo     print("[*] Da tim thay", len(py_files), "file .py trong repo.")
>> "%RUNNER%" echo.
>> "%RUNNER%" echo     imports = set()
>> "%RUNNER%" echo     for py_file in py_files:
>> "%RUNNER%" echo         try:
>> "%RUNNER%" echo             tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
>> "%RUNNER%" echo         except SyntaxError as e:
>> "%RUNNER%" echo             print("[!] Bo qua", py_file.name, ":", e)
>> "%RUNNER%" echo             continue
>> "%RUNNER%" echo         for node in ast.walk(tree):
>> "%RUNNER%" echo             if isinstance(node, ast.Import):
>> "%RUNNER%" echo                 for alias in node.names:
>> "%RUNNER%" echo                     top = alias.name.split(".")[0].strip()
>> "%RUNNER%" echo                     if top:
>> "%RUNNER%" echo                         imports.add(top)
>> "%RUNNER%" echo             elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
>> "%RUNNER%" echo                 top = node.module.split(".")[0].strip()
>> "%RUNNER%" echo                 if top:
>> "%RUNNER%" echo                     imports.add(top)
>> "%RUNNER%" echo.
>> "%RUNNER%" echo     third_party = sorted(m for m in imports if m and m not in stdlib and m not in local_mods)
>> "%RUNNER%" echo     missing = []
>> "%RUNNER%" echo     for mod in third_party:
>> "%RUNNER%" echo         if mod not in sys.builtin_module_names and importlib.util.find_spec(mod) is None:
>> "%RUNNER%" echo             pkg = IMPORT_MAP.get(mod, mod)
>> "%RUNNER%" echo             if pkg not in missing:
>> "%RUNNER%" echo                 missing.append(pkg)
>> "%RUNNER%" echo.
>> "%RUNNER%" echo     if importlib.util.find_spec("streamlit") is None and "streamlit" not in missing:
>> "%RUNNER%" echo         missing.append("streamlit")
>> "%RUNNER%" echo.
>> "%RUNNER%" echo     if missing:
>> "%RUNNER%" echo         print("[*] Dang cai ban moi nhat cho tung goi:", ", ".join(missing))
>> "%RUNNER%" echo         for pkg in missing:
>> "%RUNNER%" echo             print("[*] Cai dat:", pkg)
>> "%RUNNER%" echo             res = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", pkg])
>> "%RUNNER%" echo             if res.returncode != 0:
>> "%RUNNER%" echo                 print("[!] Khong cai duoc goi (co the chua ho tro Python hien tai):", pkg)
>> "%RUNNER%" echo     else:
>> "%RUNNER%" echo         print("[OK] Tat ca thu vien can thiet da co san.")
>> "%RUNNER%" echo.
>> "%RUNNER%" echo     app_path = root / "app.py"
>> "%RUNNER%" echo     if not app_path.exists():
>> "%RUNNER%" echo         print("[LOI] Khong tim thay", app_path)
>> "%RUNNER%" echo         return 1
>> "%RUNNER%" echo.
>> "%RUNNER%" echo     print("[*] Dang khoi chay Streamlit app.py...")
>> "%RUNNER%" echo     return subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], cwd=str(root)).returncode
>> "%RUNNER%" echo.
>> "%RUNNER%" echo if __name__ == "__main__":
>> "%RUNNER%" echo     try:
>> "%RUNNER%" echo         sys.exit(main())
>> "%RUNNER%" echo     except Exception as exc:
>> "%RUNNER%" echo         print("[LOI]", exc)
>> "%RUNNER%" echo         sys.exit(1)

"%PY_EXE%" "%RUNNER%"
set "EXIT_CODE=%errorlevel%"
if exist "%RUNNER%" del /q "%RUNNER%" >nul 2>&1

echo.
echo [*] Ket thuc voi ma thoat: %EXIT_CODE%
pause
exit /b %EXIT_CODE%