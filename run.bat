@echo off
cd /d "%~dp0"
title TSNE Customer Clustering

:: 1. Tim Python kha dung
set "PY_EXE="
for /f "delims=" %%i in ('where py 2^>nul') do if not defined PY_EXE set "PY_EXE=py -3"
if not defined PY_EXE for /f "delims=" %%i in ('where python 2^>nul') do if not defined PY_EXE set "PY_EXE=python"
if not defined PY_EXE for /f "delims=" %%i in ('where python3 2^>nul') do if not defined PY_EXE set "PY_EXE=python3"

if not defined PY_EXE (
    echo [LOI] Khong tim thay Python tren may.
    pause
    exit /b 1
)

echo [*] Dang dung: %PY_EXE%
%PY_EXE% --version

:: 2. Kiem tra co GPU NVIDIA hay khong
echo [*] Buoc 1: Kiem tra GPU NVIDIA...
set "HAS_GPU=0"
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set "HAS_GPU=1"
    echo [OK] Phat hien GPU NVIDIA.
) else (
    echo [-] Khong phat hien GPU NVIDIA.
)

:: 3. Kiem tra PyTorch tuong ung voi trang thai GPU
echo [*] Buoc 2: Kiem tra PyTorch tuong ung voi GPU...
set "TORCH_OK=0"
if "%HAS_GPU%"=="1" (
    %PY_EXE% -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if not errorlevel 1 set "TORCH_OK=1"
) else (
    %PY_EXE% -c "import torch" >nul 2>&1
    if not errorlevel 1 set "TORCH_OK=1"
)

if "%TORCH_OK%"=="1" (
    echo [OK] PyTorch da san sang va phu hop voi trang thai GPU.
    goto CHECK_OPENTSNE
)

:: 4. Cai PyTorch tuong ung neu chua co hoac chua khop GPU
echo [*] Buoc 3: Chua co PyTorch phu hop. Dang tu dong cai dat...
if "%HAS_GPU%"=="1" (
    echo [*] Dang cai PyTorch ban CUDA...
    %PY_EXE% -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 %PY_EXE% -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    if errorlevel 1 %PY_EXE% -m pip install --upgrade torch torchvision torchaudio
) else (
    echo [*] Dang cai PyTorch ban CPU...
    %PY_EXE% -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
)

:: 5. Kiem tra va tu dong cai openTSNE neu chua co
:CHECK_OPENTSNE
echo [*] Buoc 4: Kiem tra thu vien openTSNE...
%PY_EXE% -c "import importlib.util, subprocess, sys; ok = importlib.util.find_spec('openTSNE') is not None; print('[OK] openTSNE da co san.') if ok else print('[*] Chua co openTSNE. Dang tu dong cai ban moi nhat...'); sys.exit(0 if ok else subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'openTSNE']).returncode)"
if errorlevel 1 (
    echo [!] Khong cai duoc openTSNE tren phien ban Python hien tai. Neu can dung openTSNE, hay thu Python 3.12 hoac 3.13.
)

:: 6. Quet cac thu vien con thieu khac trong repo va khoi chay Streamlit
echo [*] Buoc 5: Quet cac file .py trong repo va khoi chay Streamlit...
%PY_EXE% -c "import ast,importlib.util,subprocess,sys;from pathlib import Path;REQ=['numpy','pandas','scipy','scikit-learn','torch','umap-learn','hdbscan','openTSNE','streamlit'];MAP={'bs4':'beautifulsoup4','cv2':'opencv-python','dotenv':'python-dotenv','fitz':'pymupdf','PIL':'Pillow','sklearn':'scikit-learn','skimage':'scikit-image','umap':'umap-learn','yaml':'PyYAML'};SKIP_D={'.git','.venv','venv','env','__pycache__','.idea','.vscode','node_modules','build','dist'};root=Path.cwd();stdlib=(set(getattr(sys,'stdlib_module_names',None) or ['os','sys','re','json','math','time','datetime','pathlib','subprocess','typing','importlib','inspect','warnings','ast','shutil','glob','csv','pickle','random','collections','itertools','functools','urllib','io','copy','hashlib','logging','threading','multiprocessing','platform','tempfile','unittest','argparse','dataclasses','enum','statistics','string','uuid','zipfile']))|{'__future__'};locals_m={p.stem for p in root.iterdir() if p.is_file() and p.suffix=='.py'}|{p.name for p in root.iterdir() if p.is_dir()};py_files=[p for p in root.rglob('*.py') if not any(x in SKIP_D for x in p.parts)];print('[*] Tim thay',len(py_files),'file .py');imports=set();[imports.update({a.name.split('.')[0] for n in ast.walk(ast.parse(f.read_text(encoding='utf-8',errors='ignore'))) if isinstance(n,ast.Import) for a in n.names}|{n.module.split('.')[0] for n in ast.walk(ast.parse(f.read_text(encoding='utf-8',errors='ignore'))) if isinstance(n,ast.ImportFrom) and n.level==0 and n.module}) for f in py_files];missing=[];[missing.append(p) for p in REQ if importlib.util.find_spec('openTSNE' if p=='openTSNE' else p.split('-')[0]) is None and p not in missing];[missing.append(MAP.get(m,m)) for m in sorted(imports) if m and m not in stdlib and m not in locals_m and m.lower()!='cuml' and m not in sys.builtin_module_names and importlib.util.find_spec(m) is None and MAP.get(m,m) not in missing];print('[*] Can cai them:',missing) if missing else print('[OK] Du thu vien.');[subprocess.run([sys.executable,'-m','pip','install','--upgrade',p]) for p in missing];sys.exit(subprocess.run([sys.executable,'-m','streamlit','run',str(root/'app.py')],cwd=str(root)).returncode)"

echo.
echo [*] Tien trinh ket thuc. Nhan phim bat ky de dong cua so.
pause