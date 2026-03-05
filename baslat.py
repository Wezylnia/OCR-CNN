"""
baslat.py — Kaggle OCR Egitim Baslatici
Calistir: !python /kaggle/input/models/serhatananana/model2/tensorflow2/default/1/OCR-CNN-main/baslat.py
"""

import subprocess, sys, os, shutil
from pathlib import Path

# ── Sabit yollar (hicbir yerde rglob/glob kullanilmiyor) ──────
SRC   = Path("/kaggle/input/models/serhatananana/model2/tensorflow2/default/1/OCR-CNN-main")
WORK  = Path("/kaggle/working/ocr")
DATA_ROOT  = "/kaggle/input/datasets/garvitchaudhary/mjsynth/mnt/ramdisk/max/90kDICT32px"
TRAIN_JSON = "/kaggle/working/mjsynth_full.json"   # tum veri — baslat.py uretir
SAVE_DIR   = "/kaggle/working/checkpoints"
CHECKPOINT = str(SRC / "checkpoints" / "stage1_mjsynth" / "final_model.pth")

def banner(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)

def ok(msg):   print(f"  [OK] {msg}", flush=True)
def warn(msg): print(f"  [!!] {msg}", flush=True)
def die(msg):
    print(f"\n  [HATA] {msg}", flush=True)
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 1. ZORUNLU KLASOR/DOSYA KONTROLLERI
# ══════════════════════════════════════════════════════════════
banner("1/6  KONTROL")

if not SRC.exists():
    die(f"Kod paketi bulunamadi:\n    {SRC}")
ok(f"Kod paketi: {SRC}")

if not Path(DATA_ROOT).exists():
    die(f"MJSynth dataset yok:\n    {DATA_ROOT}\n    Notebook'ta 'garvitchaudhary/mjsynth' eklendi mi?")
ok(f"MJSynth dataset: {DATA_ROOT}")

# ══════════════════════════════════════════════════════════════
# 2. FONT KURULUMU
# ══════════════════════════════════════════════════════════════
banner("2/6  FONT KURULUMU")

font_pkgs = [
    "fonts-liberation",
    "fonts-dejavu-core",
    "fonts-freefont-ttf",
    "fonts-urw-base35",
    "fonts-open-sans",
    "fonts-noto",
]
r = subprocess.run(
    ["apt-get", "install", "-y", "-q"] + font_pkgs,
    capture_output=True, text=True
)
if r.returncode == 0:
    # 2 seviye iterdir ile say (rglob degil)
    n = 0
    font_base = Path("/usr/share/fonts")
    for d1 in font_base.iterdir():
        if d1.is_dir():
            for d2 in d1.iterdir():
                if d2.is_dir():
                    n += sum(1 for f in d2.iterdir() if f.suffix.lower() in (".ttf", ".otf"))
                elif d2.suffix.lower() in (".ttf", ".otf"):
                    n += 1
        elif d1.suffix.lower() in (".ttf", ".otf"):
            n += 1
    ok(f"Sistem fontu: {n} (/usr/share/fonts, 2-seviye)")
else:
    warn(f"apt-get basarisiz (devam ediliyor): {r.stderr[-200:]}")

# ══════════════════════════════════════════════════════════════
# 3. PYTHON PAKET KURULUMU
# ══════════════════════════════════════════════════════════════
banner("3/6  PAKET KURULUMU")

# Kaggle'da zaten var: torch, torchvision, numpy, cv2, Pillow, scipy, tqdm, PyYAML
# Eksik olanlar:
missing_pkgs = [
    "shapely>=2.0.0",
    "pyclipper>=1.3.0",
    "pyspellchecker>=0.7.0",
    "symspellpy>=7.7.1",
]
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q"] + missing_pkgs,
    capture_output=True, text=True
)
if r.returncode == 0:
    ok("shapely  pyclipper  pyspellchecker  symspellpy")
else:
    warn(f"Paket kurulum uyarisi: {r.stderr[-300:]}")

# ══════════════════════════════════════════════════════════════
# 4. ÇALIŞMA DİZİNİ KURULUMU
# ══════════════════════════════════════════════════════════════
banner("4/6  DIZIN KURULUMU")

WORK.mkdir(parents=True, exist_ok=True)

# Kopyalanacak klasor ve dosyalar (sadece bunlar — dataset dizinine DOKUNULMAZ)
COPY_ITEMS = [
    "ocr_engine",
    "training",
    "data",
    "config.yaml",
]

for item in COPY_ITEMS:
    src_path = SRC / item
    dst_path = WORK / item
    if not src_path.exists():
        warn(f"Atlanди: {item} (bulunamadi)")
        continue
    # Onceki kopyayi temizle
    if dst_path.exists():
        if dst_path.is_dir():
            shutil.rmtree(dst_path)
        else:
            dst_path.unlink()
    # Kopyala
    if src_path.is_dir():
        shutil.copytree(src_path, dst_path)
    else:
        shutil.copy2(src_path, dst_path)
    ok(f"Kopyalandi: {item}")

os.chdir(WORK)
sys.path.insert(0, str(WORK))
ok(f"Calisma dizini: {WORK}")

# ══════════════════════════════════════════════════════════════
# 5. TAM VERİ SETİ ANNOTATION ÜRET (8.9M)
# ══════════════════════════════════════════════════════════════
banner("5/6  ANNOTATION URETIMI (tum veri)")

import json as _json

if Path(TRAIN_JSON).exists():
    existing = _json.load(open(TRAIN_JSON))
    ok(f"Annotation zaten var: {len(existing):,} ornek — atlaniyor")
    del existing
else:
    ok("Annotation uretiliyor (~8.9M dosya, 2-seviye tarama)...")
    _root   = Path(DATA_ROOT)
    samples = []
    _total  = 0

    # Yapi: DATA_ROOT / KLASOR / ALTKLASOR / <num>_<KELIME>_<num>.jpg
    # Sadece 2 seviye iterdir() — rglob yok, dataset icinde takilmaz
    for _lvl1 in sorted(_root.iterdir()):
        if not _lvl1.is_dir():
            continue
        for _lvl2 in _lvl1.iterdir():
            if not _lvl2.is_dir():
                continue
            for _img in _lvl2.iterdir():
                if _img.suffix.lower() != ".jpg":
                    continue
                # Etiket: dosya adi parcalari arasi (115_Lube_45484 -> Lube)
                parts = _img.stem.split("_")
                if len(parts) < 3:
                    continue
                label = "_".join(parts[1:-1])
                # Goreli yol: KLASOR/ALTKLASOR/dosya.jpg
                rel = f"{_lvl1.name}/{_lvl2.name}/{_img.name}"
                samples.append({"image_path": rel, "text": label})
            _total += 1
        if _total % 200 == 0 and _total > 0:
            print(f"    {_total} klasor islendi, {len(samples):,} ornek...",
                  flush=True)

    ok(f"Toplam {len(samples):,} ornek bulundu")
    with open(TRAIN_JSON, "w", encoding="utf-8") as _f:
        _json.dump(samples, _f, ensure_ascii=False)
    ok(f"Kaydedildi: {TRAIN_JSON}  "
       f"({Path(TRAIN_JSON).stat().st_size/1024**2:.1f} MB)")
    del samples

# ══════════════════════════════════════════════════════════════
# 6. EĞİTİM
# ══════════════════════════════════════════════════════════════
banner("6/6  EGITIM")

import torch

# GPU bilgisi
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram  = props.total_memory / 1024**3
    ok(f"GPU: {props.name}  |  VRAM: {vram:.1f} GB")
    BATCH = 512 if vram >= 15 else 384 if vram >= 13 else 256
    DEVICE = "cuda"
else:
    warn("CUDA bulunamadi — CPU modu (cok yavas)")
    vram  = 0
    BATCH = 32
    DEVICE = "cpu"

ok(f"Batch size: {BATCH}")

# Checkpoint
if Path(CHECKPOINT).exists():
    resume_args = ["--resume", CHECKPOINT]
    ok(f"Checkpoint: final_model.pth  ({Path(CHECKPOINT).stat().st_size/1024**2:.1f} MB)")
else:
    resume_args = []
    warn(f"Checkpoint bulunamadi, sifirdan baslanıyor:\n    {CHECKPOINT}")

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

TRAIN_SCRIPT = str(WORK / "training" / "train_recognition_mjsynth.py")

cmd = [
    sys.executable, TRAIN_SCRIPT,
    "--data_root",  DATA_ROOT,
    "--train_json", TRAIN_JSON,
    "--config",     str(WORK / "config.yaml"),
    "--epochs",     "25",
    "--batch_size", str(BATCH),
    "--augment",
    "--val_split",  "0.02",
    "--save_dir",   SAVE_DIR,
    "--device",     DEVICE,
    "--num_workers", "2",   # Kaggle: sadece 2 CPU core var
    "--quiet",      # Kaggle: tqdm kapali, log tasması önlenir
] + resume_args

print(f"\n  Komut: {' '.join(cmd[:6])} ...\n" + "="*60 + "\n", flush=True)

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    cwd=str(WORK),
    env={**os.environ, "PYTHONPATH": str(WORK)},
)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()

print()
if proc.returncode == 0:
    banner("EGITIM TAMAMLANDI")
    for f in sorted(Path(SAVE_DIR).iterdir()):
        if f.suffix == ".pth":
            print(f"  {f.name:45s} {f.stat().st_size/1024**2:.1f} MB")
else:
    banner(f"EGITIM HATA ILE SONLANDI (kod: {proc.returncode})")
    sys.exit(proc.returncode)