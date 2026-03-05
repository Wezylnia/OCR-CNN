"""
hazirla_ve_egit.py - Yerel makine icin tam annotation + egitim
RTX 3050 6GB + i5-13500H
"""
import subprocess, sys, os, json, time
from pathlib import Path
from datetime import timedelta

ROOT    = Path(__file__).parent
DATA_ROOT  = ROOT / "DataSets" / "mnt" / "ramdisk" / "max" / "90kDICT32px"
FULL_JSON  = ROOT / "data" / "mjsynth_full.json"
SAVE_DIR   = ROOT / "checkpoints" / "stage1_full"

def banner(msg): print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)
def ok(msg):     print(f"  [OK] {msg}", flush=True)
def warn(msg):   print(f"  [!!] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════
# 1. ANNOTATION URET (ilk calistirmada ~5-10 dk)
# ══════════════════════════════════════════════════════════════
banner("1/2  ANNOTATION KONTROLU")

if FULL_JSON.exists():
    existing = json.load(open(FULL_JSON))
    ok(f"mjsynth_full.json zaten var: {len(existing):,} ornek — atlaniyor")
    del existing
else:
    ok(f"mjsynth_full.json uretiliyor (3548 klasor × 4 alt klasor)...")
    t0 = time.time()
    samples = []
    lvl1_count = 0

    # Yapi: DATA_ROOT / KLASOR / ALTKLASOR / img.jpg
    # 2 seviye iterdir — rglob yok, takılma riski yok
    for lvl1 in sorted(DATA_ROOT.iterdir()):
        if not lvl1.is_dir():
            continue
        for lvl2 in sorted(lvl1.iterdir()):
            if not lvl2.is_dir():
                continue
            for img in lvl2.iterdir():
                if img.suffix.lower() != ".jpg":
                    continue
                parts = img.stem.split("_")
                if len(parts) < 3:
                    continue
                label = "_".join(parts[1:-1])
                rel = f"{lvl1.name}/{lvl2.name}/{img.name}"
                samples.append({"image_path": rel, "text": label})
        lvl1_count += 1
        if lvl1_count % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / lvl1_count * (3548 - lvl1_count)
            print(f"    {lvl1_count}/3548 klasor | {len(samples):,} ornek | "
                  f"ETA: {timedelta(seconds=int(eta))}", flush=True)

    ok(f"Toplam {len(samples):,} ornek bulundu ({time.time()-t0:.0f} sn)")
    FULL_JSON.parent.mkdir(exist_ok=True)
    with open(FULL_JSON, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    size_mb = FULL_JSON.stat().st_size / 1024**2
    ok(f"Kaydedildi: {FULL_JSON.name}  ({size_mb:.0f} MB)")
    del samples

# ══════════════════════════════════════════════════════════════
# 2. SHARD EGiTiM
#    8.9M / 1.3M ≈ 7 shard/round × 3 round = 21 shard
#    Her shard ~26 dk  →  toplam ~9 saat
#    Ctrl+C ile durdur, tekrar calistirinca kaldigi yerden devam
# ══════════════════════════════════════════════════════════════
banner("2/2  SHARD EGITIM")

import random as _random

# ── Ayarlar ──────────────────────────────────────────────────
CHUNK_SIZE  = 1_300_000  # Her shard'da kac ornek  (~26 dk)
ROUNDS      = 2          # Kac tam gecis — 2 round = ~5-6 saat, 3 round = ~8-9 saat
SEED_BASE   = 42         # Deterministik shuffle — crash sonrasi ayni plan
BATCH       = 96         # RTX 3050 6GB + AMP
NUM_WORKERS = 4          # i5-13500H icin optimal
VAL_SPLIT   = 0.01       # Shard basina %1 val (~13K ornek)

FULL_CKPT_DIR    = ROOT / "checkpoints" / "stage1_full"
MJSYNTH_CKPT_DIR = ROOT / "checkpoints" / "stage1_mjsynth"
SAVE_DIR         = ROOT / "checkpoints" / "stage1_full"
SHARD_JSON       = ROOT / "data" / "shard_active.json"
STATE_FILE       = ROOT / "data" / "shard_state.json"   # base_epoch kalici kaydi

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Full anasyonu RAM'e al (bir sefer, ~750MB) ────────────────
ok("mjsynth_full.json yukleniyor (RAM ~750MB)...")
t_load = time.time()
all_samples = json.load(open(FULL_JSON, encoding="utf-8"))
n_total     = len(all_samples)
ok(f"{n_total:,} ornek yuklendi ({time.time()-t_load:.1f} sn)")

# ── Shard plani ───────────────────────────────────────────────
n_srp        = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE   # rounds basina shard sayisi
total_shards = ROUNDS * n_srp
ok(f"Shard plani : {ROUNDS} round × {n_srp} shard = {total_shards} toplam shard")
ok(f"Her shard   : ~{CHUNK_SIZE:,} ornek  |  tahmini ~26-32 dk")
ok(f"Toplam sure : ~{total_shards * 26 // 60}-{total_shards * 32 // 60} saat")

# ── Yardimci fonksiyonlar ─────────────────────────────────────
def read_ckpt_meta(pth_path):
    """Checkpoint'tan epoch ve best_val_acc degerlerini oku."""
    import torch as _t
    ck = _t.load(pth_path, map_location="cpu", weights_only=False)
    ep  = ck.get("epoch", 0)
    acc = ck.get("best_val_acc", 0.0)
    del ck, _t
    return ep, acc

def read_epoch(pth_path):
    """Checkpoint'taki epoch degerini oku."""
    return read_ckpt_meta(pth_path)[0]

def find_resume():
    """En iyi checkpoint'u sec:
       1. stage1_full/best_model.pth  — val acc'i en yuksek anlık agirliklar
       2. stage1_full/checkpoint_epoch_*.pth — best yoksa en yüksek best_val_acc'li
       3. stage1_mjsynth fallback
    """
    # 1) best_model.pth varsa direkt kullan — sadece val acc iyilesince uzerine yazilir
    best_pth = FULL_CKPT_DIR / "best_model.pth"
    if best_pth.exists():
        try:
            ep, acc = read_ckpt_meta(str(best_pth))
            ok(f"Resume secildi: best_model.pth  "
               f"(epoch={ep}, best_val_acc={acc*100:.2f}%)")
            return str(best_pth)
        except Exception:
            pass
    # 2) best_model.pth yoksa en yuksek best_val_acc'li checkpoint_epoch_*.pth
    candidates = []
    if FULL_CKPT_DIR.exists():
        for p in FULL_CKPT_DIR.glob("checkpoint_epoch_*.pth"):
            try:
                ep, acc = read_ckpt_meta(str(p))
                candidates.append((acc, ep, str(p)))
            except Exception:
                pass
    if candidates:
        # Esitlik durumunda en dusuk epoch: acc'a ilk ulasan = en temiz model
        candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        best_acc, best_ep, best_path = candidates[0]
        ok(f"Resume secildi: {Path(best_path).name}  "
           f"(epoch={best_ep}, best_val_acc={best_acc*100:.2f}%)")
        return best_path
    # 3) Fallback: stage1_mjsynth
    for name in ["final_model.pth", "best_model.pth"]:
        p = MJSYNTH_CKPT_DIR / name
        if p.exists():
            return str(p)
    return None

# ── base_epoch: ilk seed checkpoint'un epoch degeri ──────────
#    shard_idx = current_epoch - base_epoch
if STATE_FILE.exists():
    state      = json.load(open(STATE_FILE))
    base_epoch = state["base_epoch"]
    ok(f"Shard state okundu — base_epoch={base_epoch}")
else:
    seed_ckpt = find_resume()
    if seed_ckpt:
        base_epoch = read_epoch(seed_ckpt)
        ok(f"Ilk calistirma — base_epoch={base_epoch}  ({Path(seed_ckpt).name})")
    else:
        base_epoch = 0
        ok("Ilk calistirma — sifirdan baslaniyor (base_epoch=0)")
    json.dump({"base_epoch": base_epoch}, open(STATE_FILE, "w"))

# ── Kac shard tamamlandi? ─────────────────────────────────────
current_resume = find_resume()
current_epoch  = read_epoch(current_resume) if current_resume else base_epoch
start_shard    = current_epoch - base_epoch

if start_shard >= total_shards:
    ok(f"Tum {total_shards} shard tamamlandi!")
    ok(f"Daha fazla egitim icin ROUNDS degerini artir ve STATE_FILE'i sil.")
    sys.exit(0)

ok(f"Baslangic    : shard {start_shard + 1}/{total_shards}  "
   f"(epoch {current_epoch} → hedef {base_epoch + total_shards})")

# ── Ana shard dongusu ─────────────────────────────────────────
shard_times = []

for global_shard in range(start_shard, total_shards):
    round_idx      = global_shard // n_srp
    shard_in_round = global_shard % n_srp
    target_epoch   = base_epoch + global_shard + 1

    banner(f"SHARD {global_shard + 1}/{total_shards}  |  "
           f"Round {round_idx + 1}/{ROUNDS}  |  "
           f"Epoch hedef: {target_epoch}")

    # Round'a ozel deterministik shuffle
    rng     = _random.Random(SEED_BASE + round_idx)
    indices = list(range(n_total))
    rng.shuffle(indices)

    # Bu shard'in dilimleri
    s_start        = shard_in_round * CHUNK_SIZE
    s_end          = min(s_start + CHUNK_SIZE, n_total)
    shard_samples  = [all_samples[i] for i in indices[s_start:s_end]]

    ok(f"Shard ornekleri: {len(shard_samples):,}  "
       f"({s_start:,}–{s_end:,} / {n_total:,})")

    t_write = time.time()
    with open(SHARD_JSON, "w", encoding="utf-8") as f:
        json.dump(shard_samples, f, ensure_ascii=False)
    ok(f"shard_active.json yazildi  "
       f"({SHARD_JSON.stat().st_size / 1024**2:.0f} MB, {time.time()-t_write:.1f} sn)")
    del shard_samples  # RAM geri al

    resume_now = find_resume()
    cmd = [
        sys.executable, str(ROOT / "training" / "train_recognition_mjsynth.py"),
        "--data_root",   str(DATA_ROOT),
        "--train_json",  str(SHARD_JSON),
        "--config",      str(ROOT / "config.yaml"),
        "--epochs",      str(target_epoch),
        "--batch_size",  str(BATCH),
        "--val_split",   str(VAL_SPLIT),
        "--save_dir",    str(SAVE_DIR),
        "--device",      "cuda",
        "--num_workers", str(NUM_WORKERS),
        "--lr",          "1e-5",   # 1e-4 plateau'ya girdi -> 1e-5 ile ince ayar
        "--augment",               # Shard 5+ augmentation: blur/noise/rotate -> o/e karisikligini azaltir
        "--quiet",
    ]
    if resume_now:
        resume_ep = read_epoch(resume_now)
        cmd += ["--resume", resume_now]
        ok(f"Resume: {Path(resume_now).name}  (epoch {resume_ep} → {target_epoch})")
    else:
        ok("Resume yok — sifirdan basliyor")

    t_shard = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    elapsed = time.time() - t_shard
    shard_times.append(elapsed)

    if proc.returncode != 0:
        banner(f"EGITIM HATA — Shard {global_shard + 1} (kod: {proc.returncode})")
        sys.exit(proc.returncode)

    avg_sec     = sum(shard_times) / len(shard_times)
    shards_left = total_shards - global_shard - 1
    ok(f"Shard {global_shard + 1} tamamlandi  ({timedelta(seconds=int(elapsed))})")
    if shards_left:
        ok(f"Kalan: {shards_left} shard  |  "
           f"Tahmini: ~{timedelta(seconds=int(avg_sec * shards_left))}")

banner("TUM SHARDLAR TAMAMLANDI")
for f in sorted(SAVE_DIR.glob("*.pth")):
    print(f"  {f.name:45s} {f.stat().st_size/1024**2:.1f} MB")