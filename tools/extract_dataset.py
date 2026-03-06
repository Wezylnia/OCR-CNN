"""
.tar.gz dosyasini cikartan yardimci script
"""

import tarfile
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_tar_gz(tar_path: str, output_dir: str):
    """
    .tar.gz dosyasini cikar
    
    Args:
        tar_path: .tar.gz dosya yolu
        output_dir: Cikartilacak konum
    """
    tar_path = Path(tar_path)
    output_dir = Path(output_dir)
    
    if not tar_path.exists():
        print(f"❌ Dosya bulunamadi: {tar_path}")
        return
    
    print(f"📦 Dosya: {tar_path}")
    print(f"📂 Cikartilacak konum: {output_dir}")
    print(f"💾 Dosya boyutu: {tar_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")
    print()
    
    # Klasor olustur
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("⏳ Cikariliyor... (Bu işlem birkaç dakika sürebilir)")
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Toplam dosya sayisi
            members = tar.getmembers()
            total = len(members)
            
            print(f"📊 Toplam {total:,} dosya bulundu")
            print()

            # Güvenlik: path traversal saldırısına karşı her yolu doğrula
            abs_output = output_dir.resolve()
            for member in members:
                target = (abs_output / member.name).resolve()
                if not str(target).startswith(str(abs_output)):
                    raise ValueError(f"Güvensiz arşiv yolu algılandı: {member.name!r}")

            tar.extractall(path=output_dir, members=tqdm(members, desc="Extracting", unit="file"))
        
        print()
        print("✅ Basariyla cikarildi!")
        print(f"📂 Konum: {output_dir.absolute()}")
        
        # Icerik kontrol
        print("\n📋 Cikartilan dosyalar:")
        for item in output_dir.iterdir():
            if item.is_dir():
                file_count = len(list(item.rglob('*')))
                print(f"  📁 {item.name}/ ({file_count} dosya)")
            else:
                print(f"  📄 {item.name}")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        print("\nManuel olarak deneyebilirsiniz:")
        print(f"  tar -xzf {tar_path} -C {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Tar.gz dosyasini cikar')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Giris .tar.gz dosyasi')
    parser.add_argument('--output', '-o', type=str, default='C:/Datasets',
                        help='Cikartilacak konum')
    
    args = parser.parse_args()
    
    extract_tar_gz(args.input, args.output)


if __name__ == '__main__':
    main()
