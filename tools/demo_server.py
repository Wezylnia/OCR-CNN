"""
Basit Recognition Test Server
Sadece kelimeleri okur (detection yok, tek kelime/satir icin)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, render_template_string
import torch
import cv2
import numpy as np
from PIL import Image
import io
import argparse

from ocr_engine.recognition.vocab import Vocabulary
from ocr_engine.recognition.model import CRNN
from ocr_engine.recognition.decoder import CTCDecoder

app = Flask(__name__)

# Global model
model = None
vocab = None
decoder = None
device = None

def load_model(checkpoint: str = "checkpoints/realworld_finetune/best_model.pth",
               device_str: str = "cuda"):
    """Modeli yukle"""
    global model, vocab, decoder, device

    print("Model yukleniyor...")

    # Device
    device = torch.device(device_str if device_str != "cuda" or torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Vocabulary
    vocab = Vocabulary()
    print(f"Vocabulary: {vocab.size} karakter")
    
    # Model
    model = CRNN(
        num_classes=vocab.size,
        input_channels=1,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
        encoder_type='vgg'
    ).to(device)
    
    # Checkpoint yukle
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        print(f"Checkpoint yukleniyor: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Epoch {checkpoint['epoch']} yuklendi")
    else:
        print("UYARI: Checkpoint bulunamadi, rastgele agirliklar kullaniliyor!")
    
    model.eval()
    
    # Decoder
    decoder = CTCDecoder(vocab)
    
    print("Model hazir!")

def preprocess_image(image_bytes):
    """Gorseli model icin hazirla"""
    # PIL Image
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('L')  # Grayscale
    
    # Numpy
    img_np = np.array(img)
    
    # Resize (32 height, width oranli)
    h, w = img_np.shape
    target_h = 32
    target_w = int(w * (target_h / h))
    target_w = max(target_w, 32)  # Minimum width
    target_w = min(target_w, 256)  # Maximum width
    
    img_resized = cv2.resize(img_np, (target_w, target_h))
    
    # Pad to 256
    if target_w < 256:
        pad_w = 256 - target_w
        img_resized = np.pad(img_resized, ((0, 0), (0, pad_w)), mode='constant', constant_values=255)
    
    # Normalize
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5  # [-1, 1]
    
    # Add batch and channel dims
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 32, 256]
    
    return img_tensor

@app.route('/')
def index():
    """Ana sayfa"""
    html = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Recognition Test</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 800px;
            width: 100%;
            padding: 40px;
        }
        
        h1 {
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9ff;
        }
        
        .upload-area:hover {
            background: #e8ebff;
            border-color: #764ba2;
        }
        
        .upload-area.drag-over {
            background: #e8ebff;
            border-color: #764ba2;
            transform: scale(1.02);
        }
        
        input[type="file"] {
            display: none;
        }
        
        .upload-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        .upload-text {
            font-size: 1.2em;
            color: #667eea;
            font-weight: bold;
        }
        
        .upload-hint {
            color: #999;
            margin-top: 10px;
            font-size: 0.9em;
        }
        
        .preview-container {
            margin: 20px 0;
            text-align: center;
        }
        
        #preview {
            max-width: 100%;
            max-height: 200px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .result-container {
            background: #f8f9ff;
            border-radius: 15px;
            padding: 30px;
            margin-top: 30px;
            display: none;
        }
        
        .result-container.show {
            display: block;
            animation: slideIn 0.3s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result-label {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .result-text {
            background: white;
            padding: 20px;
            border-radius: 10px;
            font-size: 1.5em;
            color: #333;
            word-break: break-word;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }
        
        .error.show {
            display: block;
        }
        
        .info-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-top: 20px;
            border-radius: 5px;
            font-size: 0.9em;
            color: #856404;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔤 OCR Test</h1>
        <p class="subtitle">Kelime/Metin Tanıma (CRNN Model - Epoch 3)</p>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📷</div>
            <div class="upload-text">Görsel Seç veya Sürükle</div>
            <div class="upload-hint">PNG, JPG, JPEG (Tek kelime/satır için en iyi)</div>
            <input type="file" id="fileInput" accept="image/*">
        </div>
        
        <div class="preview-container" id="previewContainer" style="display: none;">
            <img id="preview" alt="Preview">
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>Model çalışıyor...</div>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="result-container" id="resultContainer">
            <div class="result-label">📝 Tanınan Metin:</div>
            <div class="result-text" id="resultText"></div>
        </div>
        
        <div class="info-box">
            <strong>💡 İpucu:</strong> Bu model sadece <strong>metin tanıma</strong> yapıyor. 
            En iyi sonuç için tek kelime veya temiz bir satır içeren görsel kullanın.
        </div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const previewContainer = document.getElementById('previewContainer');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const resultContainer = document.getElementById('resultContainer');
        const resultText = document.getElementById('resultText');
        
        // Upload area click
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
        
        function handleFile(file) {
            // Preview
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                previewContainer.style.display = 'block';
            };
            reader.readAsDataURL(file);
            
            // Upload
            uploadImage(file);
        }
        
        async function uploadImage(file) {
            // Hide previous results
            resultContainer.classList.remove('show');
            error.classList.remove('show');
            loading.classList.add('show');
            
            const formData = new FormData();
            formData.append('image', file);
            
            try {
                const response = await fetch('/recognize', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                loading.classList.remove('show');
                
                if (data.success) {
                    resultText.textContent = data.text || '[Boş]';
                    resultContainer.classList.add('show');
                } else {
                    error.textContent = 'Hata: ' + (data.error || 'Bilinmeyen hata');
                    error.classList.add('show');
                }
            } catch (err) {
                loading.classList.remove('show');
                error.textContent = 'Bağlantı hatası: ' + err.message;
                error.classList.add('show');
            }
        }
    </script>
</body>
</html>
    """
    return render_template_string(html)

@app.route('/recognize', methods=['POST'])
def recognize():
    """Gorsel tanima"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'Gorsel bulunamadi'})
        
        file = request.files['image']
        image_bytes = file.read()
        
        # Preprocess
        img_tensor = preprocess_image(image_bytes).to(device)
        
        # Inference
        with torch.no_grad():
            log_probs = model(img_tensor)
        
        # Decode
        predictions = decoder.decode_greedy(log_probs)
        text = predictions[0] if predictions else ""
        
        return jsonify({
            'success': True,
            'text': text
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description="OCR Demo Server")
    _parser.add_argument("--checkpoint", default="checkpoints/realworld_finetune/best_model.pth",
                         help="Kullanılacak checkpoint yolu")
    _parser.add_argument("--port", type=int, default=5000)
    _parser.add_argument("--device", default="cuda")
    _args = _parser.parse_args()

    load_model(_args.checkpoint, _args.device)
    print("\n" + "="*60)
    print(f"TEST SERVER BASLATILIYOR — http://localhost:{_args.port}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=_args.port, debug=False)
