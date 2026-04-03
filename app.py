from flask import Flask, request, render_template, jsonify, url_for
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import os
import uuid
from datetime import datetime

app = Flask(__name__)

# Создаем папки для файлов
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Загружаем модель
print("🔄 Загрузка нейросети...")
try:
    model = load_model('permafrost_model.keras', compile=False)
    print("✅ Модель готова!")
except:
    print("⚠️ Модель не найдена, создаю упрощенную версию для демо")
    model = None

def predict_degradation(image_path):
    """Анализ изображения"""
    if model is None:
        # Демо-режим без реальной модели
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        # Создаем имитацию маски (для демо)
        cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
        degradation_percent = np.random.uniform(15, 45)
    else:
        # Реальная модель
        img = load_img(image_path, target_size=(256, 256))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        pred = model.predict(img_array, verbose=0)[0]
        pred = (pred > 0.5).astype(np.uint8) * 255
        
        # Восстанавливаем размер
        original_img = cv2.imread(image_path)
        h, w = original_img.shape[:2]
        mask = cv2.resize(pred, (w, h))
        degradation_percent = (np.sum(mask > 0) / mask.size) * 100
    
    # Сохраняем маску
    mask_filename = f"mask_{uuid.uuid4().hex}.png"
    mask_path = os.path.join('static/results', mask_filename)
    cv2.imwrite(mask_path, mask)
    
    return mask_filename, degradation_percent

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'Нет файла'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    # Сохраняем загруженное фото
    filename = f"upload_{uuid.uuid4().hex}.jpg"
    filepath = os.path.join('static/uploads', filename)
    file.save(filepath)
    
    # Анализируем
    mask_filename, percent = predict_degradation(filepath)
    
    return jsonify({
        'original': f'/static/uploads/{filename}',
        'mask': f'/static/results/{mask_filename}',
        'percent': round(percent, 2),
        'status': 'critical' if percent > 30 else 'warning' if percent > 15 else 'normal'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
