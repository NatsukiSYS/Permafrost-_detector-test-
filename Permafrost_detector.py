import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

# Проверка установки TensorFlow
try:
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError:
    print("❌ TensorFlow не установлен!")
    print("Установите командой: pip install tensorflow")
    sys.exit(1)


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


PATH_TO_IMAGES = r"C:\Users\david\Gis-It\Python\permafrost_data\images"
PATH_TO_MASKS = r"C:\Users\david\Gis-It\Python\permafrost_data\masks"

IMG_SIZE = (256, 256)
BATCH_SIZE = 2  
EPOCHS = 3
RANDOM_SEED = 42


def load_data(image_path, mask_path, img_size=(256, 256)):
    """Загружает изображения и маски из папок"""
    images = []
    masks = []
    
    if not os.path.exists(image_path):
        print(f"❌ Папка не найдена: {image_path}")
        return np.array([]), np.array([])
    
    if not os.path.exists(mask_path):
        print(f"❌ Папка не найдена: {mask_path}")
        return np.array([]), np.array([])
    

    image_files = [f for f in os.listdir(image_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
    
    print(f"\n📁 Найдено изображений: {len(image_files)}")
    
    if len(image_files) == 0:
        print("❌ Нет изображений в папке!")
        return np.array([]), np.array([])
    
    # Загружаем каждое изображение
    for img_file in tqdm(image_files, desc="Загрузка"):
        # Путь к изображению
        img_full_path = os.path.join(image_path, img_file)
        
        # Ищем маску с таким же именем
        name = os.path.splitext(img_file)[0]
        mask_file = None
        
        for ext in ['.png', '.jpg', '.jpeg', '.tif']:
            potential = os.path.join(mask_path, name + ext)
            if os.path.exists(potential):
                mask_file = potential
                break
        
        if mask_file is None:
            print(f"⚠️ Нет маски для {img_file}, пропускаем")
            continue
        
        try:
            # Загружаем изображение
            img = cv2.imread(img_full_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = img.astype(np.float32) / 255.0
            
            # Загружаем маску
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = cv2.resize(mask, img_size)
            mask = (mask > 127).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)
            
            images.append(img)
            masks.append(mask)
            
        except Exception as e:
            print(f"Ошибка с {img_file}: {e}")
            continue
    
    if len(images) == 0:
        print("❌ Не загружено ни одной пары!")
        return np.array([]), np.array([])
    
    print(f"✅ Загружено {len(images)} пар изображение-маска")
    return np.array(images), np.array(masks)

def create_synthetic_data(num_samples=20, img_size=(256, 256)):
    """Создает синтетические данные для обучения"""
    print(f"\n🔧 Создаю {num_samples} синтетических примеров...")
    images = []
    masks = []
    
    for i in tqdm(range(num_samples), desc="Создание данных"):
        # Случайное изображение (имитация спутникового снимка)
        img = np.random.rand(img_size[0], img_size[1], 3) * 255
        img = img.astype(np.float32) / 255.0
        
        # Маска деградации: случайные пятна
        mask = np.zeros(img_size, dtype=np.float32)
        
        # Добавляем 1-3 случайных пятна деградации
        num_patches = np.random.randint(1, 4)
        for _ in range(num_patches):
            x = np.random.randint(0, img_size[1] - 30)
            y = np.random.randint(0, img_size[0] - 30)
            w = np.random.randint(20, 50)
            h = np.random.randint(20, 50)
            mask[y:y+h, x:x+w] = 1.0
        
        # Добавляем немного шума к маске
        mask = mask + np.random.rand(img_size[0], img_size[1]) * 0.1
        mask = np.clip(mask, 0, 1)
        
        images.append(img)
        masks.append(np.expand_dims(mask, axis=-1))
    
    print(f"✅ Создано {num_samples} синтетических пар")
    return np.array(images), np.array(masks)

# ==================== МОДЕЛЬ U-NET ====================
def create_unet(input_size=(256, 256, 3)):
    """Создает упрощенную модель U-Net"""
    
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)  # Уменьшил количество фильтров
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bottom
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    
    # Decoder
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = concatenate([up5, conv3])
    conv5 = Conv2D(128, 3, activation='relu', padding='same')(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = concatenate([up6, conv2])
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = concatenate([up7, conv1])
    conv7 = Conv2D(32, 3, activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# ==================== МЕТРИКИ ====================
def dice_coef(y_true, y_pred):
    smooth = 1e-7
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# ==================== ВИЗУАЛИЗАЦИЯ ====================
def show_predictions(model, X, y, num_samples=2):
    """Показывает примеры предсказаний"""
    num_samples = min(num_samples, len(X))
    indices = np.random.choice(len(X), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Предсказание
        pred = model.predict(X[idx:idx+1], verbose=0)[0]
        pred_binary = (pred > 0.5).astype(np.float32)
        
        axes[i, 0].imshow(X[idx])
        axes[i, 0].set_title('Изображение')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(y[idx].squeeze(), cmap='Reds')
        axes[i, 1].set_title('Истинная маска')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_binary.squeeze(), cmap='Reds')
        axes[i, 2].set_title('Предсказание')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
    """Главная функция"""
    print("="*60)
    print("🌍 ДЕТЕКТОР ДЕГРАДАЦИИ МНОГОЛЕТНЕЙ МЕРЗЛОТЫ")
    print("="*60)
    
    # 1. Загрузка данных
    print("\n📂 ЗАГРУЗКА ДАННЫХ...")
    X, y = load_data(PATH_TO_IMAGES, PATH_TO_MASKS, IMG_SIZE)
    
    # Если реальных данных мало, создаем синтетические
    if len(X) < 10:
        print(f"\n⚠️ Найдено только {len(X)} реальных изображений (нужно минимум 10)")
        print("📊 Добавляю синтетические данные для обучения...")
        
        synthetic_X, synthetic_y = create_synthetic_data(num_samples=30, img_size=IMG_SIZE)
        
        if len(X) > 0:
            # Объединяем реальные и синтетические данные
            X = np.concatenate([X, synthetic_X])
            y = np.concatenate([y, synthetic_y])
            print(f"✅ Всего данных: {len(X)} (реальных: {len(X)-30}, синтетических: 30)")
        else:
            X = synthetic_X
            y = synthetic_y
            print(f"✅ Использую только синтетические данные: {len(X)} примеров")
    
    if len(X) < 3:
        print("\n❌ НЕДОСТАТОЧНО ДАННЫХ ДЛЯ ОБУЧЕНИЯ!")
        print("Минимум нужно 3 примера, а у вас:", len(X))
        return None, None
    
    # 2. Разделение данных (с учетом малого количества)
    print("\n📊 РАЗДЕЛЕНИЕ ДАННЫХ...")
    
    # Для малого количества данных используем простой подход
    if len(X) < 10:
        # Не делаем отдельную тестовую выборку
        indices = np.random.permutation(len(X))
        train_size = max(2, int(len(X) * 0.7))
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx] if len(val_idx) > 0 else X[train_idx[:1]]
        y_val = y[val_idx] if len(val_idx) > 0 else y[train_idx[:1]]
        X_test = X_val[:min(2, len(X_val))]
        y_test = y_val[:min(2, len(y_val))]
        
        print(f"   Обучающая выборка: {len(X_train)}")
        print(f"   Валидационная: {len(X_val)}")
        print(f"   Тестовая: {len(X_test)}")
    else:
        # Нормальное разделение для достаточного количества данных
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED
        )
        
        print(f"   Обучающая выборка: {len(X_train)}")
        print(f"   Валидационная: {len(X_val)}")
        print(f"   Тестовая: {len(X_test)}")
    
    # 3. Создание модели
    print("\n🏗️ СОЗДАНИЕ МОДЕЛИ...")
    model = create_unet(input_size=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', dice_coef]
    )
    
    model.summary()
    
    # 4. Обучение
    print(f"\n🚀 ОБУЧЕНИЕ ({EPOCHS} ЭПОХ)...")
    
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=2),
    ]
    
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=min(BATCH_SIZE, len(X_train)),
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        return None, None
    
    # 5. Оценка
    print("\n📈 ОЦЕНКА МОДЕЛИ...")
    if len(X_test) > 0:
        test_loss, test_acc, test_dice = model.evaluate(X_test, y_test, verbose=0)
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Test Dice: {test_dice:.4f}")
    else:
        print("   ⚠️ Нет тестовых данных для оценки")
    
    # 6. Визуализация
    print("\n🎨 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ...")
    show_predictions(model, X_val, y_val, num_samples=min(2, len(X_val)))
    
    # 7. Сохранение
    model.save('permafrost_model.keras')
    print("\n💾 Модель сохранена как 'permafrost_model.keras'")
    
    print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    return model, history

# ==================== ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ ====================
def predict_image(model, image_path):
    """Предсказание на новом изображении"""
    # Загрузка
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Не удалось загрузить {image_path}")
        return None
    
    original_size = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    
    # Предсказание
    pred = model.predict(img_input, verbose=0)[0]
    pred_binary = (pred > 0.5).astype(np.uint8) * 255
    pred_original = cv2.resize(pred_binary, (original_size[1], original_size[0]))
    
    # Визуализация
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Оригинал')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred.squeeze(), cmap='jet')
    plt.title('Вероятность деградации')
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    overlay = img.copy()
    overlay[pred_original.squeeze() > 127] = [255, 0, 0]
    plt.imshow(overlay)
    plt.title('Деградация (красное)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Статистика
    degradation_percent = (np.sum(pred_original > 127) / pred_original.size) * 100
    print(f"\n📊 Деградация: {degradation_percent:.2f}% площади")
    
    return pred_original

# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    # Создаем папки если их нет
    os.makedirs(PATH_TO_IMAGES, exist_ok=True)
    os.makedirs(PATH_TO_MASKS, exist_ok=True)
    
    print("="*60)
    print("🚀 ЗАПУСК ДЕТЕКТОРА ДЕГРАДАЦИИ МЕРЗЛОТЫ")
    print("="*60)
    
    # Запускаем обучение
    model, history = main()
    
    # Если модель создана, показываем пример предсказания
    if model is not None:
        print("\n" + "="*60)
        print("🎯 ПРИМЕР ПРЕДСКАЗАНИЯ")
        print("="*60)
        
        # Берем первое тестовое изображение
        test_files = [f for f in os.listdir(PATH_TO_IMAGES) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if test_files:
            test_path = os.path.join(PATH_TO_IMAGES, test_files[0])
            predict_image(model, test_path)
