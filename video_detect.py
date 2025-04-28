import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image  # Импортируем PIL для переиспользования функций

# Размеры, как в обучении
IMG_HEIGHT = 272
IMG_WIDTH = 464
MODEL_PATH = 'best_model_v3.h5'  # Используем модель, сохраненную через ModelCheckpoint
INPUT_VIDEO_PATH = 'test.mp4'
OUTPUT_VIDEO_PATH = 'mouse_segmented_new_2804.mp4'

# Загрузка модели с кастомной метрикой
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

# Dice коэффициент
def dice_coefficient(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

model = load_model(MODEL_PATH, custom_objects={'iou_metric': iou_metric, 'dice_loss': dice_loss, 'dice_coefficient':dice_coefficient})

# Функция предобработки кадра (переиспользуем из обучения)
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb.astype(np.float32) / 255.0
    return np.expand_dims(frame_norm, axis=0), frame_resized


def overlay_mask(original_frame, mask, alpha=0.4):
    mask_colored = np.zeros_like(original_frame)
    mask_colored[:, :, 1] = mask
    blended = cv2.addWeighted(original_frame, 1.0, mask_colored, alpha, 0)
    return blended

# Открытие видео
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Видео для записи
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

frame_idx = 0

print(" Начинаем обработку видео...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Предобработка
    input_tensor, resized_frame = preprocess_frame(frame)

    # Предсказание маски
    pred = model.predict(input_tensor)[0]
    mask = (pred[:, :, 0] > 0.5).astype(np.uint8) * 255

    # Маску ресайзим к размеру оригинального кадра
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # Наложение маски
    result_frame = overlay_mask(frame, mask_resized)

    # Сохраняем кадр
    out.write(result_frame)

    frame_idx += 1
    if frame_idx % 10 == 0:
        print(f"Обработано кадров: {frame_idx}")

cap.release()
out.release()
print(f" Обработка завершена. Видео сохранено: {OUTPUT_VIDEO_PATH}")