import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from PIL import Image
from sklearn.metrics import jaccard_score
from tensorflow.keras.callbacks import ModelCheckpoint

# Константы
IMG_HEIGHT = 272
IMG_WIDTH = 464
BATCH_SIZE = 64
EPOCHS = 30
MODEL_PATH = 'unet_simple_color_2804.h5'

# Пути
train_img_dir = 'images'
train_mask_dir = 'masks'
test_img_dir = 'test_images'
test_mask_dir = 'test_masks'
output_dir = 'predictions'
os.makedirs(output_dir, exist_ok=True)

# Загрузка и обработка изображений
def load_image_mask_pair(img_path, mask_path):
    # Загрузка цветного изображения (RGB)
    img = tf.io.read_file(img_path)
    img = tf.io.decode_bmp(img)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0  # Нормализация [0, 1]
    # Загрузка маски
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_bmp(mask)
    # Преобразование в grayscale, если маска RGB (3 канала)
    mask_shape = tf.shape(mask)
    channels = mask_shape[-1]
    def rgb_to_gray():
        return tf.image.rgb_to_grayscale(mask)
    def already_gray():
        return mask  # уже одноканальная
    mask = tf.cond(tf.equal(channels, 3), rgb_to_gray, already_gray)

    # Resize и бинаризация
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH])
    mask = tf.cast(mask > 127, tf.float32)  # бинарная маска [0,1]

    return img, mask

def get_dataset(img_dir, mask_dir):
    img_filenames = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.bmp')])
    mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.bmp')])

    # Оставим только те файлы, у которых есть и изображение, и маска
    common_filenames = sorted(list(set(img_filenames) & set(mask_filenames)))

    img_paths = [os.path.join(img_dir, f) for f in common_filenames]
    mask_paths = [os.path.join(mask_dir, f) for f in common_filenames]

    print(f"Совпадающих пар изображение-маска: {len(common_filenames)}")

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    dataset = dataset.map(load_image_mask_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), img_paths, mask_paths

# Упрощённая модель U-Net
def simple_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # Downsampling
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Dropout(0.2)(c1)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    # Bottleneck
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(b)

    # Upsampling
    u1 = layers.UpSampling2D(2)(b)
    u1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)

    u2 = layers.UpSampling2D(2)(c3)
    u2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(c4)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(c4)

    return models.Model(inputs, outputs)

# IoU метрика
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



# Графики обучения
def plot_training_history(history):
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure(figsize=(18, 5))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history.history['loss'], 'b', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
    plt.title('Loss по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()

    # IoU
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history.history['iou_metric'], 'b', label='Training IoU')
    plt.plot(epochs, history.history['val_iou_metric'], 'r', label='Validation IoU')
    plt.title('IoU по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('IoU')
    plt.legend()

    # Dice
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history.history['dice_coefficient'], 'b', label='Training Dice')
    plt.plot(epochs, history.history['val_dice_coefficient'], 'r', label='Validation Dice')
    plt.title('Dice по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Dice')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

# Загрузка датасета
train_ds, train_img_paths, _ = get_dataset(train_img_dir, train_mask_dir)
test_ds, test_img_paths, test_mask_paths = get_dataset(test_img_dir, test_mask_dir)

print(f" Тренировочных изображений: {len(train_img_paths)}")
print(f" Тестовых изображений: {len(test_img_paths)}")

checkpoint = ModelCheckpoint(
    'best_model_v3.h5',
    monitor='val_dice_coefficient',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Обучение модели
print(" Обучение модели...")
model = simple_unet()
model.compile(optimizer='adam', loss=dice_loss, metrics=[iou_metric, dice_coefficient])
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# Сохранение модели
model.save(MODEL_PATH)
print(f" Модель сохранена: {MODEL_PATH}")

# График обучения
plot_training_history(history)

# Сохранение и оценка предсказаний
def save_and_evaluate(dataset, img_paths, mask_paths, output_folder):
    ious = []
    losses = []
    bce = tf.keras.losses.BinaryCrossentropy()

    for batch_images, _ in dataset:
        preds = model.predict(batch_images)

        for j in range(len(preds)):
            pred = preds[j]
            mask = (pred > 0.5).astype(np.uint8) * 255
            filename = os.path.splitext(os.path.basename(img_paths[0]))[0]
            Image.fromarray(mask.squeeze()).convert("L").save(os.path.join(output_folder, f"{filename}_pred.bmp"))
            img_paths.pop(0)

            gt_mask = Image.open(mask_paths[0]).convert('L').resize((IMG_WIDTH, IMG_HEIGHT))
            gt_mask = np.array(gt_mask)
            gt_mask = (gt_mask > 127).astype(np.uint8) * 255
            mask_paths.pop(0)

            iou = jaccard_score(gt_mask.flatten() // 255, mask.flatten() // 255)
            ious.append(iou)

            y_true = tf.convert_to_tensor(gt_mask / 255.0, dtype=tf.float32)
            y_pred = tf.convert_to_tensor(mask / 255.0, dtype=tf.float32)
            loss = bce(tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1])).numpy()
            losses.append(loss)

    print(f"\nСредний IoU на тесте: {np.mean(ious):.4f}")
    print(f" Средний Loss на тесте: {np.mean(losses):.4f}")

save_and_evaluate(test_ds, test_img_paths.copy(), test_mask_paths.copy(), output_dir)