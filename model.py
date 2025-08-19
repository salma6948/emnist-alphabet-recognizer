import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models


train_img, test_img = tfds.load('emnist/letters', split=['train', 'test'], as_supervised=True)

# Preprocess images and labels
def preprocess(img, lbl):
    img = tf.reshape(img, (28, 28, 1))      # Ensure shape (28, 28, 1)
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
    lbl = lbl - 1                           # Adjust labels: 1–26 → 0–25
    lbl = tf.keras.utils.to_categorical(lbl, num_classes=26)  # One-hot
    return img, lbl

# Apply preprocessing
train_new = train_img.map(preprocess).shuffle(10000).batch(64)
test_new = test_img.map(preprocess).batch(64)

# Define CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(26, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_new, epochs=5, validation_data=test_new)

# Save model
model.save('model.keras')
