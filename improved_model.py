import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

train_img, test_img = tfds.load('emnist/letters', split=['train', 'test'], as_supervised=True)

# Enhanced preprocessing with data augmentation for training
def preprocess_train(img, lbl):
    img = tf.reshape(img, (28, 28, 1))
    img = tf.cast(img, tf.float32) / 255.0
    
    # Add data augmentation to make model more robust
    # Random brightness to simulate different drawing intensities
    img = tf.image.random_brightness(img, 0.3)
    
    # Random contrast to handle thick vs thin strokes
    img = tf.image.random_contrast(img, 0.7, 1.3)
    
    # Small random rotations using TensorFlow 2.x compatible method
    # Create rotation angle in radians
    angle = tf.random.uniform([], -0.2, 0.2)  # radians
    
    # Use tfa.image.rotate if available, otherwise use a custom rotation
    try:
        import tensorflow_addons as tfa
        img = tfa.image.rotate(img, angle)
    except ImportError:
        # Alternative: use tf.keras.preprocessing.image or skip rotation
        # For now, we'll skip rotation to avoid the dependency
        pass
    
    # Ensure values stay in valid range
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    lbl = lbl - 1
    lbl = tf.keras.utils.to_categorical(lbl, num_classes=26)
    return img, lbl

# Regular preprocessing for validation/test
def preprocess_test(img, lbl):
    img = tf.reshape(img, (28, 28, 1))
    img = tf.cast(img, tf.float32) / 255.0
    lbl = lbl - 1
    lbl = tf.keras.utils.to_categorical(lbl, num_classes=26)
    return img, lbl

# Apply preprocessing
train_new = train_img.map(preprocess_train).shuffle(10000).batch(64)
test_new = test_img.map(preprocess_test).batch(64)

# Improved CNN model - more layers and dropout
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    
    # First conv block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),  # Helps with training stability
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # Prevent overfitting
    
    # Second conv block  
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third conv block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(26, activation='softmax')
])

# Better optimizer settings
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Train for more epochs with callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-7),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

print("Starting training...")
model.fit(
    train_new, 
    epochs=15,  # More epochs
    validation_data=test_new,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(test_new, verbose=0)
print(f'Final test accuracy: {test_acc:.4f}')

# Save model
model.save('improved_model.keras')
print("Model saved as 'improved_model.keras'")