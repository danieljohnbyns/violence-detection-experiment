# evaluate_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('trained_model/violence_detection.h5')

# Recreate the validation data generator
val_dir = 'dataset'  # Path to your dataset
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Use the same validation split as training
)

# Evaluate the model on validation data
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Optional: Plot training history if you saved it (e.g., from CSV)
# import pandas as pd
# history_df = pd.read_csv('trained_model/training_history.csv')
# plt.plot(history_df['accuracy'], label='Training Accuracy')
# plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
# plt.legend()
# plt.show()