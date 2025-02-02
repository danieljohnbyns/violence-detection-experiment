# detect_image.py
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('trained_model/violence_detection.h5')

def predict_violence(image_path):
	img = cv2.imread(image_path)
	img = cv2.resize(img, (224, 224))
	img = img / 255.0
	img = np.expand_dims(img, axis=0)

	prediction = model.predict(img)
	return prediction[0][0]

# Example usage
if __name__ == "__main__":
	image_path = 'test_image.jpg'  # Replace with your image path
	prediction = predict_violence(image_path)
	print(f"Violence Probability: {prediction:.4f}")
	if prediction > 0.5:
		print("Violence detected!")
	else:
		print("No violence detected.")

	print("Hello World")

	image_path2 = 'test_image2.jpg'  # Replace with your image path
	prediction2 = predict_violence(image_path2)
	print(f"Violence Probability: {prediction2:.4f}")
	if prediction2 > 0.5:
		print("Violence detected!")
	else:
		print("No violence detected.")