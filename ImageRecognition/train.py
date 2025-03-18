import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.model_selection import train_test_split

# Load the Excel file
file_path = "shoe_data_with_photos.xlsx"  # Replace with your file path
df = pd.read_excel(file_path)

# Load images
image_size = (224, 224)  # Resize to fit model input
image_dir = "shoe_images/"  # Directory containing shoe images

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0  # Normalize
    return img_array

# Prepare dataset
image_data = []
labels = []

for _, row in df.iterrows():
    img_path = os.path.join(image_dir, row['image_paths'].strip("[]").split('\\')[-1].replace('"', ''))
    if os.path.exists(img_path):
        image_data.append(load_and_preprocess_image(img_path))
        labels.append(row['category'])  # Predicting category, can be changed

# Convert to NumPy arrays
X = np.array(image_data)
y = np.array(labels)

# Encode labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save("shoe_classifier.h5")

# Predict a new shoe category
def predict_shoe_category(image_path):
    img = load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    return label_encoder.inverse_transform([np.argmax(prediction)])[0]

# Example usage
predicted_category = predict_shoe_category("path_to_new_shoe_image.jpg")
print("Predicted Shoe Category:", predicted_category)
