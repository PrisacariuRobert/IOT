import cv2
from pyzbar import pyzbar
import json
from datetime import datetime
import mysql.connector
from keras import models
from keras import layers
import numpy as np
import time

# Database connection configuration
db_config = {
    'user': 'root',
    'password': '1234',
    'host': 'localhost',
    'database': 'QRSTART'
}

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Custom function to handle DepthwiseConv2D
def custom_depthwise_conv2d(**kwargs):
    kwargs.pop('groups', None)  # Ignore 'groups' if present
    return layers.DepthwiseConv2D(**kwargs)

# Function to load the model with custom objects handling
def load_keras_model(model_path):
    try:
        model = models.load_model(model_path, compile=False, custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model
model = load_keras_model("keras_Model.h5")

# Load the labels and strip any extra whitespace/newlines
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Function to decode QR code
def decode_qr(frame):
    decoded_objects = pyzbar.decode(frame)
    for obj in decoded_objects:
        try:
            if obj.type == 'QRCODE':  # Only process QR codes
                qr_data = obj.data.decode('utf-8')
                return json.loads(qr_data)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error decoding QR code: {e}")
    return None

# Function to check age from database
def check_age(name):
    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)

        # Query to get the birth date
        query = "SELECT birthDate FROM person WHERE name = %s"
        cursor.execute(query, (name,))
        result = cursor.fetchone()

        # Close the connection
        cursor.close()
        connection.close()

        if result and result['birthDate']:
            birth_date = result['birthDate']
            current_date = datetime.now().date()
            age = (current_date - birth_date).days // 365
            return age
        else:
            print("User not found or birth date not available.")
            return None
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Function to perform image classification
def classify_image():
    last_detected_class = None

    while True:
        ret, image = cap.read()
        if not ret:
            break

        # Resize the raw image into (224-height, 224-width) pixels
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Show the image in the same window
        cv2.imshow("QR Code Scanner", image_resized)

        # Make the image a numpy array and reshape it to the model's input shape
        image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image_array = (image_array / 127.5) - 1

        if model:
            # Predict using the model
            prediction = model.predict(image_array)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = prediction[0][index]

            # Print only if confidence score is above 70% and a new class is detected
            if confidence_score >= 0.99 and class_name != last_detected_class:
                print(f"Class: {class_name}, Confidence Score: {confidence_score * 100:.2f}%")
                last_detected_class = class_name
                break  # Stop the loop after detecting and printing

        # Listen to the keyboard for presses and exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Function to start the QR code scanning process
def start_qr_scanning():
    last_qr_data = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame for QR scanning
        cv2.imshow('QR Code Scanner', frame)

        # Decode QR code
        qr_data = decode_qr(frame)
        if qr_data and qr_data != last_qr_data:
            print("QR Code Data:", qr_data)
            name = qr_data.get("name")
            if name:
                age = check_age(name)
                if age is not None:
                    if age >= 21:
                        print(f"Access Granted: {name} is {age} years old.")
                        return  # Exit after access is granted
                    else:
                        print(f"Access Denied: {name} is {age} years old.")
            last_qr_data = qr_data

        # Listen to the keyboard for presses and exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    start_qr_scanning()  # Perform QR code scanning once
    while True:
        classify_image()  # Perform image classification
        print("Restarting image classification after 10 seconds...")
        time.sleep(10)  # Wait for 10 seconds
