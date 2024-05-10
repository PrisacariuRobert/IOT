import cv2
from pyzbar import pyzbar
import json
from datetime import datetime
import mysql.connector
import tensorflow 
from tensorflow.python.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Database connection configuration
db_config = {
    'user': 'root',
    'password': '1234',
    'host': 'localhost',
    'database': 'QRSTART'
}

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = [name.strip() for name in open("labels.txt", "r").readlines()]

# Function to classify image
def classify_image(img_path):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Load image and resize
    img = Image.open(img_path).convert("RGB")
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(img)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the class
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

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

# Function to start webcam and analyze QR codes
def start_webcam():
    cap = cv2.VideoCapture(0)
    last_qr_data = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
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
                        # Prompt for image file path
                        img_path = input("Enter the path to the image file: ")
                        class_name, confidence_score = classify_image(img_path)
                        print(f"Class: {class_name}, Confidence Score: {confidence_score}")
                    else:
                        print(f"Access Denied: {name} is {age} years old.")
            last_qr_data = qr_data

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_webcam()
