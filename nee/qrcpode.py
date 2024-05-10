import mysql.connector
import qrcode
import json

# Database connection configuration
db_config = {
    'user': 'root',
    'password': '1234',
    'host': 'localhost',
    'database': 'QRSTART'
}

# Function to generate QR code
def generate_qr_code(data, filename):
    json_data = json.dumps(data)
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(json_data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    img.save(filename)

# Function to fetch user from the database and generate QR code
def generate_qr_code_for_user(name):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)

        query = "SELECT name, birthDate FROM person WHERE name = %s"
        cursor.execute(query, (name,))
        user = cursor.fetchone()

        if user:
            user_data = {
                "name": user["name"],
                "dob": user["birthDate"].strftime("%Y-%m-%d")
            }
            filename = f"qr_{user['name'].replace(' ', '_')}.png"
            generate_qr_code(user_data, filename)
            print(f"Generated QR code for {user['name']}")
        else:
            print(f"No user found with the name: {name}")

        cursor.close()
        connection.close()

    except mysql.connector.Error as err:
        print(f"Error: {err}")

if __name__ == "__main__":
    user_name = input("Enter the name of the person to generate QR code for: ")
    generate_qr_code_for_user(user_name)
