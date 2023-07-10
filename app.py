from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import requests

app = Flask(__name__)

# Memuat model saat aplikasi Flask diinisialisasi
model = load_model('C:/Users/gilym/Downloads/Tes/fixed.h5')

@app.route('/', methods=['GET'])
def home():
    return ({'message': 'Sukses'})

@app.route('/predict', methods=['POST'])
def predict():
    # Menerima file gambar dari request POST
    image_file = request.files['image']
    image = Image.open(image_file)
    # Resizing gambar
    input_shape = (256, 256)  # Ubah sesuai dengan ekspektasi model
    resized_image = image.resize(input_shape)

    # Convert gambar menjadi array NumPy
    image_array = np.array(resized_image)

    # Normalisasi nilai piksel (opsional, jika diperlukan oleh model)
    normalized_image = image_array / 255.0  # Normalisasi ke rentang [0, 1]

    # Tambahkan dimensi batch
    input_data = np.expand_dims(normalized_image, axis=0)

    # Konversi tipe data menjadi float32
    input_data = input_data.astype(np.float32)

    # Melakukan prediksi menggunakan model yang dimuat
    predictions = model.predict(input_data)

    # Dapatkan indeks kelas prediksi
    predicted_class_index = np.argmax(predictions)

    url = "https://backend-dot-recipe-finder-388213.as.r.appspot.com/findfood"

        # Menyiapkan data permintaan
    data = {
            "predicted_class": str(predicted_class_index),
          
        }

    headers = {
        "Content-Type": "application/json",
        "Authentication": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MzgsImlhdCI6MTY4NjUwNDI4OX0.L3QT-mSohJ1phz7cpF2b63_smqJbsfMeMxpnuezPud0"
        }
      # Mengirim permintaan POST ke URL API dengan payload JSON
    response = requests.post(url, json=data ,headers=headers)

        # Mendapatkan respons dari API
    api_response = response.json()

        # Mengembalikan respons dari API sebagai respons Flask
    return jsonify(api_response)



if __name__ == '__main__':
    app.run()
