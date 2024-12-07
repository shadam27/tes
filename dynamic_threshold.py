import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_preprocessors():
    
    scaler = StandardScaler()
    scaler.mean_ = np.load("G:/anomali1/models/scaler_mean.npy")
    scaler.scale_ = np.load("G:/anomali1/models/scaler_std.npy")

    le_product = LabelEncoder()
    le_payment = LabelEncoder()
    le_product.classes_ = np.load("G:/anomali1/models/label_product.npy", allow_pickle=True)
    le_payment.classes_ = np.load("G:/anomali1/models/label_payment.npy", allow_pickle=True)

    return scaler, le_product, le_payment

def preprocess_new_data(new_data, scaler, le_product, le_payment):

    new_data['Produk'] = new_data['Produk'].apply(
        lambda x: le_product.transform([x])[0] if x in le_product.classes_ else 0
    )
    new_data['Pembayaran'] = new_data['Pembayaran'].apply(
        lambda x: le_payment.transform([x])[0] if x in le_payment.classes_ else 0
    )


    new_data['Jumlah'] = new_data['Jumlah'].replace(0, 1)
    new_data['Harga Satuan'] = new_data['Total Harga'] / new_data['Jumlah']
    new_data['Harga Total Terkalkulasi'] = new_data['Jumlah'] * new_data['Harga Satuan']
    X_new = new_data[['Produk', 'Jumlah', 'Total Harga', 'Pembayaran', 'Harga Satuan', 'Harga Total Terkalkulasi']]

    X_new_scaled = scaler.transform(X_new)
    return X_new_scaled

def visualize_predictions(predictions):
    plt.hist(predictions, bins=20, edgecolor='black', alpha=0.7)
    plt.title("Distribusi Skor Prediksi")
    plt.xlabel("Skor Prediksi (Probabilitas Anomali)")
    plt.ylabel("Frekuensi")
    plt.show()

if __name__ == "__main__":
    model = load_model("G:/anomali1/models/anomaly_detection_model.h5")
    scaler, le_product, le_payment = load_preprocessors()


    new_data = pd.read_csv("G:/anomali1/data/transaksi_baru.csv")


    X_new_scaled = preprocess_new_data(new_data, scaler, le_product, le_payment)


    predictions = model.predict(X_new_scaled).flatten()


    visualize_predictions(predictions)


    dynamic_threshold = np.percentile(predictions, 90)
    print(f"Threshold dinamis: {dynamic_threshold:.4f}")


    for i, pred in enumerate(predictions):
        status = "Anomali" if pred > dynamic_threshold else "Normal"
        print(f"Transaksi {i + 1}: {status} (Skor: {pred:.4f})")
