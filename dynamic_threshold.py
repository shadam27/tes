import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve, classification_report

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
    new_data['Produk'] = le_product.transform(new_data['Produk'])
    new_data['Pembayaran'] = le_payment.transform(new_data['Pembayaran'])
    new_data['Harga Satuan'] = new_data['Total Harga'] / new_data['Jumlah']
    new_data['Harga Total Terkalkulasi'] = new_data['Jumlah'] * new_data['Harga Satuan']

    X_new = new_data[['Produk', 'Jumlah', 'Total Harga', 'Pembayaran', 'Harga Satuan', 'Harga Total Terkalkulasi']]
    X_new_scaled = scaler.transform(X_new)
    return X_new_scaled

def determine_optimal_threshold(y_true, predictions):
    fpr, tpr, thresholds = roc_curve(y_true, predictions)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Threshold optimal berdasarkan ROC: {optimal_threshold:.4f}")
    return optimal_threshold

def visualize_predictions(predictions, labels=None):
    plt.hist(predictions, bins=20, edgecolor='black', alpha=0.7, label="Semua Prediksi")
    if labels is not None:
        plt.hist(predictions[labels == 0], bins=20, edgecolor='black', alpha=0.5, label="Normal")
        plt.hist(predictions[labels == 1], bins=20, edgecolor='black', alpha=0.5, label="Anomali")
        plt.legend()

    plt.title("Distribusi Skor Prediksi")
    plt.xlabel("Skor Prediksi (Probabilitas Anomali)")
    plt.ylabel("Frekuensi")
    plt.show()

def evaluate_predictions(y_true, predictions, threshold):
    y_pred = (predictions > threshold).astype(int)
    print("\nHasil Evaluasi Model:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomali"]))

if __name__ == "__main__":
    model = load_model("G:/anomali1/models/anomaly_detection_model.h5")
    scaler, le_product, le_payment = load_preprocessors()

    new_data = pd.read_csv("G:/anomali1/data/transaksi_baru.csv", delimiter=';')

    
    X_new_scaled = preprocess_new_data(new_data, scaler, le_product, le_payment)

    
    predictions = model.predict(X_new_scaled).flatten()

    
    visualize_predictions(predictions)

    
    if 'Label' in new_data.columns:
        y_true = new_data['Label'].values
       
        optimal_threshold = determine_optimal_threshold(y_true, predictions)
       
        evaluate_predictions(y_true, predictions, optimal_threshold)

    
    dynamic_threshold = np.percentile(predictions, 90)
    print(f"\nThreshold dinamis berdasarkan persentil ke-90: {dynamic_threshold:.4f}")

    
    for i, pred in enumerate(predictions):
        status = "Anomali" if pred > dynamic_threshold else "Normal"
        print(f"Transaksi {i + 1}: {status} (Skor: {pred:.4f})")
