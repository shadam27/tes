import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def load_and_preprocess_data(file_anomali, file_normal):

    data_anomali = pd.read_csv(file_anomali)
    data_normal = pd.read_csv(file_normal)


    data_anomali['Label'] = 1
    data_normal['Label'] = 0


    data = pd.concat([data_anomali, data_normal], ignore_index=True)


    le_product = LabelEncoder()
    le_payment = LabelEncoder()
    data['Produk'] = le_product.fit_transform(data['Produk'])
    data['Pembayaran'] = le_payment.fit_transform(data['Pembayaran'])


    data['Jumlah'] = data['Jumlah'].replace(0, 1)
    data['Harga Satuan'] = data['Total Harga'] / data['Jumlah']
    data['Harga Total Terkalkulasi'] = data['Jumlah'] * data['Harga Satuan']

    # Kolom untuk training
    X = data[['Produk', 'Jumlah', 'Total Harga', 'Pembayaran', 'Harga Satuan', 'Harga Total Terkalkulasi']]
    y = data['Label']


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le_product, le_payment

def build_neural_network(input_shape):
    model = Sequential([
        Dense(64, input_shape=(input_shape,), activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(X, y):
    model = build_neural_network(X.shape[1])
    model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2, shuffle=True)
    model.save("G:/anomali1/models/anomaly_detection_model.h5")
    print("Model saved to G:/anomali1/models/anomaly_detection_model.h5")

if __name__ == "__main__":
    X, y, scaler, le_product, le_payment = load_and_preprocess_data(
        "G:/anomali1/data/anomali.csv",
        "G:/anomali1/data/normal.csv"
    )
    train_model(X, y)


    np.save("G:/anomali1/models/scaler_mean.npy", scaler.mean_)
    np.save("G:/anomali1/models/scaler_std.npy", scaler.scale_)
    np.save("G:/anomali1/models/label_product.npy", le_product.classes_)
    np.save("G:/anomali1/models/label_payment.npy", le_payment.classes_)
