import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st

# Membaca data dari file CSV
data = pd.read_csv('es teh mbk pris.csv')

# Menghapus kolom yang tidak diperlukan ('no' dan 'tanggal')
data = data.drop(columns=['no', 'tanggal'])

# Lakukan One-Hot Encoding untuk kolom 'varian rasa' dan 'cuaca'
data_encoded = pd.get_dummies(data, columns=['varian rasa', 'cuaca'], drop_first=True)

# Fitur (X) dan target (y)
X = data_encoded.drop(columns=['penjualan'])
y = data_encoded['penjualan']

# Melatih model Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)

# Melatih model Random Forest
random_forest_model = RandomForestRegressor(random_state=42)
random_forest_model.fit(X, y)

# Prediksi untuk menghitung akurasi
linear_pred = linear_model.predict(X)
forest_pred = random_forest_model.predict(X)

# Menghitung R² (R-squared) dan MAE (Mean Absolute Error) untuk kedua model
linear_r2 = r2_score(y, linear_pred)
linear_mae = mean_absolute_error(y, linear_pred)

forest_r2 = r2_score(y, forest_pred)
forest_mae = mean_absolute_error(y, forest_pred)

# Simpan kolom fitur yang digunakan saat pelatihan
trained_columns = X.columns.tolist()

# Streamlit UI
st.title('Teafty Predik')

# Menampilkan akurasi kedua model
st.subheader('Akurasi Model')
st.write(f'Linear Regression - R²: {linear_r2:.4f}, MAE: {linear_mae:.2f}')
st.write(f'Random Forest - R²: {forest_r2:.4f}, MAE: {forest_mae:.2f}')

# Fungsi untuk melakukan prediksi
def prediksi_penjualan(varian_rasa, cuaca, harga, hari_ke, model_type):
    # Membuat data prediksi sesuai dengan format data pelatihan
    pred_data = pd.DataFrame({
        'harga': [harga],
        'varian rasa_Leci': [1 if varian_rasa == 'Leci' else 0],
        'varian rasa_Grape': [1 if varian_rasa == 'Grape' else 0],
        'varian rasa_Stroberry': [1 if varian_rasa == 'Stroberry' else 0],
        'varian rasa_Markisa': [1 if varian_rasa == 'Markisa' else 0],
        'cuaca_Berawan': [1 if cuaca == 'Berawan' else 0],
        'cuaca_Cerah': [1 if cuaca == 'Cerah' else 0],
        'cuaca_Hujan': [1 if cuaca == 'Hujan' else 0],
        'hari_ke': [hari_ke]
    })

    # Menjaga kolom input konsisten dengan model (mencocokkan kolom yang hilang)
    for col in trained_columns:
        if col not in pred_data.columns:
            pred_data[col] = 0

    # Menyusun ulang agar kolom sesuai dengan model
    pred_data = pred_data[trained_columns]

    # Pilih model untuk prediksi
    if model_type == 'Linear Regression':
        prediksi = linear_model.predict(pred_data)
    elif model_type == 'Random Forest':
        prediksi = random_forest_model.predict(pred_data)
    return prediksi[0]

# Input dari pengguna
varian_rasa = st.selectbox('Pilih Varian Rasa', ['Lemon', 'Leci', 'Grape', 'Stroberry', 'Markisa'])
cuaca = st.selectbox('Pilih Cuaca', ['Cerah', 'Berawan', 'Hujan'])
harga = st.number_input('Masukkan Harga (Rp)', min_value=0, step=1000)
hari_ke = st.number_input('Pilih Hari ke-', min_value=1, max_value=100)

# Pilihan model untuk prediksi
model_type = st.radio('Pilih Model Prediksi', ['Linear Regression', 'Random Forest'])

# Tombol untuk melakukan prediksi
if st.button('Prediksi'):
    try:
        # Lakukan prediksi
        prediksi = prediksi_penjualan(varian_rasa, cuaca, harga, hari_ke, model_type)
        st.success(f"Perkiraan Penjualan Teh pada Hari ke-{hari_ke} menggunakan {model_type}: {prediksi:.2f} unit")

        # # Jika data aktual ada untuk hari yang dipilih, hitung akurasi prediksi
        # if hari_ke <= len(data):  # Pastikan hari ke yang dipilih ada dalam data
        #     aktual = data.iloc[hari_ke - 1]['penjualan']
        #     error = abs(aktual - prediksi)
        #     if aktual > 0:
        #         akurasi = (1 - (error / aktual)) * 100
        #         st.write(f"Akurasi Prediksi: {akurasi:.2f}%")
        #     else:
        #         st.write("Akurasi Prediksi: Tidak dapat dihitung (Nilai aktual adalah 0)")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# import streamlit as st

# # Membaca data dari file CSV
# data = pd.read_csv('es teh mbk pris.csv')

# # Menghapus kolom yang tidak diperlukan ('no' dan 'tanggal')
# data = data.drop(columns=['no', 'tanggal'])

# # Lakukan One-Hot Encoding untuk kolom 'varian rasa' dan 'cuaca'
# data_encoded = pd.get_dummies(data, columns=['varian rasa', 'cuaca'], drop_first=True)

# # Fitur (X) dan target (y)
# X = data_encoded.drop(columns=['penjualan'])
# y = data_encoded['penjualan']

# # Melatih model Linear Regression
# linear_model = LinearRegression()
# linear_model.fit(X, y)

# # Melatih model Random Forest
# random_forest_model = RandomForestRegressor(random_state=42)
# random_forest_model.fit(X, y)

# # Prediksi untuk menghitung akurasi
# linear_pred = linear_model.predict(X)
# forest_pred = random_forest_model.predict(X)

# # Kombinasi prediksi (Rata-rata dari Linear Regression dan Random Forest)
# combined_pred = (linear_pred + forest_pred) / 2

# # Menghitung R² (R-squared) dan MAE (Mean Absolute Error) untuk model gabungan
# combined_r2 = r2_score(y, combined_pred)
# combined_mae = mean_absolute_error(y, combined_pred)

# # Simpan kolom fitur yang digunakan saat pelatihan
# trained_columns = X.columns.tolist()

# # Streamlit UI
# st.title('Prediksi Penjualan Teh - Model Gabungan')

# # Menampilkan akurasi model gabungan
# st.subheader('Akurasi Model Gabungan')
# st.write(f'R² (R-squared): {combined_r2:.4f}')
# st.write(f'Mean Absolute Error (MAE): {combined_mae:.2f}')

# # Fungsi untuk melakukan prediksi
# def prediksi_penjualan(varian_rasa, cuaca, harga, hari_ke):
#     # Membuat data prediksi sesuai dengan format data pelatihan
#     pred_data = pd.DataFrame({
#         'harga': [harga],
#         'varian rasa_Leci': [1 if varian_rasa == 'Leci' else 0],
#         'varian rasa_Grape': [1 if varian_rasa == 'Grape' else 0],
#         'varian rasa_Stroberry': [1 if varian_rasa == 'Stroberry' else 0],
#         'varian rasa_Markisa': [1 if varian_rasa == 'Markisa' else 0],
#         'cuaca_Berawan': [1 if cuaca == 'Berawan' else 0],
#         'cuaca_Cerah': [1 if cuaca == 'Cerah' else 0],
#         'cuaca_Hujan': [1 if cuaca == 'Hujan' else 0],
#         'hari_ke': [hari_ke]
#     })

#     # Menjaga kolom input konsisten dengan model (mencocokkan kolom yang hilang)
#     for col in trained_columns:
#         if col not in pred_data.columns:
#             pred_data[col] = 0

#     # Menyusun ulang agar kolom sesuai dengan model
#     pred_data = pred_data[trained_columns]

#     # Prediksi menggunakan model Linear Regression dan Random Forest
#     linear_pred = linear_model.predict(pred_data)
#     forest_pred = random_forest_model.predict(pred_data)

#     # Kombinasi prediksi (Rata-rata)
#     combined_pred = (linear_pred + forest_pred) / 2
#     return combined_pred[0]

# # Input dari pengguna
# varian_rasa = st.selectbox('Pilih Varian Rasa', ['Lemon', 'Leci', 'Grape', 'Stroberry', 'Markisa'])
# cuaca = st.selectbox('Pilih Cuaca', ['Cerah', 'Berawan', 'Hujan'])
# harga = st.number_input('Masukkan Harga (Rp)', min_value=0, step=1000)
# hari_ke = st.number_input('Pilih Hari ke-', min_value=1, max_value=100)

# # Tombol untuk melakukan prediksi
# if st.button('Prediksi'):
#     try:
#         # Lakukan prediksi
#         prediksi = prediksi_penjualan(varian_rasa, cuaca, harga, hari_ke)
#         st.success(f"Perkiraan Penjualan Teh pada Hari ke-{hari_ke} (Model Gabungan): {prediksi:.2f} unit")

#         # Jika data aktual ada untuk hari yang dipilih, hitung akurasi prediksi
#         if hari_ke <= len(data):  # Pastikan hari ke yang dipilih ada dalam data
#             aktual = data.iloc[hari_ke - 1]['penjualan']
#             error = abs(aktual - prediksi)
#             if aktual > 0:
#                 akurasi = (1 - (error / aktual)) * 100
#                 st.write(f"Akurasi Prediksi: {akurasi:.2f}%")
#             else:
#                 st.write("Akurasi Prediksi: Tidak dapat dihitung (Nilai aktual adalah 0)")
#     except Exception as e:
#         st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
