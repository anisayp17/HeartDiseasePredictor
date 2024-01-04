# Import library yang diperlukan
import itertools
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import streamlit as st
import time
import pickle

# Membaca file data
with open("data/hungarian.data", encoding='Latin1') as file:
  lines = [line.strip() for line in file]

# Mengambil data yang memiliki panjang 76 karakter
data = itertools.takewhile(
  lambda x: len(x) == 76,
  (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)

# Membuat DataFrame dari data
df = pd.DataFrame.from_records(data)


# Menghilangkan kolom pertama dan mengonversi nilai kolom menjadi float
df = df.iloc[:, :-1]
df = df.drop(df.columns[0], axis=1)
df = df.astype(float)


# Mengganti nilai -9.0 dengan NaN
df.replace(-9.0, np.NaN, inplace=True)

# Memilih kolom-kolom tertentu dari DataFrame
df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]

# Mapping kolom-kolom terpilih ke nama yang lebih deskriptif
column_mapping = {
  2: 'age',
  3: 'sex',
  8: 'cp',
  9: 'trestbps',
  11: 'chol',
  15: 'fbs',
  18: 'restecg',
  31: 'thalach',
  37: 'exang',
  39: 'oldpeak',
  40: 'slope',
  43: 'ca',
  50: 'thal',
  57: 'target'
}

# Mengganti nama kolom DataFrame sesuai dengan mapping yang telah didefinisikan
df_selected.rename(columns=column_mapping, inplace=True)

# Menghapus kolom yang tidak diperlukan dari DataFrame
columns_to_drop = ['ca', 'slope','thal']
df_selected = df_selected.drop(columns_to_drop, axis=1)

# Menghitung mean untuk kolom-kolom tertentu setelah menghapus nilai NaN
meanTBPS = df_selected['trestbps'].dropna()
meanChol = df_selected['chol'].dropna()
meanfbs = df_selected['fbs'].dropna()
meanRestCG = df_selected['restecg'].dropna()
meanthalach = df_selected['thalach'].dropna()
meanexang = df_selected['exang'].dropna()

# Mengonversi tipe data kolom-kolom ke tipe data float
meanTBPS = meanTBPS.astype(float)
meanChol = meanChol.astype(float)
meanfbs = meanfbs.astype(float)
meanthalach = meanthalach.astype(float)
meanexang = meanexang.astype(float)
meanRestCG = meanRestCG.astype(float)

# Menghitung mean dari kolom-kolom tersebut
meanTBPS = round(meanTBPS.mean())
meanChol = round(meanChol.mean())
meanfbs = round(meanfbs.mean())
meanthalach = round(meanthalach.mean())
meanexang = round(meanexang.mean())
meanRestCG = round(meanRestCG.mean())

# Membuat dictionary untuk nilai-nilai rata-rata yang akan digunakan untuk mengisi nilai NaN
fill_values = {
  'trestbps': meanTBPS,
  'chol': meanChol,
  'fbs': meanfbs,
  'thalach':meanthalach,
  'exang':meanexang,
  'restecg':meanRestCG
}

# Mengisi nilai NaN dengan nilai rata-rata yang telah dihitung sebelumnya
df_clean = df_selected.fillna(value=fill_values)

# Menghapus duplikat baris dari DataFrame
df_clean.drop_duplicates(inplace=True)

# Memisahkan fitur (X) dan label (y) dari DataFrame
X = df_clean.drop("target", axis=1)
y = df_clean['target']

# Melakukan oversampling menggunakan SMOTE untuk menangani ketidakseimbangan kelas
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Memuat model yang telah disimpan sebelumnya
model = pickle.load(open("model/xgb_model.pkl", 'rb'))

# Melakukan prediksi menggunakan model pada data yang telah di-resample
y_pred = model.predict(X)

# Menghitung akurasi dari prediksi model
accuracy = accuracy_score(y, y_pred)

# Mengonversi akurasi ke bentuk persentase dan membulatkannya
accuracy = round((accuracy * 100), 2)

# Membuat DataFrame final yang berisi fitur dan label
df_final = X
df_final['target'] = y

# ========================================================================================================================================================================================

# STREAMLIT

# Mengatur konfigurasi halaman Streamlit
st.set_page_config(
  page_title = "12913_Hungarian Heart Disease",
  page_icon = "logo.png"
)

# Menampilkan judul aplikasi
st.title("Hungarian Heart Disease")

# Menampilkan akurasi model sebagai bagian dari tampilan aplikasi
st.write(f"**_Model's Accuracy_** :  :green[**{accuracy}**]% (:red[_Do not copy outright_])")
st.write("")

# Membuat tiga tab dengan judul "Single-predict", "Multi-predict", dan "Description"
tab1, tab2, tab3 = st.tabs(["Single-predict", "Multi-predict", "Description"])

# Blok kode untuk tab pertama ("Single-predict")
with tab1:
  # Membuat header di sidebar untuk input pengguna
  st.sidebar.header("**User Input** Sidebar")

  # Input umur dengan batasan nilai minimum dan maksimum
  age = st.sidebar.number_input(label=":blue[**Age**]", min_value=df_final['age'].min(), max_value=df_final['age'].max())
  st.sidebar.write(f":orange[Min] value: :yellow[**{df_final['age'].min()}**], :red[Max] value: :red[**{df_final['age'].max()}**]")
  st.sidebar.write("")

  # Input jenis kelamin (Male/Female) dengan pemrosesan nilai
  sex_sb = st.sidebar.selectbox(label=":blue[**Sex**]", options=["Male", "Female"])
  st.sidebar.write("")
  st.sidebar.write("")
  if sex_sb == "Male":
    sex = 1
  elif sex_sb == "Female":
    sex = 0
  # -- Value 0: Female
  # -- Value 1: Male

  # Input jenis nyeri dada dengan pemrosesan nilai
  cp_sb = st.sidebar.selectbox(label=":blue[**Chest pain type**]", options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
  st.sidebar.write("")
  st.sidebar.write("")
  if cp_sb == "Typical angina":
    cp = 1
  elif cp_sb == "Atypical angina":
    cp = 2
  elif cp_sb == "Non-anginal pain":
    cp = 3
  elif cp_sb == "Asymptomatic":
    cp = 4
  # -- Value 1: typical angina
  # -- Value 2: atypical angina
  # -- Value 3: non-anginal pain
  # -- Value 4: asymptomatic

  # Input tekanan darah istirahat dengan batasan nilai
  trestbps = st.sidebar.number_input(label=":blue[**Resting blood pressure** (in mm Hg on admission to the hospital)]", min_value=df_final['trestbps'].min(), max_value=df_final['trestbps'].max())
  st.sidebar.write(f":orange[Min] value: :yellow[**{df_final['trestbps'].min()}**], :red[Max] value: :red[**{df_final['trestbps'].max()}**]")
  st.sidebar.write("")

  # Input kolesterol serum dengan batasan nilai
  chol = st.sidebar.number_input(label=":violet[**Serum cholestoral** (in mg/dl)]", min_value=df_final['chol'].min(), max_value=df_final['chol'].max())
  st.sidebar.write(f":orange[Min] value: :yellow[**{df_final['chol'].min()}**], :red[Max] value: :red[**{df_final['chol'].max()}**]")
  st.sidebar.write("")

  # Input gula darah puasa dengan pemrosesan nilai
  fbs_sb = st.sidebar.selectbox(label=":blue[**Fasting blood sugar > 120 mg/dl?**]", options=["False", "True"])
  st.sidebar.write("")
  st.sidebar.write("")
  if fbs_sb == "False":
    fbs = 0
  elif fbs_sb == "True":
    fbs = 1
  # -- Value 0: false
  # -- Value 1: true

  # Input hasil elektrokardiografi istirahat dengan pemrosesan nilai
  restecg_sb = st.sidebar.selectbox(label=":blue[**Resting electrocardiographic results**]", options=["Normal", "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"])
  st.sidebar.write("")
  st.sidebar.write("")
  if restecg_sb == "Normal":
    restecg = 0
  elif restecg_sb == "Having ST-T wave abnormality":
    restecg = 1
  elif restecg_sb == "Showing left ventricular hypertrophy":
    restecg = 2
  # -- Value 0: normal
  # -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST  elevation or depression of > 0.05 mV)
  # -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

  # Input detak jantung maksimum dengan batasan nilai
  thalach = st.sidebar.number_input(label=":blue[**Maximum heart rate achieved**]", min_value=df_final['thalach'].min(), max_value=df_final['thalach'].max())
  st.sidebar.write(f":orange[Min] value: :yellow[**{df_final['thalach'].min()}**], :red[Max] value: :red[**{df_final['thalach'].max()}**]")
  st.sidebar.write("")

  # Input angina yang diinduksi oleh olahraga dengan pemrosesan nilai
  exang_sb = st.sidebar.selectbox(label=":blue[**Exercise induced angina?**]", options=["No", "Yes"])
  st.sidebar.write("")
  st.sidebar.write("")
  if exang_sb == "No":
    exang = 0
  elif exang_sb == "Yes":
    exang = 1
  # -- Value 0: No
  # -- Value 1: Yes

  # Input depresi ST yang diinduksi oleh olahraga dengan batasan nilai
  oldpeak = st.sidebar.number_input(label=":blue[**ST depression induced by exercise relative to rest**]", min_value=df_final['oldpeak'].min(), max_value=df_final['oldpeak'].max())
  st.sidebar.write(f":orange[Min] value: :yellow[**{df_final['oldpeak'].min()}**], :red[Max] value: :red[**{df_final['oldpeak'].max()}**]")
  st.sidebar.write("")

  # Menyusun data input pengguna ke dalam bentuk dictionary
  data = {
    'Age': age,
    'Sex': sex_sb,
    'Chest pain type': cp_sb,
    'RPB': f"{trestbps} mm Hg",
    'Serum Cholestoral': f"{chol} mg/dl",
    'FBS > 120 mg/dl?': fbs_sb,
    'Resting ECG': restecg_sb,
    'Maximum heart rate': thalach,
    'Exercise induced angina?': exang_sb,
    'ST depression': oldpeak,
  }

  # Membuat DataFrame untuk preview input pengguna
  preview_df = pd.DataFrame(data, index=['input'])

  # Menampilkan header dan dua bagian DataFrame di tampilan utama
  st.header("User Input as DataFrame")
  st.write("")
  st.dataframe(preview_df.iloc[:, :6])
  st.write("")
  st.dataframe(preview_df.iloc[:, 6:])
  st.write("")

  # Inisialisasi hasil prediksi sebagai teks biru
  result = ":blue[-]"

  # Membuat tombol untuk melakukan prediksi
  predict_btn = st.button("**Predict**", type="primary")

  # Menampilkan spasi kosong
  st.write("")

  # Blok kode yang dijalankan saat tombol prediksi ditekan
  if predict_btn:
    # Mengambil input pengguna dan melakukan prediksi menggunakan model
    inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]

    # Menampilkan progres bar saat prediksi sedang berlangsung
    prediction = model.predict(inputs)[0]

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    # Menentukan hasil prediksi berdasarkan kategori penyakit jantung
    if prediction == 0:
      result = ":green[**Healthy**]"
    elif prediction == 1:
      result = ":orange[**Heart disease level 1**]"
    elif prediction == 2:
      result = ":orange[**Heart disease level 2**]"
    elif prediction == 3:
      result = ":red[**Heart disease level 3**]"
    elif prediction == 4:
      result = ":red[**Heart disease level 4**]"

  # Menampilkan hasil prediksi
  st.write("")
  st.write("")
  st.subheader("Prediction:")
  st.subheader(result)

# Blok kode untuk tab kedua ("Multi-predict")
with tab2:
  # Menampilkan judul
  st.header("Predict multiple data:")

  # Mengambil 5 baris pertama dari DataFrame hasil pemrosesan untuk contoh CSV
  sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

  # Menampilkan tombol untuk mengunduh contoh file CSV
  st.write("")
  st.download_button("Download CSV Example", data=sample_csv, file_name='sample_heart_disease_parameters.csv', mime='text/csv')

  # Membuat area untuk mengunggah file CSV
  st.write("")
  st.write("")
  file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

  # Blok kode yang dijalankan saat file CSV diunggah
  if file_uploaded:
    # Membaca DataFrame dari file CSV yang diunggah
    uploaded_df = pd.read_csv(file_uploaded)
    # Melakukan prediksi menggunakan model untuk setiap baris dalam DataFrame
    prediction_arr = model.predict(uploaded_df)

    # Menampilkan progres bar saat prediksi sedang berlangsung
    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 70):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)

    # Mengonversi hasil prediksi ke dalam bentuk teks kategori
    result_arr = []

    for prediction in prediction_arr:
      if prediction == 0:
        result = "Healthy"
      elif prediction == 1:
        result = "Heart disease level 1"
      elif prediction == 2:
        result = "Heart disease level 2"
      elif prediction == 3:
        result = "Heart disease level 3"
      elif prediction == 4:
        result = "Heart disease level 4"
      result_arr.append(result)

    # Membuat DataFrame untuk hasil prediksi
    uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

    # Menampilkan progres bar lebih lanjut
    for i in range(70, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    # Membagi layar menjadi dua bagian untuk menampilkan hasil prediksi dan DataFrame asli
    col1, col2 = st.columns([1, 2])

    with col1:
      # Menampilkan hasil prediksi
      st.dataframe(uploaded_result)
    with col2:
      # Menampilkan DataFrame asli
      st.dataframe(uploaded_df)

# Blok kode untuk tab ketiga ("Description")
with tab3:
    # Menampilkan judul
    st.header("Dataset Description")
    # Menampilkan penjelasan untuk setiap fitur dalam dataset
    st.write("1. **Age:**")
    st.write("   Explanation: This is your age. The older you are, the higher the risk of heart disease.")
    st.write("2. **Sex:**")
    st.write("   Explanation: Choose your gender, whether male or female. This factor can affect the risk of heart disease.")
    st.write("3. **Chest Pain Type:**")
    st.write("   Explanation: Select the level of chest pain you might experience (0-3), where 0 might mean no pain, and 3 might mean very severe pain.")
    st.write("4. **Resting Blood Pressure:**")
    st.write("   Explanation: This is your blood pressure when you are not engaged in physical activity. Normally, values below 120/80 mm Hg are considered good.")
    st.write("5. **Serum Cholesterol:**")
    st.write("   Explanation: This is the level of cholesterol in your blood. Lower values are better, and the normal range varies depending on certain factors.")
    st.write("6. **Fasting Blood Sugar:**")
    st.write("   Explanation: Choose the option that reflects your fasting blood sugar level. Normal values are usually less than 100 mg/dL.")
    st.write("7. **Resting Electrocardiographic Results:**")
    st.write("   Explanation: Select the result of the resting electrocardiogram (EKG). It reflects the heart's electrical activity at rest.")
    st.write("8. **Maximum Heart Rate Achieved:**")
    st.write("   Explanation: This is the maximum heart rate you achieve during exercise. Normally, the older you are, the lower the maximum heart rate.")
    st.write("9. **Exercise Induced Angina:**")
    st.write("   Explanation: Choose 'Yes' if you feel chest pain during exercise. This can indicate heart ischemia.")
    st.write("10. **ST Depression Induced by Exercise Relative to Rest:**")
    st.write("    Explanation: This is the depression of the ST segment that may occur during or after exercise. Higher values may indicate heart issues.")
    st.write("11. **Slope of the Peak Exercise ST Segment:**")
    st.write("    Explanation: Choose the slope of the peak exercise ST segment. Different values reflect patterns of change in the EKG during exercise.")
    st.write("12. **Number of Major Vessels Colored by Fluoroscopy:**")
    st.write("    Explanation: This is the number of major blood vessels visible in fluoroscopy images. A higher number may indicate issues with blood vessels.")
    st.write("13. **Thalassemia:**")
    st.write("    Explanation: Choose your type of thalassemia if known. Thalassemia can affect the production of red blood cells.")
    st.write("14. **Presence or Absence of Heart Disease:**")
    st.write("    Explanation: Choose 'No' if there is no heart disease, and other values reflect the severity of heart disease.")

    # Menambahkan tautan ke sumber dataset asli
    st.write("")
    st.write("For more details on the dataset, refer to the [original source](https://archive.ics.uci.edu/dataset/45/heart+disease).")

# Menambahkan garis pemisah dan kredit pembuat aplikasi
st.markdown("---")
st.write("© 2024 Heart Disease Predictor App. Created with ❤️ by [Anisa Yuliani Priyadi_A11.2020.12913]")
