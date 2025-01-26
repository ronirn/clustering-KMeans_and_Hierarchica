import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import cophenet
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

# Menambahkan banner gambar dari folder proyek
image_path = "banner.png"  # Ganti dengan path gambar yang sesuai
banner_image = Image.open(image_path)
st.image(banner_image, use_column_width=True)

# Menambahkan judul aplikasi
st.title("Aplikasi Pengelompokan Segmentasi Pelanggan dengan Hierarchical Clustering dan KMeans")


    

with st.expander("Tentang Aplikasi"):
    st.markdown("""
    ### Tujuan Aplikasi:
    Aplikasi ini dirancang untuk melakukan **segmentasi pelanggan** menggunakan algoritma **Hierarchical Clustering** dan **KMeans** berdasarkan atribut-atribut pelanggan, seperti jenis kelamin, status perkawinan, usia, pendidikan, pendapatan, pekerjaan, dan ukuran pemukiman.

    Dengan aplikasi ini, pengguna dapat:
    - Memilih fitur yang relevan untuk digunakan dalam analisis clustering.
    - Melakukan **preprocessing data** yang mencakup standarisasi data numerik dan encoding untuk fitur kategorikal.
    - Menggunakan **Hierarchical Clustering** dan **KMeans** untuk clustering data berdasarkan parameter yang dipilih.
    - Mendapatkan hasil clustering yang tersegmentasi berdasarkan kluster yang terbentuk.
    - Mengunduh hasil clustering dalam format CSV untuk analisis lebih lanjut.
    """)

data = pd.read_csv('Clustering.csv')

st.subheader("Eksplorasi Data")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Deskripsi Data", "Statistik Deskriptif", "Visualisasi Korelasi", 
    "Visualisasi Scatter Plot", "Visualisasi Distribusi", "Visualisasi Outlier", 
    "Pencarian Korelasi"
])

with tab1:
    st.write("Dataset yang digunakan:")
    st.dataframe(data.head())

with tab2:
    st.write("Statistik Deskriptif:")
    st.write(data.describe())

with tab3:
    st.write("Heatmap Korelasi:")
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

with tab4:
    st.write("Visualisasi Scatter Plot:")
    # Pilih kolom yang ingin divisualisasikan dalam scatter plot
    x_column = st.selectbox("Pilih kolom untuk sumbu X", data.columns)
    y_column = st.selectbox("Pilih kolom untuk sumbu Y", data.columns)

    # Buat scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=data[x_column], y=data[y_column], ax=ax)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f'Scatter Plot antara {x_column} dan {y_column}')
    st.pyplot(fig)

with tab5:
    st.write("Visualisasi Distribusi:")
    # Pilih kolom untuk visualisasi distribusi
    column = st.selectbox("Pilih kolom", data.columns)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[column], kde=True, ax=ax)
    ax.set_title(f'Distribusi {column}')
    st.pyplot(fig)

with tab6:
    st.write("Visualisasi Outlier:")
    # Pilih kolom untuk outlier
    column = st.selectbox("Pilih kolom untuk boxplot", data.columns)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=data[column], ax=ax)
    ax.set_title(f'Outliers pada {column}')
    st.pyplot(fig)

with tab7:
    st.write("Pencarian Korelasi Antar Kolom:")
    col1 = st.selectbox("Pilih kolom pertama", data.columns)
    col2 = st.selectbox("Pilih kolom kedua", data.columns)
    correlation = data[col1].corr(data[col2])
    st.write(f"Korelasi antara {col1} dan {col2}: {correlation:.2f}")

# Menampilkan informasi tambahan tentang dataset
st.subheader("Informasi Dataset")
st.write(f"Jumlah Data: {data.shape[0]} baris")
st.write(f"Jumlah Fitur: {data.shape[1]} kolom")
    

st.subheader("Pilih fitur untuk clustering:")
features = st.multiselect("Fitur yang digunakan:", data.columns.tolist(), default=data.columns.tolist())

if features:
    st.subheader("Preprocessing Data")

# Buat tabs untuk menampilkan berbagai langkah preprocessing
tab1, tab2, tab3, tab4, tab5, tab6, = st.tabs([
    "Pemilihan Fitur", "Label Encoding", "Standarisasi Data", 
    "Mengatasi Outliers", "Menangani Missing Values", 
    "Memastikan Tipe Data"])

with tab1:
    st.write("Pilih Data yang akan diproses:")
    selected_data = data[features]
    st.write("Data yang dipilih:")
    st.dataframe(selected_data.head())

with tab2:
    st.write("Label Encoding untuk kolom kategori:")
    le = LabelEncoder()
    for col in selected_data.select_dtypes(include='object').columns:
        selected_data[col] = le.fit_transform(selected_data[col])
    st.write("Data setelah Label Encoding:")
    st.dataframe(selected_data.head())

with tab3:
    st.write("Standarisasi Data:")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data)
    st.write("Data setelah standarisasi:")
    st.dataframe(pd.DataFrame(scaled_data, columns=features).head())

with tab4:
    st.write("Mengatasi Outliers:")
    
    # Menggunakan Z-Score untuk mendeteksi outliers
    from scipy.stats import zscore
    z_scores = np.abs(zscore(selected_data))
    
    # Debug: tampilkan beberapa Z-Score pertama
    st.write("Beberapa Z-Score pertama:")
    st.write(z_scores[:5])  # Menampilkan 5 Z-Score pertama
    
    outliers = (z_scores > 3).all(axis=1)  # Outliers dengan Z-Score > 3
    
    # Menampilkan jumlah outliers
    jumlah_outliers = np.sum(outliers)
    st.write(f"Jumlah Outliers: {jumlah_outliers}")
    
    # Jika ada outliers, tampilkan data yang mengandung outliers
    if jumlah_outliers > 0:
        st.write("Data dengan Outliers:")
        st.dataframe(selected_data[outliers].head())  # Menampilkan beberapa baris data outlier
    
    # Mengatasi outliers (menghapus outliers)
    selected_data_no_outliers = selected_data[~outliers]
    
    # Menampilkan data tanpa outliers
    st.write("Data tanpa outliers:")
    st.dataframe(selected_data_no_outliers.head())


with tab5:
    st.write("Menangani Missing Values:")
    st.write(f"Jumlah missing values per kolom:\n{selected_data.isnull().sum()}")
    
    # Menangani missing values dengan imputasi
    st.write("Imputasi missing values dengan mean (untuk kolom numerik):")
    selected_data_imputed = selected_data.fillna(selected_data.mean())
    st.dataframe(selected_data_imputed.head())

with tab6:
    st.write("Memastikan Tipe Data:")
    st.write("Tipe data setiap kolom:")
    st.write(selected_data.dtypes)
    
    # Memastikan tipe data yang benar
    for col in selected_data.select_dtypes(include='object').columns:
        selected_data[col] = selected_data[col].astype('category')
    st.write("Tipe data setelah perbaikan:")
    st.write(selected_data.dtypes)

    # Sidebar for Algorithm Selection and Sliders
    st.sidebar.subheader("Pilih Algoritma Clustering")
    algorithm = st.sidebar.selectbox("Pilih algoritma clustering:", ["Hierarchical", "KMeans"], index=0)

    if algorithm == "Hierarchical":
        st.subheader("Hierarchical Clustering")
        st.write("Penjelasan Metode:")
        with st.expander("Penjelasan tentang setiap metode"):
            st.markdown("""
        **Metode Ward** adalah metode linkage yang mengoptimalkan jumlah kuadrat perbedaan dalam mengelompokkan data. 
        Metode ini menggabungkan dua kluster dengan cara yang meminimalkan peningkatan total variansi dalam kluster yang terbentuk.
        Ward biasanya menghasilkan kluster yang lebih seimbang dan cocok untuk data dengan banyak fitur.
        
        **Metode Complete (atau Maximum Linkage)** mengukur jarak antar dua kluster berdasarkan jarak terjauh antara dua titik dalam kluster yang berbeda.
        Metode ini mengutamakan kluster yang lebih homogen dan lebih sensitif terhadap outlier karena mempertimbangkan jarak maksimum.
        
        **Metode Average Linkage** mengukur jarak antar dua kluster berdasarkan rata-rata jarak antara semua pasangan titik dalam dua kluster yang berbeda.
        Metode ini lebih seimbang dibandingkan dengan metode lengkap dan lebih toleran terhadap outlier.
        
        **Metode Single (atau Nearest Point Linkage)** mengukur jarak antar dua kluster berdasarkan jarak terdekat antara dua titik dalam kluster yang berbeda.
        Metode ini seringkali menghasilkan kluster yang lebih memanjang dan sensitif terhadap noise atau outlier.
        """)

        method = st.sidebar.selectbox("Pilih metode linkage:", ["ward", "complete", "average", "single"], index=0)
        linkage_matrix = linkage(scaled_data, method=method)
        
        
        st.write("Dendrogram:")
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(linkage_matrix, labels=data.index, truncate_mode="level", p=5)
        plt.title("Dendrogram")
        plt.xlabel("Data Points")
        plt.ylabel("Distance")
        st.pyplot(fig)
         
        st.sidebar.subheader("Pengaturan Kluster")
        # Tentukan jumlah kluster berdasarkan slider
        num_clusters = st.sidebar.slider("Pilih jumlah kluster:", min_value=2, max_value=10, value=3, step=1)
        cluster_labels = fcluster(linkage_matrix, num_clusters, criterion="maxclust")
        data["Cluster"] = cluster_labels
        st.write("Dataset dengan kluster:")
        st.dataframe(data)

    # Evaluasi performa Hierarchical Clustering
        st.subheader("Evaluasi Performa Hierarchical Clustering")
        silhouette_avg = silhouette_score(scaled_data, data["Cluster"])
        dbi = davies_bouldin_score(scaled_data, data["Cluster"])
        chi = calinski_harabasz_score(scaled_data, data["Cluster"])

        st.write(f"- **Silhouette Score**: {silhouette_avg:.2f} (Semakin mendekati 1, semakin baik kluster yang terbentuk)")
        st.write(f"- **Davies-Bouldin Index**: {dbi:.2f} (Semakin kecil, semakin baik)")
        st.write(f"- **Calinski-Harabasz Index**: {chi:.2f} (Semakin besar, semakin baik)")

        
        # Visualisasi PCA untuk Hierarchical Clustering
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)
        data["PCA1"] = reduced_data[:, 0]
        data["PCA2"] = reduced_data[:, 1]

        st.write("Visualisasi Kluster:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", palette="tab10", data=data, ax=ax)
        plt.title("Visualisasi Hasil Hierarchical Clustering dengan PCA")
        st.pyplot(fig)

if algorithm == "KMeans":
    st.subheader("KMeans Clustering")
    
        # Tambahkan Elbow Method
    st.write("### Metode Elbow")
    st.write("Metode Elbow digunakan untuk menentukan jumlah kluster optimal dengan menghitung inertia (jumlah kuadrat jarak dalam kluster).")
    max_clusters = st.slider("Jumlah maksimum kluster untuk Elbow Method:", 2, 15, 10)
    inertia_values = []

    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia_values.append(kmeans.inertia_)
        
        # Visualisasi Elbow Method
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), inertia_values, marker='o', linestyle='--', color='b')
    plt.title("Elbow Method")
    plt.xlabel("Jumlah Kluster")
    plt.ylabel("Inertia")
    plt.grid()
    st.pyplot(fig)

    st.write("Dari grafik di atas, Anda dapat memilih jumlah kluster yang optimal pada titik 'elbow', yaitu ketika penurunan inertia mulai melambat.")
    kmeans_clusters = st.sidebar.slider("Jumlah kluster:", 2, 10, 3)
    kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42)
    data["Cluster"] = kmeans.fit_predict(scaled_data)
    

    st.write("Hasil KMeans Clustering:")
    st.dataframe(data)

    # Evaluasi KMeans Clustering
    st.subheader("Evaluasi Performa KMeans")
    silhouette_avg = silhouette_score(scaled_data, data["Cluster"])
    dbi = davies_bouldin_score(scaled_data, data["Cluster"])
    chi = calinski_harabasz_score(scaled_data, data["Cluster"])

    st.write(f"- **Silhouette Score**: {silhouette_avg:.2f} (Semakin mendekati 1, semakin baik kluster yang terbentuk)")
    st.write(f"- **Davies-Bouldin Index**: {dbi:.2f} (Semakin kecil, semakin baik)")
    st.write(f"- **Calinski-Harabasz Index**: {chi:.2f} (Semakin besar, semakin baik)")

    # Tambahkan diagram visual jika diperlukan
    st.write("Visualisasi dengan PCA:")
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
    data["PCA1"] = reduced_data[:, 0]
    data["PCA2"] = reduced_data[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", palette="viridis", data=data, ax=ax)
    plt.title("Visualisasi Hasil Clustering dengan PCA")
    st.pyplot(fig)

st.subheader("Diagram Metrik Kluster")

cluster_counts = data["Cluster"].value_counts()
total_customers = len(data)

# Membuat plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=data, x="Cluster", palette="viridis", ax=ax)

# Menambahkan label dan persentase dengan penyesuaian tampilan
for p in ax.patches:
    height = p.get_height()
    percentage = (height / total_customers) * 100
    ax.annotate(f'{height} ({percentage:.2f}%)',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=9, color='black', 
                xytext=(0, 5), textcoords='offset points')  # Menambahkan spasi agar label tidak bertumpuk

plt.title("Distribusi Pelanggan berdasarkan Kluster", fontsize=16)
plt.xlabel("Kluster", fontsize=14)
plt.ylabel("Jumlah Pelanggan", fontsize=14)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)

st.subheader("Metrik Kluster")

cluster_counts = data["Cluster"].value_counts()
tabs = [f"Kluster {cluster_num}" for cluster_num in cluster_counts.index]
tab_objs = st.tabs(tabs)

for idx, cluster_num in enumerate(cluster_counts.index):
    with tab_objs[idx]:
        count = cluster_counts[cluster_num]
        percentage = (count / len(data)) * 100
        st.markdown(f"### Kluster {cluster_num}")
        st.metric(
            label="Jumlah Pelanggan", 
            value=f"{count}", 
            delta=f"{percentage:.2f}% dari total pelanggan", 
            delta_color="normal"
        )
        
        
# Penjelasan tentang hasil clustering menggunakan st.expander
st.subheader("Penjelasan tentang Hasil Clustering")

with st.expander("Klik untuk melihat penjelasan lebih lanjut tentang hasil clustering"):
    st.markdown("""
    Dalam dataset ini, terdapat beberapa fitur yang relevan untuk melakukan segmentasi pelanggan, yaitu **Sex**, **Marital status**, **Age**, **Education**, **Income**, **Occupation**, dan **Settlement size**. Hasil clustering akan membantu dalam mengelompokkan individu berdasarkan kesamaan pada atribut-atribut tersebut, yang dapat digunakan untuk segmentasi pelanggan.

    Berikut adalah penjelasan bagaimana hasil clustering dapat digunakan dalam konteks dataset ini:

    1. **Segmentasi Berdasarkan Jenis Kelamin (Sex)**:
       - Misalnya, dengan clustering, kita bisa mengidentifikasi segmen pelanggan yang didominasi oleh jenis kelamin tertentu (misalnya pria atau wanita). Hal ini dapat digunakan untuk menyesuaikan strategi pemasaran, seperti iklan yang lebih relevan bagi setiap kelompok.

    2. **Segmentasi Berdasarkan Status Perkawinan (Marital status)**:
       - Pelanggan yang sudah menikah mungkin memiliki kebutuhan atau perilaku yang berbeda dibandingkan dengan yang belum menikah. Misalnya, mereka mungkin lebih tertarik pada produk atau layanan keluarga. Clustering dapat membantu mengelompokkan mereka ke dalam segmen yang sesuai untuk penawaran yang lebih spesifik.

    3. **Segmentasi Berdasarkan Usia (Age)**:
       - Kelompok usia yang berbeda mungkin memiliki preferensi produk atau layanan yang berbeda. Misalnya, pelanggan yang lebih muda mungkin lebih tertarik pada teknologi atau mode, sementara pelanggan yang lebih tua mungkin lebih fokus pada produk yang mendukung kenyamanan dan kesehatan. Clustering dapat membantu mengidentifikasi kelompok usia ini untuk pemasaran yang lebih tertarget.

    4. **Segmentasi Berdasarkan Pendidikan (Education)**:
       - Tingkat pendidikan dapat mempengaruhi preferensi atau daya beli seseorang. Mereka yang memiliki pendidikan lebih tinggi mungkin lebih tertarik pada produk atau layanan yang lebih canggih atau berbasis pengetahuan. Clustering dapat membantu memetakan segmen-segmen ini dan merancang penawaran yang lebih sesuai.

    5. **Segmentasi Berdasarkan Penghasilan (Income)**:
       - Penghasilan adalah salah satu faktor yang paling mempengaruhi perilaku pembelian. Individu dengan penghasilan tinggi mungkin tertarik pada produk premium, sementara mereka yang berpenghasilan lebih rendah mungkin lebih tertarik pada produk yang lebih terjangkau. Hasil clustering bisa menunjukkan kelompok-kelompok pelanggan berdasarkan kisaran penghasilan untuk penawaran yang lebih relevan.

    6. **Segmentasi Berdasarkan Pekerjaan (Occupation)**:
       - Pekerjaan seseorang dapat mempengaruhi kebutuhan atau preferensi mereka terhadap produk atau layanan. Misalnya, profesional mungkin lebih tertarik pada produk yang mendukung gaya hidup sibuk atau pekerjaan mereka, sementara pekerja di sektor lain mungkin lebih tertarik pada produk yang lebih praktis atau ekonomis.

    7. **Segmentasi Berdasarkan Ukuran Pemukiman (Settlement size)**:
       - Ukuran pemukiman tempat seseorang tinggal dapat memberikan wawasan tentang gaya hidup mereka. Individu yang tinggal di pemukiman kecil mungkin memiliki preferensi yang berbeda dibandingkan dengan mereka yang tinggal di pemukiman besar. Clustering dapat digunakan untuk membedakan segmen ini dan menyesuaikan penawaran produk atau layanan berdasarkan kondisi tempat tinggal mereka.

    Dengan menggunakan hasil clustering berdasarkan dataset ini, Anda dapat mengidentifikasi kelompok pelanggan yang memiliki kesamaan dalam satu atau lebih atribut, yang kemudian dapat digunakan untuk menyusun strategi pemasaran yang lebih efektif dan menyasar segmen pelanggan yang lebih tepat.
    """)


st.subheader("Unduh Hasil")
# Mengunduh dalam format CSV
csv = data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Hasil Clustering (CSV)",
    data=csv,
    file_name="data_hasil_klustering.csv",
    mime="text/csv",
    use_container_width=True
)

# Mengunduh dalam format JSON
json_data = data.to_json(orient='records')
st.download_button(
    label="Download Hasil Clustering (JSON)",
    data=json_data,
    file_name="data_hasil_klustering.json",
    mime="application/json",
    use_container_width=True
)

# # Sidebar - Tentang Saya
# with st.sidebar.markdown:
#     st.header("Tentang Saya")
#     st.markdown("""
#     **Nama**: Roni  
#     **NIM**: 211220108  
#     **Email**: [Email](mailto:ronn.7ex@gmail.com)  
#     **Github**: [Github](https://github.com/ronirn/)  
#     **LinkedIn**: [Profil LinkedIn Saya](https://www.linkedin.com/in/roni-ansyah/)  
#     **Dribbble**: [Portofolio Dribbble Saya](https://dribbble.com/RONI_ANSYAH)  
#     """)

# Watermark Teks di Tengah Bawah
st.markdown(
    """
    <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); font-size: 14px; color: rgba(0, 0, 0, 0.5);">
        Dibuat oleh Roni
    </div>
    """, 
    unsafe_allow_html=True
)