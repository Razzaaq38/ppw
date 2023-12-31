# program1.py (contoh salah satu program menggunakan Streamlit)
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import validators

def scrap():
        data = []
        url = 'https://pta.trunojoyo.ac.id/c_search/byprod/9/'
        for page in range(1, 143):
            req = requests.get(url + str(page))
            soup = BeautifulSoup(req.text, 'html.parser')
            items = soup.findAll('li', {'data-cat': '#luxury'})

            for item in items:
                link = item.find('a', 'gray button')['href']
                st.write(link)

                req2 = requests.get(link)
                soup2 = BeautifulSoup(req2.text, 'html.parser')

                penulis_elem = soup2.find('div', {'style': 'padding:2px 2px 2px 2px;'}).find('span')
                penulis = penulis_elem.text
                st.write(penulis)

                dospem_elem = soup2.find('div', {'style': 'float:left; width:540px;'}).findAll('div', {'style': 'padding:2px 2px 2px 2px;'})

                dospem_i = 'Dosen Pembimbing I tidak ditemukan'
                dospem_ii = 'Dosen Pembimbing II tidak ditemukan'

                for dospem in dospem_elem:
                    dospem_text = dospem.find('span').text
                    if 'Dosen Pembimbing I :' in dospem_text:
                        dospem_i = dospem_text.replace('Dosen Pembimbing I :', '').strip()
                    elif 'Dosen Pembimbing II :' in dospem_text:
                        dospem_ii = dospem_text.replace('Dosen Pembimbing II :', '').strip()
                st.write("Dosen Pembimbing I :", dospem_i, "\nDosen Pembimbing II :", dospem_ii)

                judul_elem = item.find('a', 'title')
                judul = judul_elem.text
                st.write(judul)

                absk_elem = soup2.find('div', {'style': 'margin: 15px 15px 15px 15px;'}).find('p')
                absk = absk_elem.text if absk_elem else 'Abstrak tidak ditemukan'
                st.write(absk)

                data.append([judul, penulis, dospem_i, dospem_ii, absk])

        pta = pd.DataFrame(data, columns=['Judul', 'penulis', 'Dosen Pembimbing I', 'Dosen Pembimbing II', 'Abstrak'])

        return pta

def dat(link):
    data = pd.read_csv(link)

    data = data.dropna(subset=['Abstrak'])
    data = data.reset_index(drop=True)

    #data['Abstrak'].fillna('', inplace=True)
    jumlah_entri = data.shape[0]
    return data
    
##############################################################################################################################################
def run():
    st.title("Tugas 1 PTA")
    tab1, tab2, tab3, tab4, tab5, tab6, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs(
        ["Crawling", "Data", "Preproses", "Stopword", "Penggabungan",
         "LDA", "Judul", "Tabel LDA", "K-Means", "Label", "Klastering", "Filtrasi"])

    with tab1:
        if st.button("Mulai Scraping"):
            pta = scrap()
            csv_data = pta.to_csv(index=False)

            st.success("Scraping Selesai! Data tersimpan di 'ptaindustri.csv'")
            # Membuat tombol unduhan
            st.download_button(
                label="Download data as CSV",
                data=csv_data.encode('utf-8'),  # Menggunakan encode utf-8
                file_name='ptaindustri.csv',
                mime='text/csv',
            )    

    with tab2:
        st.title('CSV Data Viewer')

        user_input = st.text_input("Masukkan URL", "https://raw.githubusercontent.com/Razzaaq38/ppw/main/data_crawling.csv")

        if user_input != "https://":
            if validators.url(user_input):
                st.write("URL valid:", user_input)
                data = dat(user_input)
                data['Abstrak'] = data['Abstrak'].str.lower()
                data_lower_case = data['Abstrak']
                st.write("Data Kolom 'Abstrak' dalam huruf kecil:")
                st.write(data_lower_case)
            else:
                st.write("URL tidak valid. Masukkan URL yang valid.")

    with tab3:
        st.title('Preproses')
        clean = []

        for i in range(len(data['Abstrak'])):
            clean_symbols = re.sub("[^a-zA-Zï ]+", " ", data['Abstrak'].iloc[i])  # Pembersihan karakter
            clean_tag = re.sub("@[A-Za-z0-9_]+", "", clean_symbols)  # Pembersihan mention
            clean_hashtag = re.sub("#[A-Za-z0-9_]+", "", clean_tag)  # Pembersihan hashtag
            clean_https = re.sub(r'http\S+', '', clean_hashtag)  # Pembersihan URL link
            clean_whitespace = re.sub(r'\s+', ' ', clean_https).strip()  # Mengganti spasi berlebih dengan spasi tunggal
            clean.append(clean_whitespace)


        clean_result = pd.DataFrame(clean,columns=['Cleansing Abstrak'])
        st.write(clean_result)

    with tab4:
        st.title('Slank Word')
        slang_dict = pd.read_csv("https://raw.githubusercontent.com/noneneedme/ppw/main/combined_slang_words.txt", sep=" ", header=None)

        # Membuat fungsi untuk mengubah slang words menjadi kata Indonesia yang benar
        def replace_slang_words(text):
            words = nltk.word_tokenize(text.lower())
            words_filtered = [word for word in words if word not in stopwords.words('indonesian')]
            for i in range(len(words_filtered)):
                if words_filtered[i] in slang_dict:
                    words_filtered[i] = slang_dict[words_filtered[i]]
            return ' '.join(words_filtered)

        # Contoh penggunaan

        slang_words=[]
        for i in range(len(clean)):
          slang = replace_slang_words(clean[i])
          slang_words.append(slang)

        data_slang = pd.DataFrame(slang_words, columns=["Slang Word Corection"])
        st.write(data_slang)

    with tab5:
        st.title('Kustom Stopword')
        # Daftar kata yang ingin ditambahkan sebagai stopword
        custom_stopwords = ['ab', 'zam', 'abai', 'aadalah', 'abdoel', 'ppp']

        words = []
        for i in range (len(data_slang)):
          tokens = word_tokenize(slang_words[i])
          listStopword =  set(stopwords.words('indonesian'))

          # Menambahkan kata-kata custom ke dalam set stopword
          listStopword.update(custom_stopwords)

          removed = []
          for t in tokens:
              if t not in listStopword:
                  removed.append(t)

          words.append(removed)
          st.write(removed)

    with tab6:
        st.title('Join Kata')
        gabung=[]
        for i in range(len(words)):
          joinkata = ' '.join(words[i])
          gabung.append(joinkata)

        result = pd.DataFrame(gabung, columns=['Join_Kata'])
        st.write(result)

    with tab8:
        st.title('LDA')
        st.write('Latent Dirichlet Allocation')
        lda = LatentDirichletAllocation(n_components=2, doc_topic_prior=0.2, topic_word_prior=0.1,random_state=42,max_iter=1)
        lda_top=lda.fit_transform(df_countvect)

        st.write(lda_top.shape)
        st.write(lda_top)
        st.write(lda.components_)
        st.write(lda.components_.shape)

    with tab9:
        st.title('Topik dan Judul')

        st.write('Topik')
        topics = pd.DataFrame(lda_top, columns=['Topik 1','Topik 2'])
        st.write(topics)

        st.write('Judul')
        judul=data['Judul']
        st.write('judul')

        st.write('Gabung')
        gabung = pd.concat([judul, topics], axis=1)
        st.write(gabung)

    with tab10:
        st.title('Label')
        label=[]
        for i in range (1,(lda.components_.shape[1]+1)):
          masukan = df_countvect.columns[i-1]
          label.append(masukan)
        VT_tabel = pd.DataFrame(lda.components_,columns=label)
        VT_tabel.rename(index={0:"Topik 1",1:"Topik 2",2:"Topik 3"}).transpose()

        st.write(VT_tabel)

    with tab11:
        st.title('K-Means')
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(lda_top)
        cluster_labels = kmeans.labels_
        data = {'Dokumen': range(len(cluster_labels)), 'Cluster': cluster_labels}
        duf = pd.DataFrame(data)
        st.write(duf)

    with tab12:
        st.title('Klastering')
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(lda_top)
        cluster_labels = kmeans.labels_

        # Hitung Silhouette Coefficient
        silhouette_avg = silhouette_score(lda_top, cluster_labels)
        st.write("Silhouette Coefficient:", silhouette_avg)

    with tab13:
        st.title('Filtrasi')
        # Filter dokumen dengan cluster 0 dan 1
        cluster_0_documents = duf[duf['Cluster'] == 0]
        cluster_1_documents = duf[duf['Cluster'] == 1]

        # Ambil judul dokumen untuk masing-masing cluster
        cluster_0_document_titles = judul[cluster_0_documents['Dokumen']]
        cluster_1_document_titles = judul[cluster_1_documents['Dokumen']]

        # Buat DataFrame untuk masing-masing cluster
        cluster_0_df = pd.DataFrame({'Judul Dokumen': cluster_0_document_titles, 'Dokumen': cluster_0_documents['Dokumen'], 'Cluster': cluster_0_documents['Cluster']})
        cluster_1_df = pd.DataFrame({'Judul Dokumen': cluster_1_document_titles, 'Dokumen': cluster_1_documents['Dokumen'], 'Cluster': cluster_1_documents['Cluster']})

        # Tampilkan tabel untuk Cluster 0
        st.write("Dokumen Cluster 0:")
        st.write(cluster_0_df)

        # Tampilkan tabel untuk Cluster 1
        st.write("\nDokumen Cluster 1:")
        st.write(cluster_1_df)
