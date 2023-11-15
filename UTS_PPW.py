import streamlit as st
import pandas as pd
import re
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
import requests
from bs4 import BeautifulSoup
import csv

def crawl():
    data = []
    url = 'https://pta.trunojoyo.ac.id/c_search/byprod/7/'
    for page in range(1, 207):
        # Mengambil data dari halaman web dengan web scraping
        req = requests.get(url + str(page))
        soup = BeautifulSoup(req.text, 'html.parser')
        items = soup.findAll('li', {'data-cat': '#luxury'})

        for item in items:
            # Mengambil informasi judul, penulis, dosen pembimbing, dan abstrak dari halaman tesis
            # dan menyimpannya dalam data list
            link = item.find('a', 'gray button')['href']
            print(link)

            req2 = requests.get(link)
            soup2 = BeautifulSoup(req2.text, 'html.parser')

            penulis_elem = soup2.find('div', {'style': 'padding:2px 2px 2px 2px;'}).find('span')
            penulis = penulis_elem.text
            print(penulis)

            dospem_elem = soup2.find('div', {'style': 'float:left; width:540px;'}).findAll('div', {'style': 'padding:2px 2px 2px 2px;'})

            dospem_i = 'Dosen Pembimbing I tidak ditemukan'
            dospem_ii = 'Dosen Pembimbing II tidak ditemukan'

            for dospem in dospem_elem:
                dospem_text = dospem.find('span').text
                if 'Dosen Pembimbing I :' in dospem_text:
                    dospem_i = dospem_text.replace('Dosen Pembimbing I :', '').strip()
                elif 'Dosen Pembimbing II :' in dospem_text:
                    dospem_ii = dospem_text.replace('Dosen Pembimbing II :', '').strip()
            print("Dosen Pembimbing I :", dospem_i, "\nDosen Pembimbing II :", dospem_ii)

            judul_elem = item.find('a', 'title')
            judul = judul_elem.text
            print(judul)

            absk_elem = soup2.find('div', {'style': 'margin: 15px 15px 15px 15px;'}).find('p')
            absk = absk_elem.text if absk_elem else 'Abstrak tidak ditemukan'
            print(absk)

            data.append([judul, penulis, dospem_i, dospem_ii, absk])

    pta = pd.DataFrame(data, columns=['Judul', 'penulis', 'Dosen Pembimbing I', 'Dosen Pembimbing II', 'Abstrak'])
    return pta

## STREAMLIT TAB ##                                                                                                     ## STREAMLIT TAB ##


## STREAMLIT TAB ##                                                                                                     ## STREAMLIT TAB ##
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Crawling/Scraping", "Preprocessing", "Cleansing", "TF-IDF", "Model LDA", "Bobot Topik pada Dokumen", "Bobot Kata Terhadap Topik"])

with tab1:
    #Tahap ini berfokus pada pengambilan data dari situs web.
    #Dalam loop, kode mengirim permintaan HTTP ke halaman web,
    #mengurai halaman web dengan BeautifulSoup, dan mengambil
    # informasi seperti judul tesis, penulis, dosen pembimbing,
    # dan abstrak

    st.header("Crawling/Scraping")
    data = pd.read_csv("https://raw.githubusercontent.com/Razzaaq38/ppw/main/data_crawling.csv")
    st.write(data)
    judul = data['Judul']
    if st.button("Jalankan"):
        pta=crawl()
        data=pta
        st.dataframe(pta)
    # Data yang diambil dari setiap halaman web disimpan dalam bentuk list data.
    # data yang telah diambil digabungkan menjadi DataFrame pta.
    
with tab2:
    st.header("Preprocessing")
    # Menghapus data yang memiliki nilai null pada kolom 'Abstrak'
    # Mengubah teks dalam kolom 'Abstrak' menjadi huruf kecil (lowercase)
    data = data.dropna(subset=['Abstrak'])
    data = data.reset_index(drop=True)
    data['Abstrak'] = data['Abstrak'].str.lower()
    data_lower_case = data['Abstrak']
    st.dataframe(data_lower_case)
    # Hasilnya, DataFrame data diubah menjadi DataFrame
    # data_lower_case yang berisi 'Abstrak' yang telah diubah
    # menjadi huruf kecil.

with tab3:
    st.header("Cleansing")
    clean = []

    #Tahap ini bertujuan untuk membersihkan teks dalam kolom 'Abstrak'.

    for i in range(len(data['Abstrak'])):
        # Melakukan pembersihan teks dengan menghapus karakter khusus, mention, hashtag, dan URL
        clean_symbols = re.sub("[^a-zA-Zï ]+", " ", data['Abstrak'].iloc[i])  # Pembersihan karakter
        clean_tag = re.sub("@[A-Za-z0-9_]+", "", clean_symbols)  # Pembersihan mention
        clean_hashtag = re.sub("#[A-Za-z0-9_]+", "", clean_tag)  # Pembersihan hashtag
        clean_https = re.sub(r'http\S+', '', clean_hashtag)  # Pembersihan URL link
        clean_whitespace = re.sub(r'\s+', ' ', clean_https).strip()  # Mengganti spasi berlebih dengan spasi tunggal
        clean.append(clean_whitespace)

        # spasi berlebih diganti dengan spasi tunggal.
    clean_result = pd.DataFrame(clean,columns=['Cleansing Abstrak'])
    st.dataframe(clean_result)
    #Hasil dari pembersihan ini disimpan dalam DataFrame clean_result.
    

with tab4:
    st.header("TF-IDF")
    # Pada tahap ini, kamus slang words dan kata-kata Indonesia yang benar
    # digunakan untuk mengganti kata-kata slang dalam teks.

    # teks dibagi menjadi kata-kata (tokenisasi) dan stop words dihapus.
    # Tokenisasi kata-kata dalam teks dan menghapus stop words
    # Menggunakan TfidfVectorizer dan CountVectorizer untuk menghitung TF-IDF dan frekuensi kata-kata

    # Membuat kamus slang words dan kata Indonesia yang benar
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
    ##########
    words = []
    for i in range (len(data_slang)):
        tokens = word_tokenize(slang_words[i])
        listStopword =  set(stopwords.words('indonesian'))

        removed = []
        for t in tokens:
          if t not in listStopword:
              removed.append(t)

        words.append(removed)
    ###########  
    gabung=[]
    for i in range(len(words)):
        joinkata = ' '.join(words[i])
        gabung.append(joinkata)

    result = pd.DataFrame(gabung, columns=['Join_Kata'])

    result = result.dropna(subset=['Join_Kata'])
    df=result

    # Extract the 'Join_Kata' column
    gabung = df['Join_Kata'].tolist()

    # TfidfVectorizer
    tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    tfidf_wm = tfidfvectorizer.fit_transform(gabung)
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens)

    # CountVectorizer
    countvectorizer = CountVectorizer(analyzer='word', stop_words='english')
    count_wm = countvectorizer.fit_transform(gabung)
    count_tokens = countvectorizer.get_feature_names_out()
    df_countvect = pd.DataFrame(data=count_wm.toarray(), columns=count_tokens)

    #digunakan TfidfVectorizer dan CountVectorizer untuk menghitung TF-IDF dan frekuensi kata-kata.

    st.dataframe(df_countvect)
    # Hasilnya, bobot kata-kata pada dokumen disimpan dalam DataFrame df_countvect.
    
with tab5:
    #LDA digunakan untuk mengidentifikasi topik-topik dalam dataset dan 
    # menghitung bobot topik pada setiap dokumen.
    st.header("Model LDA")
    lda = LatentDirichletAllocation(n_components=3, doc_topic_prior=0.2, topic_word_prior=0.1,random_state=42,max_iter=1)
    lda_top=lda.fit_transform(df_countvect)
    # Penerapan model LDA pada data yang telah diproses sebelumnya
    st.dataframe(lda_top)
    # Hasilnya disimpan dalam DataFrame lda_top.
    
with tab6:
    #Pada tahap ini, judul tesis digabungkan dengan bobot topik dari LDA.
    #Hal ini memungkinkan untuk melihat hubungan antara judul tesis dan topik-topik yang ditemukan oleh LDA.
    st.header("Bobot Topik pada Dokumen")
    # Menggabungkan judul tesis dengan bobot topik dari LDA
    topics = pd.DataFrame(lda_top, columns=['Topik 1','Topik 2','Topik 3'])
    gabung = pd.concat([judul, topics], axis=1)
    st.dataframe(gabung)
    #Hasil penggabungan disimpan dalam DataFrame gabung.
    
with tab7:
    #Tahap terakhir adalah menghitung bobot kata-kata terhadap 
    # setiap topik yang dihasilkan oleh model LDA.
    st.header("Bobot Kata Terhadap Topik")
    label=[]
    for i in range (1,(lda.components_.shape[1]+1)):
        masukan = df_countvect.columns[i-1]
        label.append(masukan)
    VT_tabel = pd.DataFrame(lda.components_,columns=label)
    mungkin=VT_tabel.rename(index={0:"Topik 1",1:"Topik 2",2:"Topik 3"}).transpose()
    #Menghitung bobot kata-kata terhadap setiap topik yang dihasilkan oleh model LDA
    
    st.dataframe(mungkin)

    #Dalam hal ini, dilakukan perhitungan bobot kata terhadap topik
    # dan hasilnya disajikan dalam DataFrame mungkin.

