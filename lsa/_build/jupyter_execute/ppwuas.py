#!/usr/bin/env python
# coding: utf-8

# # Crawling

# Data crawling adalah program yang menghubungkan halaman web, kemudian mengunduh kontennya. Program crawling dalam data science hanya akan online untuk mencari dua hal, yaitu data yang dicari oleh pengguna dan penjelajahan target dengan jangkauan yang lebih luas.

# In[1]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
stop_words=set(nltk.corpus.stopwords.words('english'))


# In[2]:


df = pd.read_csv('./hasilmanajemen.csv')


# In[15]:


df.head(10)


# In[16]:


df.drop(['judul'],axis=1,inplace=True)
df.drop(['penulis'],axis=1,inplace=True)
df.drop(['dosen_1'],axis=1,inplace=True)
df.drop(['dosen_2'],axis=1,inplace=True)


# In[17]:


df.head(10)


# In[18]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000) # to play with. min_df,max_df,max_features etc...


# In[19]:


vect_text=vect.fit_transform(df['abstrak'].values.astype('U'))


# In[20]:


print(vect_text.shape)
#print(vect_text)
type(vect_text)
df = pd.DataFrame(vect_text.toarray())
print(df)
idf=vect.idf_


# In[21]:


idf=vect.idf_


# In[22]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(l[0],l[-1])
print(dd['dan'])
print(dd['yamaha'])  # police is most common and forecast is least common among the news headlines.


# <h1>LSA</h1>

# <p>Latent Semantic Analysis (LSA) merupakan sebuah metode yang memanfaatkan model statistik matematis untuk menganalisa struktur semantik suatu teks. LSA bisa digunakan untuk menilai esai dengan mengkonversikan esai menjadi matriks-matriks yang diberi nilai pada masing-masing term untuk dicari kesamaan dengan term referensi.</p>

# <h1>Algoritma LSA</h1>

# <p>Tahapan-tahapan algoritma LSA dalam prosessing teks</p>

# <h1>Menghitung Term-document Matrix</h1> 

# <p>Document Term Matrix merupakan algoritma – Metode perhitungan yang sering kita temui dalam text minning.</p>

# <p>Melalui Document Term Matrix, kita dapat melakukan analisis yang lebih menarik. Mudah untuk menentukan jumlah kata individual untuk setiap dokumen atau untuk semua dokumen. Misalkan untuk menghitung agregat dan statistik dasar seperti jumlah istilah rata-rata, mean, median, mode, varians, dan deviasi standar dari panjang dokumen, serta dapat mengetahui istilah mana yang lebih sering dalam kumpulan dokumen dan dapat menggunakan informasi tersebut untuk menentukan istilah mana yang lebih mungkin “mewakili” dokumen tersebut.</p>

# <h1>Singular Value Decomposition</h1>

# <p>Singular Value Decomposition adalah seuatu teknik untuk mendekomposisi matriks berukuran apa saja (biasanya diaplikasikan untuk matriks dengan ukuran sangat besar), untuk mempermudah pengolahan data. Hasil dari SVD ini adalah singular value yang disimpan dalam sebuah matriks diagonal, D,  dalam urutan yang sesuai dengan koresponding singular vector-ya.</p>

# In[23]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[24]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[25]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# In[26]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# <h1>Mengekstrak Topik dan Term</h1>

# In[27]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# <h1>K-Means</h1>

# In[28]:


pip install pandas


# In[34]:


import pandas as pd
import numpy as np
#Import Library untuk Tokenisasi
import string 
import re #regex library

# import word_tokenize & FreqDist dari NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords

import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_blobs
from sklearn.metrics import davies_bouldin_score


# In[35]:


true_k = 4
model = KMeans(n_clusters= true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(vect_text)


# In[36]:


print("kata teratas per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vect.get_feature_names()
for i in range(4):
    print("cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print


# In[37]:


print( "\n" ) 
print( "Prediksi" )

Y = vect.transform([ "mengetahui media pemasaran" ]) 
prediksi = model.predict(Y) 
print(prediksi)


# In[38]:


print(order_centroids)


# In[ ]:




