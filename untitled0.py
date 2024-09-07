# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

'''from keras_preprocessing.text import Tokenizer 
ya da bu 
from tensorflow.keras.preprocessing.text import Tokenizer'''

import re
from zeyrek import MorphAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import keras_tuner as kt

# Veri setini yükleme
yorumlar = pd.read_csv('veri_seti.csv', sep=',', header=None, names=['sonuç', 'yorum'])

"noktalama işaretleri,sembolleri sil"
import re

yorum = ""
sonuçlar = []  # olumsuz kelime hariç, verilerin son halidir
stopwords = []  # türkçe stopwordsler
veriler = []  # verilerin son halidir (olumsuz kelimeler dahil)
çıkartılan = []  # olumsuz kelimelerin eklerini çıkartılmasını engellemek için

"türkçe stopwordsleri dosyadan okuyup arraye aktar"
with open("stopwords.txt", 'r', encoding='utf-8') as dosya:
    for satir in dosya:
        stopwords.append(satir.strip())  # Her satırı diziye ekle, strip() ile gereksiz boşlukları temizle

"verileri küçültme ve kelime olarak split et"


def veri(yorum, i):
    "büyük-küçük harf problemi: hepsini küçült"
    yorum = yorum.lower()

    "yorumu listeye çevir"
    yorum = yorum.split()


"stopwords'den arındır"


def removeStopwords(yorum):
    yorum_list = yorum.split()  # Split the string into a list of words
    index = 0
    while index < len(yorum_list):
        kelime = yorum_list[index]
        if kelime in stopwords:
            yorum_list.pop(index)
        else:
            index += 1
    return ' '.join(yorum_list)  # Join the list back into a string and return


"olumsuz eki çıkartmama"


def removeNegativeWord(yorum):
    index = 0
    while index < len(yorum):
        kelime = yorum[index]
        if "sız" in kelime or "siz" in kelime or "suz" in kelime or "süz" in kelime:
            çıkartılan.append(yorum[index])
            yorum.remove(yorum[index])
        else:
            index += 1


"gövde ve eki ayrıştırma işlemi"
from zeyrek import MorphAnalyzer

zeyrek = MorphAnalyzer()


def stemmer(yorum):
    kelimeler = yorum.split()  # Stringi kelimelere ayır
    for kelime in kelimeler:  # Her bir kelime için döngüyü çalıştır
        sonuç = zeyrek.lemmatize(kelime)  # Her kelimenin kökünü bul
        sonuçlar.append(min(sonuç[0][1], key=len).lower())  # En kısa kökü seç ve results listesine ekle


def main():
    for i in range(len(yorumlar)):
        yorum = re.sub('[^a-zA-ZçğıöşüÇĞİÖŞÜ]', ' ', yorumlar["yorum"][i])
        veri(yorum, i)
        removeStopwords(yorum)
        removeNegativeWord(yorum)
        stemmer(yorum)
        sonSonuç = sonuçlar + çıkartılan
        sonSonuç = ' '.join(sonSonuç)
        veriler.append(sonSonuç)
        çıkartılan.clear()
        sonuçlar.clear()


main()

# Vektör sayacı
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(veriler).toarray() # Bağımsız değişken
y = yorumlar["sonuç"].values

# Makine Öğrenmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=None)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Hata matrisi hesaplama
y_predict = gnb.predict(X_test)
cm = confusion_matrix(y_test, y_predict)

print("Makine Öğrenmesi Doğruluğu:", (cm[0,0] + cm[1,1]) / np.sum(cm) * 100)
print(cm)  # Hata matrisi

# Veri setini yükleme
yorumlar = pd.read_csv('veri_seti.csv', sep=',', header=None, names=['sonuç', 'yorum'])

# Derin Öğrenme Modeli
X = yorumlar['yorum'].values
y = yorumlar['sonuç'].values

# Etiketleri sayısal değerlere dönüştürme
le = LabelEncoder()
y = le.fit_transform(y)

# Metin verisini sayısal vektörlere dönüştürme
max_words = 1000
max_len = 150

tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X = tf.keras.utils.pad_sequences(sequences, maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiperparametre ayarlamaları için model fonksiyonu
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(
        input_dim=max_words,
        output_dim=hp.Int('embedding_dim', min_value=100, max_value=300, step=50),
        input_length=max_len
    ))
    model.add(keras.layers.SpatialDropout1D(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(keras.layers.LSTM(
        units=hp.Int('lstm_units', min_value=50, max_value=200, step=50),
        dropout=hp.Float('recurrent_dropout_rate', min_value=0.2, max_value=0.5, step=0.1),
        recurrent_dropout=hp.Float('recurrent_dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    ))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Keras Tuner araması
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    project_name='text_classification'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[stop_early])

# En iyi modeli al ve eğit
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[stop_early])

# Modeli değerlendirme
score = model.evaluate(X_test, y_test, verbose=0)
print("Deep Learning Doğruluğu:", score[1] * 100)
