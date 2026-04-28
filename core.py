import re
import string
from nltk.corpus import stopwords

#modulo 1
stop_words = set(stopwords.words('english'))

def preprocess_post(text: str) -> str:

    # Eliminar URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Eliminar números
    text = re.sub(r'\d+', '', text)

    # Convertir a minúsculas
    text = text.lower()

    # Eliminar puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Eliminar stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Eliminar múltiples espacios
    text = re.sub(r'\s+', ' ', text)

    return text



#modulo 2
import numpy as np
import os
import time
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import pandas as pd

def train_and_classify_subreddit(reddit_data, input_text):
    # Muestra de datos
    sampled_data = reddit_data.sample(n=30000).dropna(subset=['clean_post', 'subreddit'])
    X = sampled_data['clean_post']
    y = sampled_data['subreddit']
    sentences = [text.split() for text in sampled_data['clean_post']]

    # Verificar si el modelo Word2Vec ya existe
    model_path = "word2vec.model"
    if os.path.exists(model_path):
        word2vec = Word2Vec.load(model_path)
    else:
        start_time = time.time()
        word2vec = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
        word2vec.save(model_path)

    def vectorize_text(text: str) -> np.ndarray:
        words = text.split()
        vectors = [word2vec.wv[word] for word in words if word in word2vec.wv]  # si la palabra no está en el modelo no se tiene en cuenta
        return np.mean(vectors, axis=0) if vectors else np.zeros(300)

    # Convertir texto en vectores
    X_embeddings = X.apply(vectorize_text)

    # Convertir a un formato adecuado para sklearn
    X_embeddings = np.array(X_embeddings.tolist())

    # Dividir y entrenar el modelo
    X_train_embed, X_test_embed, y_train_embed, y_test_embed = train_test_split(X_embeddings, y, test_size=0.3, random_state=42)

    # Clasificador
    svm_embed = SVC()
    svm_embed.fit(X_train_embed, y_train_embed)

    # Función para clasificar un nuevo texto
    def classify_text(text: str) -> str:
        vector = vectorize_text(text).reshape(1, -1)
        prediction = svm_embed.predict(vector)
        return prediction[0]

    # Clasificar el texto de entrada
    result = classify_text(input_text)
    return result


#modulo 3

def find_subreddit_mentions(text: pd.Series)->list:
    """Permitirá extraer los subreddits mencionados en
    un post. Por ejemplo: “'I\'m cross posting this from /r/cyberlaw, hopefully you guys find it
    as interesting”. Se debe extraer en este caso /r/cyberlaw. En caso de que haya más de uno,se deberán extraer todos y guardarlos en una lista. Para ello, se deberá utiliza una única
    expresión regular."""
    re_mentions = r'(/r/.*?)[/\]\.,\s\)\\]'
    match = text.str.extractall(re_mentions)
    salida = match[0].unique().tolist()
    return salida

def url_extraction(text: str)->list:
    """Permitirá extraer todas las URLs en un post mediante una única expresión regular"""
    re_url = r'(http[s]?://.*?)[;\)\s]'
    match = text.str.extractall(re_url)
    return match[0].unique().tolist()

def phone_number_extracion(text: pd.Series)->list:
    """Permitirá la extracción de números de teléfono mediante una única expresión regular"""
    re_phone = r'(\d{3}-\d{3}-\d{4})'
    match = text.str.extractall(re_phone)
    salida = match[0].unique().tolist()
    return salida

def dates_extraction(text: str)->list:
    """Permitirá la extracción de todas las fechas contenidas en un post."""
    re_url = r'(\d{2}/\d{2}/\d{4})'
    match = text.str.extractall(re_url)
    salida = match[0].unique().tolist()
    return salida

def code_extraction(text:str)->list:
    """Extracción de código de programación o HTML incluido en un post. Permitirá la extracción de todo el código que se incluya en un post."""
    re_code = r'^ {4,}(.*)|(&gt.*)|(&lt.*)|(^//.*)|(\*\*.*?\*\*)|({.*})'
    match = text.str.extractall(re_code)
    match_0 = match[0].unique().tolist()# con esta buscamos sacar código identado
    match_1 = match[1].unique().tolist()#corresponnde al greater than the html
    match_2 = match[2].unique().tolist()#corresponde al less than the html
    match_3 = match[3].unique().tolist()# coresponde a comentarios de código
    match_4 = match[4].unique().tolist()# corresponde a cosas entre dobles asteriscos
    match_5 = match[5].unique().tolist()# corresponde a cosas entre llaves
    
    # usar & para extraer los html del codigo
    return match_0 + match_1 + match_2 + match_3 + match_4 + match_5


#modulo 4

from gensim.models import Word2Vec
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

def train_and_classify_sentimiento(reddit_data, input_text):
    # Muestra de datos
    reddit_data = reddit_data.sample(n=200000, random_state=42)

    # Crear el modelo Word2Vec a partir de los datos
    sentences = [text.split() for text in reddit_data['clean_post']]

    # Verificar si el modelo Word2Vec ya existe
    model_path = "word2vec.model"
    if os.path.exists(model_path):
        word2vec = Word2Vec.load(model_path)
    else:
        word2vec = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
        word2vec.save(model_path)

    def vectorize_text(text: str) -> np.ndarray:
        words = text.split()
        vectors = [word2vec.wv[word] for word in words if word in word2vec.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(300)

    X = reddit_data['clean_post']
    y = reddit_data['sentiment']

    # Convertir los posts en vectores de Word2Vec
    X_vectors = X.apply(vectorize_text)

    # Eliminar cualquier valor nulo en X_vectors y y
    X_vectors = X_vectors.dropna()
    y = y[X_vectors.index].astype(str)

    # Dividir los datos en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X_vectors.tolist(), y, test_size=0.3, random_state=42)

    # Convertir a DataFrame para eliminar filas con NaN en X_train y y_train
    X_train_df = pd.DataFrame(X_train)
    y_train_df = pd.Series(y_train.values, index=X_train_df.index)
    X_train_df = X_train_df.dropna()
    y_train_df = y_train_df[X_train_df.index]

    # Convertir de nuevo a listas
    X_train = X_train_df.values.tolist()
    y_train = y_train_df.values.tolist()

    # Entrenamiento de un clasificador Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # Función para clasificar un nuevo texto
    def classify_text(text: str) -> str:
        vector = vectorize_text(text).reshape(1, -1)
        prediction = nb.predict(vector)
        return prediction[0]

    # Clasificar el texto de entrada
    result = classify_text(input_text)
    return result

#modulo 5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re

def post_summarisation(text: pd.Series) -> pd.Series:#método propio basandonos en la teoría
    #utilizamos los datos originales sin preprocesar con la función 1
    # en el caso de querer usar datos preprocesados no pasaría nada, simplemente no haría la siguiente limpieza
    stopword = set(stopwords.words('english'))
    original = text.copy()
    # comenzamos con la limpieza del texto:
    # observando las expresiones regulares que se utilizan en el ejercicio 3, vamos a quitar, utilizo mi propio limpiador de texto
    # si veo que al tokenizar el texto, hay palabras clave que no me interesen tb las quitaré
    text = text.str.replace(r'(http[s]?://.*?)[;\)\s]', ' ', regex=True) # quitar urls
    text = text.str.replace(r'(\d{3}-\d{3}-\d{4})', ' ', regex=True) # quitar numeros de telefono
    text = text.str.replace(r'(\d{2}/\d{2}/\d{4})', ' ', regex=True) # quitar numeros de fechas
    text = text.str.replace(r'^ {4,}(.*)|(&gt.*)|(&lt.*)|(^//.*)|(\*\*.*?\*\*)|({.*})',' ', regex=True) # quitamos codigo
    text = text.str.replace(r'(\\)', ' ', regex=True)
    text = text.str.replace(r'(/r/.*?)[/\]\.,\s\)\\]', ' ', regex=True)
    text = text.apply(lambda x: x.lower())
    text = text.str.replace(r"([^\w\s,'])",'', regex=True)
    text = text.str.replace(r'(\s{2,})',' ', regex=True)
    text = text.str.replace(r'(doubleclick,)',' ', regex=True)
    text = text.str.replace(r'displaycpcgooglenetwork',' ', regex=True)
    text = text.str.replace(r'(doubleclick,)',' ', regex=True)
    text = text.str.replace(r'(doubleclick,)',' ', regex=True)
    text = text.str.replace(r'(--)',' ', regex=True)
    # removemos las stopwords
    text = text.str.split(' ').apply(lambda x: [word for word in x if word not in stopword])
    
    # ahora queremos sacar un diccionario con la frequencia de cada palabra
    def word_freq(text: list) -> dict:
        diccionario = {}
        for i in text:
            if i in diccionario:
                diccionario[i] += 1
            else:
                diccionario[i] = 1
        return diccionario
    # aplicamos la función a cada fila y creamos un serie que contiene  un diccionario de cada fila con las palabras y su frecuencia
    dic = text.apply(lambda x: word_freq(x)).apply(lambda x: dict(sorted(x.items(), key=lambda item: item[1], reverse=True)))
    
    # ahora para hemos elegido como método basado en frecuencias el tf-idf
    def frecuencia_tf(dic):# función que calcula la frecuencia tf de cada palabra
        max = list(dic.values())[0]
        return {k: round(v/max,3) for k,v in dic.items()}
    # aplicamos la función a nuestra serie diccionario y nos quedamos con otra serie ahora de frecuencias tf
    tf = dic.apply(lambda x: frecuencia_tf(x))
    
    salida_limpia_unida = text.apply(lambda x: ' '.join(x)) # nos quedamos con una copia del texto unido para calcular el itf
    
    # calculamos el logaritmo de la frecuencia inversa de documento, que es el numero de documentos dividido por el numero de documentos que contienen la palabra
    def frecuencia_itf(dict,serie):
        return {k: round(np.log(round(serie.shape[0]/serie.str.contains(k).sum(),3)),2) for k,v in dict.items()}
    # aplicamos la función a nuestra serie diccionario y nos quedamos con otra serie ahora de frecuencias itf
    itf = dic.apply(lambda x: frecuencia_itf(x,salida_limpia_unida))

    
    #finalmente con las frecuencias calculadas antes, calculamos el tf-idf, como la multiplicación de ambas
    # esta claro que los datos de entrenamiento sería el numero de 
    def frecuencia_tf_itf(dict1, dict2):
        result = {}
        for key in dict1:
            if key in dict2:
                result[key] = round(dict1[key] * dict2[key], 3)
        return result

    freq = pd.DataFrame({'tf':tf,'itf':itf})# creamos un dataframe para aplicar la fución a cada post
    freq['tf_itf'] = freq.apply(lambda row: frecuencia_tf_itf(row['tf'], row['itf']), axis=1)# aplicamos la función a cada fila
    
    def valoracion_de_frases(sentences, t_prueba):
        array = np.zeros(len(sentences))
        for i in range(len(sentences)):
            total = 0  
            for word in sentences[i]:
                if word in t_prueba:
                    total += t_prueba[word]
                array[i] = round(total,3)
        return array
    freq['sentences'] = original.apply(lambda x: [word_tokenize(i) for i in sent_tokenize(x)])
    freq['ponderacion'] = freq.apply(lambda row: valoracion_de_frases(row['sentences'], row['tf_itf']), axis=1)
    def seleccion_de_frases(ponder,frases):
        if len(ponder) == 1:
            return ' '.join(frases[0])
        # seleccionamos las frases que tienen una ponderación mayor al 75% de la ponderación total
        resumen = []
        ordenado = ponder
        ordenado.sort()
        threshold_index = int(len(ordenado) * 0.75)
        valores = ordenado[ordenado >= ordenado[threshold_index]]
        for i in range(len(ponder)):
            if ponder[i] in valores:
                resumen.append(' '.join(frases[i]))
                resum = ' '.join(resumen)
                return resum
    resumen = freq.apply(lambda row: seleccion_de_frases(row['ponderacion'],row['sentences']), axis=1)
    
    return resumen




#modulo 6
import pandas as pd
import numpy as np
import nltk
from collections import Counter
from scipy.spatial.distance import cosine,euclidean

def texts_distance(text1: str, text2: str):
    tokens_list = [nltk.word_tokenize(text1.lower())] # Tokenize and to lower case
    vocabulary = sorted(set(word for tokens in tokens_list for word in tokens)) # Create a vocabulary
    count_list = [Counter(tokens) for tokens in tokens_list]
    bow_matrix1 = pd.DataFrame(count_list, columns=vocabulary).fillna(0)
    
    tokens_list2 = [nltk.word_tokenize(text2.lower())] # Tokenize and to lower case
    vocabulary2 = sorted(set(word for tokens in tokens_list2 for word in tokens)) # Create a vocabulary
    count_list2 = [Counter(tokens) for tokens in tokens_list2]
    bow_matrix12 = pd.DataFrame(count_list2, columns=vocabulary2).fillna(0)
    
    concatenado = pd.concat([bow_matrix1,bow_matrix12], axis=0).fillna(0)
    
    fila1 = concatenado.iloc[0].values
    fila2 = concatenado.iloc[1].values
    
    distancia_coseno = cosine(fila1,fila2)
    
    return distancia_coseno
