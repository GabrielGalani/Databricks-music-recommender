# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # Transformando os dados

# COMMAND ----------

!pip install scikit-image

# COMMAND ----------

pip install --upgrade numpy

# COMMAND ----------

!pip install spotipy

# COMMAND ----------

import pyspark.pandas as ps
import numpy as np
from scipy.spatial.distance import euclidean
import plotly.express as px
from pyspark.sql.types import FloatType
import pyspark.sql.functions as f
from pyspark.sql.functions import rand
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import PCA 
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import matplotlib.pyplot as plt
from skimage import io

# COMMAND ----------

display(dbutils.fs.ls('dbfs:/FileStore/tables/Spotify/dados_tratados/'))

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Carregando e limpando dados nulos

# COMMAND ----------

dataframe = ps.read_parquet('dbfs:/FileStore/tables/Spotify/dados_tratados/data.parquet/')
dataframe = dataframe.dropna()
dataframe.info()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Criando coluna de artista e som para poder encontrar os sons recomendados

# COMMAND ----------

dataframe['artists_song'] = dataframe.artists + ' - ' + dataframe.name
dataframe.head()


# COMMAND ----------

dataframe.info()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Removendo colunas strings para passar as numericas no ML

# COMMAND ----------

X = dataframe.columns.to_list()
X.remove('artists')
X.remove('id')
X.remove('name')
X.remove('release_date')
X.remove('artists_song')

X

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Voltando o dataframe para spark

# COMMAND ----------

dataframe = dataframe.to_spark()
dataframe

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Vetorizando

# COMMAND ----------

dataframe_encoded_vector = VectorAssembler(inputCols = X, 
                                           outputCol = 'features').transform(dataframe)


# COMMAND ----------

dataframe_encoded_vector.select('features').show(truncate=False, n=10)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Padronizando / Normalizando os dados

# COMMAND ----------

scaler = StandardScaler(inputCol='features', outputCol='features_scaled')
model_scaler = scaler.fit(dataframe_encoded_vector)
dataframe_songs_scaler = model_scaler.transform(dataframe_encoded_vector)

# COMMAND ----------

dataframe_songs_scaler.select('features_scaled').show(truncate=False, n=10)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Removendo a quantidade de features (colunas) | PCA - Principal Component Analysis

# COMMAND ----------

k = len(X)
k

# COMMAND ----------

pca = PCA(k=k, inputCol='features_scaled', outputCol='pca_features')
model_pca = pca.fit(dataframe_songs_scaler)
dataframe_songs_pca = model_pca.transform(dataframe_songs_scaler)

# COMMAND ----------

dataframe_songs_pca.select('pca_features').show(truncate=False, n=10)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Verificando com o método explainedVariance o quanto que cada feature (colunas) explica acerca do nosso Dataframe separadamente e depois multiplicando com 100 para transfromar em porcentagem

# COMMAND ----------

model_pca.explainedVariance

# COMMAND ----------

sum(model_pca.explainedVariance)*100

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Diminuindo a quantidade de features de forma a explicar pelo menos 70% do dataframe com a menor quantidade de features (colunas possíveis)
# MAGIC <br> 
# MAGIC
# MAGIC Metodologia: <br>
# MAGIC Somamos cada item da nossa lista que explica a nosso dataframe até que alcance 100%, depois vamos filtrar apenas as features que explicam 70% (0.7) e refazer o processo de PCA

# COMMAND ----------

list_values = [sum(model_pca.explainedVariance[0:i+1]) for i in range(k)]
list_values

# COMMAND ----------

K = sum(np.array(list_values) <=0.7)
K

# COMMAND ----------

pca = PCA(k=K, inputCol='features_scaled', outputCol='pca_features')
model_pca = pca.fit(dataframe_songs_scaler)
dataframe_songs_pca = model_pca.transform(dataframe_songs_scaler)

# COMMAND ----------

dataframe_songs_pca.select('pca_features').show(truncate=False, n=10)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Criando Pipeline de transformação

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Criando o pipeline

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Usando o método Pipeline para encapsular todo tratamento em apenas uma varável que tornará possível passar todo tratamento apenas por um local e de uma vez

# COMMAND ----------

pca_pipeline =Pipeline(stages=[
    VectorAssembler(inputCols=X, outputCol='features'),
    StandardScaler(inputCol='features', outputCol='features_scaled'),
    PCA(k=6, inputCol='features_scaled', outputCol='pca_features')
]) 

# COMMAND ----------

 model_pca_pipeline = pca_pipeline.fit(dataframe)
 projection = model_pca_pipeline.transform(dataframe)

# COMMAND ----------

projection.select('pca_features').show(truncate=False, n=10)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Modelo de Machine Learning com K-Means

# COMMAND ----------

SEED = 1224

# COMMAND ----------

kmeans = KMeans(k=50,
                featuresCol='pca_features',
                predictionCol='cluster_pca',
                seed=SEED)
model_kmeans = kmeans.fit(projection)
projection_kmeans= model_kmeans.transform(projection)

projection_kmeans.select(['pca_features', 'cluster_pca']).show(truncate=False, n=10)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Analisando de forma visual os cluster

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Desvetorizando os dados

# COMMAND ----------

projection_means = projection_kmeans.withColumn('x', vector_to_array('pca_features')[0])\
                                    .withColumn('y', vector_to_array('pca_features')[1])

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Graficos

# COMMAND ----------

projection_means.select(['x', 'y', 'cluster_pca', 'artists_song']).show(truncate=False, n=10)

# COMMAND ----------

fig = px.scatter(projection_means.toPandas(), x='x', y='y', color='cluster_pca', hover_data=['artists_song'])
fig.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Sistema de recomendação

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Criando variável para encontrar musicas recomandadas em um cluster

# COMMAND ----------

nome_musica = 'Taylor Swift - Blank Space'

# COMMAND ----------

cluster = projection_means.filter(projection_means.artists_song==nome_musica).select('cluster_pca').collect()[0][0]
cluster

# COMMAND ----------

musicas_recomendadas = projection_means.filter(projection_means.cluster_pca == cluster)\
                                       .select('artists_song', 'id', 'pca_features')
musicas_recomendadas.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Otimizando os resultados
# MAGIC

# COMMAND ----------

componenetes_musica = musicas_recomendadas.filter(musicas_recomendadas.artists_song == nome_musica)\
                                          .select('pca_features').collect()[0][0]
componenetes_musica                             

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Distancia Euclidiana

# COMMAND ----------

def calcula_distancia(value):
    return euclidean(componenetes_musica, value)

udf_calcula_distancia = f.udf(calcula_distancia, FloatType())

musicas_recomendadas_dist = musicas_recomendadas.withColumn('Dist', udf_calcula_distancia('pca_features'))

# COMMAND ----------

recomendados = spark.createDataFrame(musicas_recomendadas_dist.sort('Dist').take(10)).select(['artists_song', 'id', 'Dist'])

recomendados.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Esse é o dataframe de recomendação por caracteristicas entre musicas e, vale ressaltar que a primeira ocorrencia é a própria música procurada, visto que a distancia de ela para ela mesma é 0 e a segunda, também é ela, mas com outro ID, o que indica que essa música também foi lancaçada em outro algum e em outro momento pela mesma artista.

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Criando a função final de recomendação

# COMMAND ----------

def recomendador(nome_musica): 
    cluster = projection_means.filter(projection_means.artists_song==nome_musica).select('cluster_pca').collect()[0][0]
    musicas_recomendadas = projection_means.filter(projection_means.cluster_pca == cluster)\
                                       .select('artists_song', 'id', 'pca_features')
    componenetes_musica = musicas_recomendadas.filter(musicas_recomendadas.artists_song == nome_musica)\
                                          .select('pca_features').collect()[0][0]
    
    def calcula_distancia(value):
        return euclidean(componenetes_musica, value)
    udf_calcula_distancia = f.udf(calcula_distancia, FloatType())
    musicas_recomendadas_dist = musicas_recomendadas.withColumn('Dist', udf_calcula_distancia('pca_features'))

    recomendados = spark.createDataFrame(musicas_recomendadas_dist.sort('Dist').take(10)).select(['artists_song', 'id', 'Dist'])
    return recomendados    

# COMMAND ----------

nome_musica = 'Taylor Swift - Blank Space'
df_recomedada = recomendador(nome_musica=nome_musica)
df_recomedada.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Criando API de apresentação e disponibilização

# COMMAND ----------

scope = "user-library-read playlist-modify-private"

OAuth = SpotifyOAuth(
        scope=scope,         
        redirect_uri='http://localhost:5000/callback',
        client_id = 'CLIENT_ID',
        client_secret = 'CLIENT_SECRET')

# COMMAND ----------

client_credentials_manager = SpotifyClientCredentials(client_id = 'CLIENT_ID',
                                                      client_secret = 'CLIENT_SECRET')

sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

# COMMAND ----------

id = projection_means.filter(projection_means.artists_song == nome_musica).select('id').collect()[0][0]
id

# COMMAND ----------

sp.track(id)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Coletando os atribuitos da musica

# COMMAND ----------

df_recomedada.select('id')

# COMMAND ----------

playlist_id = df_recomedada.select('id').collect()

# COMMAND ----------

playlist_track = []
for id in playlist_id:
    playlist_track.append(sp.track(id[0]))

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Imagem dos albuns

# COMMAND ----------

nome_musica = 'Taylor Swift - Blank Space'

id = projection_means\
          .filter(projection_means.artists_song == nome_musica)\
          .select('id').collect()[0][0]

track = sp.track(id)

url = track["album"]["images"][1]["url"]
name = track["name"]

image = io.imread(url)
plt.imshow(image)
plt.xlabel(name, fontsize = 10)
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Função para percorrer todas as imagens da recomendação

# COMMAND ----------

def visualize_songs(name,url):

    plt.figure(figsize=(15,10))
    columns = 5
    for i, u in enumerate(url):
        ax = plt.subplot(len(url) // columns + 1, columns, i + 1)
        image = io.imread(u)
        plt.imshow(image)
        ax.get_yaxis().set_visible(False)
        plt.xticks(color = 'w', fontsize = 0.1)
        plt.yticks(color = 'w', fontsize = 0.1)
        plt.xlabel(name[i], fontsize = 10)
        plt.tight_layout(h_pad=0.7, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.grid(visible=None)
    plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Testando a função

# COMMAND ----------

playlist_id = recomendados.select('id').collect()

name = []
url = []
for i in playlist_id:
    track = sp.track(i[0])
    url.append(track["album"]["images"][1]["url"])
    name.append(track["name"])

# COMMAND ----------

visualize_songs(name,url)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Atualizando a função de recomendador

# COMMAND ----------

def recomendador(nome_musica):
    #Calcula musicas recomendadas 
    cluster = projection_means.filter(projection_means.artists_song==nome_musica).select('cluster_pca').collect()[0][0]
    musicas_recomendadas = projection_means.filter(projection_means.cluster_pca == cluster)\
                                       .select('artists_song', 'id', 'pca_features')
    componenetes_musica = musicas_recomendadas.filter(musicas_recomendadas.artists_song == nome_musica)\
                                          .select('pca_features').collect()[0][0]
    
    def calcula_distancia(value):
        return euclidean(componenetes_musica, value)
    udf_calcula_distancia = f.udf(calcula_distancia, FloatType())
    musicas_recomendadas_dist = musicas_recomendadas.withColumn('Dist', udf_calcula_distancia('pca_features'))

    recomendados = spark.createDataFrame(musicas_recomendadas_dist.sort('Dist').take(10)).select(['artists_song', 'id', 'Dist'])

    #Pegar informações da API
    id = projection_means.filter(projection_means.artists_song == nome_musica).select('id').collect()[0][0]
    playlist_id = recomendados.select('id').collect()
    playlist_track = []
    for id in playlist_id:
        playlist_track.append(sp.track(id[0]))


    #Plotando capas 
    plt.figure(figsize=(15,10))
    columns = 5
    for i, u in enumerate(url):
        ax = plt.subplot(len(url) // columns + 1, columns, i + 1)
        image = io.imread(u)
        plt.imshow(image)
        ax.get_yaxis().set_visible(False)
        plt.xticks(color = 'w', fontsize = 0.1)
        plt.yticks(color = 'w', fontsize = 0.1)
        plt.xlabel(name[i], fontsize = 10)
        plt.tight_layout(h_pad=0.7, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.grid(visible=None)
    plt.show()
    

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Resultado final

# COMMAND ----------

nome_musica = 'Taylor Swift - Blank Space'
recomendador(nome_musica)

# COMMAND ----------


