import seaborn as sns
# from distutils.command.build import build
# from json import encoder
# from random import shuffle
# from tabnanny import verbose
# from unittest import result
# import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Cargar y explorar Dataset 
Direccion = "C:/Users/yulen/OneDrive/Documentos/CIDESI/Maestria/Proyecto de tesis/Codigo/Secuencias_Breast_mutations/Datos_para_clasificador.xlsx"
df = pd.read_excel(Direccion)
df = df.drop(['ID'], axis=1)

#Correlacion de los datos
# correlacion = df.corr()
# plt.figure(figsize = (7,7))
# ax = sns.heatmap(correlacion, vmax=1, square=True, annot=True, cmap='viridis')
# plt.show()

#Exploracion de los datos sin normalizar
#dataset = dataframe.values
#print('nombre de columnas', dataframe.columns)
#dataframe = dataframe.drop([0], axis=0)
#desc = df.describe()
# patoben = df.groupby('Significado clinico').size()
# tipoDato = list(df.dtypes)
# dataset = df.values
# print(dataset.shape)

#Mostrar histogramas datos sin normalizar
# plt.figure(1)
# plt.subplots(figsize=(20,20))
# plt.subplot(421)
# sns.histplot(df['Posicion cifrada A'])
# plt.title('Posicion de A cifrada')
# plt.grid(True)

# plt.subplot(422)
# sns.histplot(df['Posicion cifrada C'])
# plt.title('Posicion de C cifrada')
# plt.grid(True)

# plt.subplot(423)
# sns.histplot(df['Posicion cifrada G'])
# plt.title('Posicion de G cifrada')
# plt.grid(True)

# plt.subplot(424)
# sns.histplot(df['Posicion cifrada T'])
# plt.title('Posicion de T cifrada')
# plt.grid(True)

# plt.subplot(423)
# sns.histplot(df['X'])
# plt.title('X')
# plt.grid(True)

# plt.subplot(424)
# sns.histplot(df['Y'])
# plt.title('Y')
# plt.grid(True)

# plt.subplot(423)
# sns.histplot(df['Posicion de cadena de inicio'])
# plt.title('Posicion de cadena de inicio')
# plt.grid(True)

# plt.subplot(424)
# sns.histplot(df['Cromosoma'])
# plt.title('Cromosoma')
# plt.grid(True)

# plt.show()

#Normalizando los datos con el metodo BoxCox
caract_con_sesg = df[['Posicion de A cifrada', 'Posicion de C cifrada', 'Posicion de G cifrada', 'Posicion de T cifrada']]
pt = PowerTransformer(method='box-cox', standardize=True)
skl_boxcox=pt.fit(caract_con_sesg)
calc_lambdas = skl_boxcox.lambdas_
skl_boxcox = pt.transform(caract_con_sesg)
print(calc_lambdas)
df_feature = pd.DataFrame(data=skl_boxcox, columns=['Posicion cifrada A', 'Posicion cifrada C', 'Posicion cifrada G', 'Posicion cifrada T'])
#print(df_feature)
#df_feature.to_excel('Verif_data.xlsx')

#Histogramas de datos normalizados
# plt.figure(1)
# plt.subplots(figsize=(20,20))
# plt.subplot(421)
# sns.histplot(df_feature['Posicion cifrada A'])
# plt.title('Posicion de A cifrada')
# plt.grid(True)

# plt.subplot(422)
# sns.histplot(df_feature['Posicion cifrada C'])
# plt.title('Posicion de C cifrada')
# plt.grid(True)

# plt.subplot(423)
# sns.histplot(df_feature['Posicion cifrada G'])
# plt.title('Posicion de G cifrada')
# plt.grid(True)

# plt.subplot(424)
# sns.histplot(df_feature['Posicion cifrada T'])
# plt.title('Posicion de T cifrada')
# plt.grid(True)

# plt.show()

#Actualizacion de dataframe con datos normalizados
# df_d = df.drop(['Posicion de A cifrada'], axis=1)
# df_d = df_d.drop(['Posicion de C cifrada'], axis=1)
# df_d = df_d.drop(['Posicion de G cifrada'], axis=1)
# df_d = df_d.drop(['Posicion de T cifrada'], axis=1)
# df1 = pd.concat([df_d, df_feature], axis=1)
# cols = df1.columns.tolist()
# print(cols)

# def intercambiar_elemento(columna, pos1, pos2):
#     columna[pos1], columna[pos2] = columna[pos2], columna[pos1]
#     return columna

# intercambiar_elemento(cols, 5, 9)
# df1 = df1[cols]
# #print(df)

# #Exploracion de los datos normalizados
# dataset = df1.values
# dataset_0 = df.values
# # print(df1)
# # print(df)
# # print(dataset)
# # print(dataset_0)
# #print('nombre de columnas', dataframe.columns)
# #dataframe = dataframe.drop([0], axis=0)
# #desc = df.describe()
# patoben = df.groupby('Significado clinico').size()
# print(patoben)
# tipoDato = list(df.dtypes)
# dataset = df.values
#print(dataset_0.shape)

# #Generar las entradas(X) y salida(Y)
# X1 = dataset_0[:, 0:9].astype(float)
# Y1 = dataset_0[:, 9]

# #Transformar las clases a binarias (0, 1)
# encode = LabelEncoder()
# encode.fit(Y1)
# encoded_Y1 = encode.transform(Y1)
# # #print(encoded_Y)


# '''
# Generando modelo base:
# Una unica capa completamente conectada
# Funcion de activacion ReLu
# La capa de salida contiene una sola neurona para hacer predicciones utilizando funcion dde activacion Sigmoidal (para que proporcione probabilidades entre 0 y 1)
# Se usara la fucnion de perdida logaritmica binaria (binary_crossentropy)
# Algoritmo de optimizacion SGD y Acc como metrica
# '''

# def createModel():
#     model = Sequential()
#     # model.add(Dense(9, input_dim=9, activation='relu'))
#     model.add(Dense(9, input_dim=9, activation='relu'))
#     model.add(Dense(3, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     # model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
#     model.compile(loss='binary_crossentropy', optimizer=sgd, metrics='accuracy')
#     return model

# #evaluar modelo con el dataset estandarizado
# estimator = KerasClassifier(build_fn=createModel, epochs=100, batch_size=5, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results= cross_val_score(estimator, X1, encoded_Y1, cv=kfold)
# acc = results.mean()*100
# var = results.std()*100
# print(results)
# print("Linea base sin normalizar: %.2f%% (%.2f%%)" % (acc, var))


# X = dataset[:, 0:9].astype(float)
# Y = dataset[:, 9]

# #Transformar las clases a binarias (0, 1)
# encode = LabelEncoder()
# encode.fit(Y)
# encoded_Y = encode.transform(Y)
# # #print(encoded_Y)

# #evaluar modelo con el dataset estandarizado
# estimator = KerasClassifier(build_fn=createModel, epochs=100, batch_size=5, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results= cross_val_score(estimator, X, encoded_Y, cv=kfold)
# acc = results.mean()*100
# var = results.std()*100
# print(results)
# print("Linea base normalizando: %.2f%% (%.2f%%)" % (acc, var))

# # plt.plot(results)
# # plt.title('Accuracy')
# # plt.show()

