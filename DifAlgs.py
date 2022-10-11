import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.optimizers import SGD
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Cargar y explorar Dataset 
Direccion = "C:/Users/yulen/OneDrive/Documentos/CIDESI/Maestria/Proyecto de tesis/Codigo/Secuencias_Breast_mutations/Datos2.xlsx"
df = pd.read_excel(Direccion)
df = df.drop(['ID'], axis=1)

#Normalizando los datos con el metodo BoxCox
caract_con_sesg = df[['Posicion cifrada A', 'Posicion cifrada C', 'Posicion cifrada G', 'Posicion cifrada T']]
pt = PowerTransformer(method='box-cox', standardize=True)
skl_boxcox=pt.fit(caract_con_sesg)
calc_lambdas = skl_boxcox.lambdas_
skl_boxcox = pt.transform(caract_con_sesg)
df_feature = pd.DataFrame(data=skl_boxcox, columns=['Posicion cifrada A', 'Posicion cifrada C', 'Posicion cifrada G', 'Posicion cifrada T'])

#Actualizacion de dataframe con datos normalizados
df_d = df.drop(['Posicion cifrada A'], axis=1)
df_d = df_d.drop(['Posicion cifrada C'], axis=1)
df_d = df_d.drop(['Posicion cifrada G'], axis=1)
df_d = df_d.drop(['Posicion cifrada T'], axis=1)
df1 = pd.concat([df_d, df_feature], axis=1)
cols = df1.columns.tolist()

def intercambiar_elemento(columna, pos1, pos2):
    columna[pos1], columna[pos2] = columna[pos2], columna[pos1]
    return columna

intercambiar_elemento(cols, 5, 9)
df1 = df1[cols]

dataset = df1.values
dataset_0 = df.values

X = dataset[:, 0:9].astype(float)
Y = dataset[:, 9]

#Transformar las clases a binarias (0, 1)
encode = LabelEncoder()
encode.fit(Y)
encoded_Y = encode.transform(Y)

X1 = dataset_0[:, 0:9].astype(float)
Y1 = dataset_0[:, 9]

#Transformar las clases a binarias (0, 1)
encode_0 = LabelEncoder()
encode_0.fit(Y1)
encoded_Y1 = encode_0.transform(Y1)

def createModel():
    model = Sequential()
    model.add(Dense(9, input_dim=9, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics='accuracy')
    return model

models = []
models.append(('LoR', LogisticRegression(solver='lbfgs', max_iter=1000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('k-NN', KNeighborsClassifier(n_neighbors=3)))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBC', GradientBoostingClassifier(n_estimators=100, learning_rate= 0.5)))
models.append(('BC-NN', KerasClassifier(build_fn=createModel, epochs=100, batch_size=5, verbose=0)))

results = []
names = []
cs_results = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X, encoded_Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean()*100.0:,.2f}% ({cv_results.std()*100.0:,.2f}%)")

    model.fit(X, encoded_Y)
    predicted = model.predict(X)
    cohen_score = cohen_kappa_score(encoded_Y, predicted)
    cs_results.append(cohen_score)
    print(f"Cohens score: {name} -> {cohen_score*100.0:,.2f}%")

    cm= confusion_matrix(encoded_Y, predicted)
    print(cm)
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sns.heatmap(df_cm, annot=True, cmap='YlGnBu', cbar=False, linewidths=3, square=True, xticklabels=['Positive','Negative'], yticklabels=['Positive','Negative'])
    plt.title(name)
    plt.show()

fig = plt.figure()
fig.suptitle("Algorithms comparation")
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()