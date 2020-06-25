'''
FEATURE ENGINEERING
Redes Neurais Artificiais: Teoria e Aplicacao na Mineracao de Dados

@claudio alves monteiro
github.com/claudioalvesmonteiro
junho 2020
'''

# importar pacotes
import pandas as pd
import numpy as np
import keras 

# importar dados
df = pd.read_csv('data/preprocessed/preprocessed_data.csv')

# dividir base em alvo e caracteristicas
alvo = df['alvo']
caracteristicas = df[df.columns[2:] ]

# dividir base em treino e teste
from sklearn.model_selection import train_test_split
caracteristicas_treino, caracteristicas_teste, alvo_treino, alvo_teste = train_test_split(caracteristicas, alvo, test_size=0.25)

#========================================
# rede neural artificial com keras
#========================================

# inicializa um modelo sequencial 
model = keras.models.Sequential() 
# camada de entrada, input no mesmo shape dos dados
model.add(keras.layers.core.Dense(32, input_shape=tuple([caracteristicas.shape[1]]), activation='sigmoid'))
# camada oculta
model.add(keras.layers.core.Dense(32, activation='relu'))
model.add(keras.layers.core.Dense(32, activation='relu'))
# camada de saida (decisao)
model.add(keras.layers.core.Dense(2,  activation='softmax'))
# otimizacao
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# treinamento
model.fit(caracteristicas_treino, rotulo_treino, epochs=90)

# previsao
predictions = model.predict(caracteristicas_teste)
predictions[0:5]

# decidir por '0' se a probabilidade de 0 for maior, decidir por 1 ao contrario
rotulo_pred = [0 if x[0] > x[1] else 1 for x in predictions]
rotulo_pred[0:10]

#========================================
# avaliando modelo
#========================================

# gerar matriz de confusao 
from sklearn.metrics import confusion_matrix
vp, fn, fp, vn = confusion_matrix(alvo_teste, previsoes, labels=[1,0]).reshape(-1)
total = vp+fn+fp+vn
print('Verdadeiro Positivo: ', vp, '/', round(vp/total*100, 2), '%')
print('Falso Positivo: ', fp, '/', round(fp/total*100, 2), '%')
print('Verdadeiro Negativo: ', vn, '/', round(vn/total*100, 2), '%')
print('Falso Negativo: ', fn, '/', round(fn/total*100, 2), '%')

#---- AUC-ROC

# selecionar probabilidade para classe 1
probs = modelo_reg_log.predict_proba(caracteristicas_teste)
probs = [x[1] for x in probs]

from sklearn.metrics import roc_curve,roc_auc_score
taxa_falso_positivo , taxa_verdadeiro_positivo , thresholds = roc_curve(alvo_teste , probs)

# visualizar ROC
def plot_roc_curve(taxa_falso_positivo, taxa_verdadeiro_positivo): 
    import matplotlib.pyplot as plt
    plt.plot(taxa_falso_positivo, taxa_verdadeiro_positivo) 
    plt.axis([0,1,0,1]) 
    plt.xlabel('Taxa Falso Positivo') 
    plt.ylabel('Taxa Verdadeiro Positivo') 
    plt.show()    

plot_roc_curve(taxa_falso_positivo, taxa_verdadeiro_positivo) 