
'''
Redes Neurais Artificiais: Teoria e Aplicacao na Mineracao de Dados

Regressao Logistica
@claudio alves monteiro
github.com/claudioalvesmonteiro
junho 2020
'''

# importar pacotes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# importar dados
df = pd.read_csv('data/preprocessed_data.csv')

# dividir base em alvo e caracteristicas
alvo = df['alvo']
caracteristicas = df[df.columns[1:] ]

# dividir base em treino e teste
from sklearn.model_selection import train_test_split
caracteristicas_treino, caracteristicas_teste, alvo_treino, alvo_teste = train_test_split(caracteristicas, alvo, test_size=0.25)

#========================================
# construindo e treinando reg log
#========================================

# carregar modelo regressao logistica 
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

modelo_reg_log = LogisticRegression()

# treinamento do modelo
modelo_reg_log.fit(caracteristicas_treino, alvo_treino)

# prevendo casos desconhecidos
previsoes = modelo_reg_log.predict(caracteristicas_teste)

#========================================
# avaliando o modelo
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