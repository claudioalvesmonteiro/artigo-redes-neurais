'''
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 2.4.3
      /_/

Redes Neurais Artificiais: Teoria e Aplicacao na Mineracao de Dados

@claudio alves monteiro
github.com/claudioalvesmonteiro
junho 2020
'''

# abrir Apache Spark para processamento distribuido
exec(open('/home/pacha/spark/python/pyspark/shell.py').read())

#==========================
# PRE-PROCESSAMENTO
#==========================

# importar pacotes
import pyspark.sql.functions as SF

def importCensoEscolar(lista):
    '''
    '''
    for i in range(len(lista)):
        print(lista[i])
        if i == 0:
            df = spark.read.csv('data/microdados_educacao_basica_2019/DADOS/MATRICULA_'+lista[i]+'.CSV', header=True, inferSchema=True, sep='|')
        else:
            df2 = spark.read.csv('data/microdados_educacao_basica_2019/DADOS/MATRICULA_'+lista[i]+'.CSV', header=True, inferSchema=True, sep='|')
            df = df.union(df2)
    return df

# capturar dados das regioes
regioes =  ['NORDESTE',  'SUDESTE', 'NORTE',  'SUL', 'CO']
censo19 = importCensoEscolar(regioes)

#
censo19 = spark.read.csv('data/microdados_educacao_basica_2019/DADOS/MATRICULA_NORDESTE.CSV', header=True, inferSchema=True, sep='|')


# selecionar ensino medio
censo19 = censo19.filter((censo19.TP_ETAPA_ENSINO >= 25) & (censo19.TP_ETAPA_ENSINO <= 27) )

# remover casos nulos
#censo19 = censo19.filter(censo19['TP_ETAPA_ENSINO'].isNotNull() )

# combinar com idade_serie correta para  criar variavel alvo
idade_serie = spark.read.csv('artigo-redes-neurais/idade_serie.csv', header=True, inferSchema=True, sep=';')

data = censo19.join(idade_serie,(censo19.TP_ETAPA_ENSINO==idade_serie.TP_ETAPA_ENSINO_id)&(
                                 censo19.NU_IDADE==idade_serie.NU_IDADE_id),'left')

data.count()
# tirar amostra aleatoria
datax = data.sample(False, 0.01, 45)

# transformar em pandas DF e savar
datapd = datax.toPandas()
datapd.to_csv('data.csv', index = False)

