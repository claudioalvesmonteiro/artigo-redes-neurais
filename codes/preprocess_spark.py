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
    ''' importar dados de matricula, combinando regioes
    '''
    for i in range(len(lista)):
        print(lista[i])
        if i == 0:
            matricula = spark.read.csv('data/microdados_educacao_basica_2019/DADOS/MATRICULA_'+lista[i]+'.CSV', header=True, inferSchema=True, sep='|')
        else:
            mat_temp = spark.read.csv('data/microdados_educacao_basica_2019/DADOS/MATRICULA_'+lista[i]+'.CSV', header=True, inferSchema=True, sep='|')
            matricula = matricula.union(mat_temp)
    return matricula

# capturar dados de matricula
regioes =  ['NORDESTE',  'SUDESTE', 'NORTE',  'SUL', 'CO']
matricula = importCensoEscolar(regioes)

# selecionar ensino medio
matricula = matricula.filter((matricula.TP_ETAPA_ENSINO >= 25) & (matricula.TP_ETAPA_ENSINO <= 27) )

# importar dados de escola e renomear colunas
escola = spark.read.csv('data/microdados_educacao_basica_2019/DADOS/ESCOLAS.CSV', header=True, inferSchema=True, sep='|')
escola = escola.select(*(SF.col(x).alias(x + '_esc') for x in escola.columns))

# combinar matricula com escola
data = matricula.join(escola,matricula.CO_ENTIDADE == escola.CO_ENTIDADE_esc,'inner')

# combinar com idade_serie correta para  criar variavel alvo
idade_serie = spark.read.csv('artigo-redes-neurais/idade_serie.csv', header=True, inferSchema=True, sep=';')

data = data.join(idade_serie,(data.TP_ETAPA_ENSINO==idade_serie.TP_ETAPA_ENSINO_id)&(
                                 data.NU_IDADE==idade_serie.NU_IDADE_id),'left')

# tirar amostra para cada classe e combinar resultado
data_alvo1 = data.filter(data['NU_IDADE_id'].isNull()).sample(False, 0.2, 45)
data_alvo2 = data.filter(data['NU_IDADE_id'].isNotNull()).sample(False, 0.05, 45)

data_alvo1.count()
data_alvo2.count()

data_final = data_alvo1.union(data_alvo2)

# transformar em pandas DF e salvar
datapd = data_final.toPandas()
datapd.to_csv('data.csv', index = False)

