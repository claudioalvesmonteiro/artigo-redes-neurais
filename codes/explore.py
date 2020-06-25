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

# tirar amostra aleatoria
datax = data.sample(False, 0.04, 45)

datax.select('CO_MUNICIPIO', 'CO_MUNICIPIO_esc').show()

# transformar em pandas DF e salvar
datapd = datax.toPandas()
datapd.to_csv('data.csv', index = False)

