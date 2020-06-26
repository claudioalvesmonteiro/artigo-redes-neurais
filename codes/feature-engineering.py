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
import re

#======================================================
# criacao de variaveis com base nos dados de MATRICULA
#======================================================

# importar dados de matrciula
data = pd.read_csv('data.csv')

# criar alvo - aluno fora da idade padrao para a serie
data['alvo'] = np.where(np.isnan(data['NU_IDADE_id'].values), 1, 0)

# sexo
data['aluno_sexo_feminino'] = np.where(data['TP_SEXO'].values == 2, 1, 0)

# raca/cor preta e parda
data['aluno_cor_raca_preta_parda'] = np.where((data['TP_COR_RACA'].values == 2) | (data['TP_COR_RACA'].values == 3), 1, 0)

# nao mora na mesma cidade que estuda
data['aluno_dif_mora_estuda'] = np.where((data['CO_MUNICIPIO_END'].values != data['CO_MUNICIPIO'].values) , 1, 0)

# mora na zona rural
data['aluno_mora_zona_rural'] = np.where((data['TP_ZONA_RESIDENCIAL'].values == 2) , 1, 0)

# possui alguma decifiencia
data['aluno_deficiencia'] = np.where((data['IN_NECESSIDADE_ESPECIAL'].values ==1) |
                                    (data['IN_BAIXA_VISAO'].values ==1)         |
                                    (data['IN_CEGUEIRA'].values ==1)            |
                                    (data['IN_DEF_AUDITIVA'].values ==1)        |
                                    (data['IN_DEF_FISICA'].values ==1)          |
                                    (data['IN_DEF_INTELECTUAL'].values ==1)     | 
                                    (data['IN_SURDEZ'].values ==1)              |  
                                    (data['IN_SURDOCEGUEIRA'].values ==1)       | 
                                    (data['IN_DEF_MULTIPLA'].values ==1)        | 
                                    (data['IN_AUTISMO'].values ==1),  1, 0)

# utiliza transporte publico
data['aluno_transporte_publico'] = np.where((data['IN_TRANSPORTE_PUBLICO'].values == 1) , 1, 0)

# aula presencial
data['aluno_mediacao_didatica_presencial'] = np.where((data['TP_MEDIACAO_DIDATICO_PEDAGO'].values == 1) , 1, 0)

#======================================================
# criacao de variaveis com base nos dados de ESCOLA
#======================================================

# dependia admnistrativa da escola 'TP_DEPENDENCIA'
data['escola_estadual'] = np.where((data['TP_DEPENDENCIA'].values == 2) , 1, 0)
data['escola_municipal'] = np.where((data['TP_DEPENDENCIA'].values == 3) , 1, 0)

# localizacao da escola rural
data['escola_rural'] = np.where((data['TP_LOCALIZACAO'].values == 2) , 1, 0)

# escola possui vinculo com seguranca publica 
data['escola_vinculo_seguranca'] = np.where((data['IN_VINCULO_SEGURANCA_PUBLICA_esc'].values == 1) , 1, 0)

# escola regulamentada
data['escola_regulamentada'] = np.where((data['IN_VINCULO_SEGURANCA_PUBLICA_esc'].values == 1) , 1, 0)

# agua potavel na escola
data['escola_agua'] = np.where((data['IN_AGUA_POTAVEL_esc'].values == 1) , 1, 0)

# energia na escola
data['escola_sem_energia'] = np.where((data['IN_ENERGIA_INEXISTENTE_esc'].values == 1) , 1, 0)

# esgoto na escola
data['escola_sem_esgoto'] = np.where((data['IN_ESGOTO_INEXISTENTE_esc'].values == 1) , 1, 0)

# escola com coleta de lixo
data['escola_coleta_lixo'] = np.where((data['IN_LIXO_SERVICO_COLETA_esc'].values == 1) , 1, 0)

# escola com banheiro
data['escola_banheiro'] = np.where((data['IN_BANHEIRO_esc'].values == 1) , 1, 0)

# escola com biblioteca
data['escola_biblioteca'] = np.where((data['IN_BIBLIOTECA_SALA_LEITURA_esc'].values == 1) , 1, 0)

# escola com lab de ciencias
data['escola_lab_ciencias'] = np.where((data['IN_LABORATORIO_CIENCIAS_esc'].values == 1) , 1, 0)

# escola com lab de informatica
data['escola_lab_informatica'] = np.where((data['IN_LABORATORIO_INFORMATICA_esc'].values == 1) , 1, 0)

# escola com patio coberto
data['escola_patio_coberto'] = np.where((data['IN_PATIO_COBERTO_esc'].values == 1) , 1, 0)

# escola com quadra de esportes
data['escola_quadra'] = np.where((data['IN_QUADRA_ESPORTES_esc'].values == 1) , 1, 0)

# escola com n salas climatizadas
data['escola_salas_climatizadas'] = np.where((data['QT_SALAS_UTILIZA_CLIMATIZADAS_esc'].values == 1) , 1, 0)

# escola com internet
data['escola_internet'] = np.where((data['IN_LIXO_SERVICO_COLETA_esc'].values == 1) , 1, 0)

#======================================================
# criacao de variaveis com base nos dados de MUNICIPIO
#======================================================

# importar codigos dos municipios e populacao
code_pop = pd.read_csv('data/raw/CODE_MUNI.csv')

# violencia
obitos_agressao = pd.read_csv('data/raw/datasus_obitos_agressoes.csv', sep=';')
obitos_agressao['code_muni'] = [int(re.findall('\d+', x )[0]) for x in obitos_agressao['municipio']]

# saude
obitos_infant = pd.read_csv('data/raw/datasus_obitos_infantis.csv', sep=';')
obitos_infant['code_muni'] = [int(re.findall('\d+', x )[0]) for x in obitos_infant['municipio']]

# pib/idh
atlas = pd.read_excel('data/raw/atlas_brasil_data.xlsx')
atlas.columns =['code_muni2', 'municipio', 'porcent_trab_com_ensino_medio', 'porcent_agua_encanada', 'idhm']

# combinar dados
municipio_df = code_pop.merge(obitos_agressao, left_on='code_muni', right_on='code_muni', how='left')
municipio_df = municipio_df.merge(obitos_infant, left_on='code_muni', right_on='code_muni', how='left')
municipio_df = municipio_df.merge(atlas, left_on='code_muni2', right_on='code_muni2', how='left')

# vars
municipio_df['obitos_infant_prop_pop'] = municipio_df['obitos_infantis'] / municipio_df['pop_2012'] * 10000
municipio_df['obitos_agressao_prop_pop'] = municipio_df['obitos_agressao'] / municipio_df['pop_2012'] * 10000

# normalizacao
cols = ['log_to_capital',  'porcent_trab_com_ensino_medio', 'porcent_agua_encanada', 'idhm', 'obitos_agressao_prop_pop', 'obitos_infant_prop_pop']
for i in cols:
    municipio_df[i] = keras.utils.normalize(municipio_df[i], axis=1) 


municipio_df[i] = keras.utils.normalize(municipio_df[i], axis=1) 

# selecionar variaveis de interesse
data = data[['alvo', 
            'aluno_sexo_feminino', 
            'aluno_cor_raca_preta_parda', 
            'aluno_dif_mora_estuda', 
            'aluno_mora_zona_rural', 
            'aluno_deficiencia', 
            'aluno_transporte_publico', 
            'aluno_mediacao_didatica_presencial', 
            'escola_estadual', 
            'escola_municipal', 
            'escola_rural',
            'escola_vinculo_seguranca', 
            'escola_regulamentada', 
            'escola_agua', 
            'escola_sem_energia', 
            'escola_sem_esgoto', 
            'escola_coleta_lixo', 
            'escola_banheiro', 
            'escola_biblioteca', 
            'escola_lab_ciencias', 
            'escola_lab_informatica',
            'escola_patio_coberto', 
            'escola_quadra', 
            'escola_salas_climatizadas', 
            'escola_internet']]

# salvar dados
data.to_csv('data/preprocessed_data.csv',index=False)