import pandas as pd
# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer

# leitura dos conjuntos de dados
data_train = pd.read_csv("../input/train_users_2.csv")
data_test = pd.read_csv("../input/test_users.csv")

# merge dos conjuntos de dados
data_users = pd.concat([data_train, data_test], sort=False)

data_users.set_index('id')
data_users.reset_index(inplace=True)

# faz limpeza nos dados de idade dos usuários
def fix_age(dataset):
    # ajustando os usuários com idade > 1900
    dataset.loc[(dataset['age'] > 1900), 'age'] = 2014 - dataset['age']

    # ajustando os usuários com idade entre 0 - 15 anos e idade > que 95 anos
    dataset.loc[(dataset['age'] > 95), 'age'] = None
    dataset.loc[(dataset['age'] < 15), 'age'] = None

    return dataset

# cria um atributo categórico referente a estação do ano
def add_estacao(dataset):
    # adiciona nova coluna setando a estacao do ano de acordo com o atributo date_account_created    
    seasons = pd.DataFrame(columns=['season'])
    datas = dataset['date_account_created']       
    
    def mes_para_estacao(mes):
        if (mes <= 5) and (mes >= 3):
            return 'SPRING'
        elif (mes <= 8) and (mes >= 6):
            return 'SUMMER'
        elif (mes <= 11) and (mes >= 9):
            return 'FALL'
        else:
            return 'WINTER'

    for i in range(len(datas)):
        mes = int(datas[i].split('-')[1])
        seasons.loc[i] = [ mes_para_estacao(mes) ]
        
    dataset['season'] = seasons
    return dataset

# realiza o pré-processamento dos dados
# preenche os valores nulos com a média das idades e faz o hot-encoding dos atributos categóricos
def pre_processing_data(dataset, numeric, categorical, others):
    numeric_data = dataset[numeric]
    categorical_data = dataset[categorical]
    others_data = dataset[others]

    imputer = Imputer(strategy='median')
    imputer.fit(numeric_data)
    numeric_data = imputer.fit_transform(numeric_data)

    # convertendo para numpy to frame...
    # numeric_frame = pd.DataFrame(numeric_data, columns=['age','sum_secs_elapsed', 'ajax_refresh_subtotal', 'index', 'personalize', 'search', 'search_results', 'show'])
    numeric_frame = pd.DataFrame(numeric_data, columns=['age'])
    # hot encoding
    categorical_data = pd.get_dummies(categorical_data)

    full_dataset = pd.concat([numeric_frame, categorical_data, others_data], axis=1)
    return full_dataset

##################################

print('ajustando as idades...')
data_users = fix_age(data_users) # chama a função fix_age

print('adiciona coluna de estacao do ano...')
data_users = add_estacao(data_users) # add a coluna de estacoes do ano

numeric_attribute = ['age']
categorical_attibute = ['gender', 'signup_method', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'season']
others_attribute = ['id', 'country_destination']

print('pré-processando os dados...')
data_users = pre_processing_data(data_users, numeric_attribute, categorical_attibute, others_attribute)

# gerar csv de saída
print('gerando arquivo csv...')
data_users.to_csv('../output/users_clean.csv', index=False)
print('finish...')
