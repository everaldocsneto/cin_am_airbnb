# Projeto Aprendizagem de Máquina - Airbnb

## Descrição
Projeto referente a solução do desafio 'Airbnb New User Bookings' no Kaggle. Link: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

## Instalação
- python >= 3.x
```
$ git clone https://github.com/AlexandreSGV/cin_am_airbnb
$ cd cin_am_airbnb && sudo pip3 install requirements.txt
```
## Datasets
 - O conjunto de dados é um pouco grande para ser incluído no repositório e deve se baixado diretamente do página do desafio no site do Kaggle(https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data).
 - Nesta solução utilizamos os arquivos: train_users_2.csv, test_users.csv e sessions.csv.
 - Crie uma pasta 'input' e salve estes arquivos.


## Como executar
- Limpeza do dataset de treino e teste:
```
$ python3 src/clean_dataset.py
```

- Limpeza dos dados de session:
```
$ python3 src/clean_session.py
```

- Treino:
Configure a variável session=True/False para usar ou não os dados de session
```
$ python3 src/train.py
```

- Predição:
Configure a variável session=True/False para usar ou não os dados de session
```
$ python3 src/predict.py
```

## Dependências
- pandas
- scikit-learn
- numpy
- xgboost
- pickle
