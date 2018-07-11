# Projeto Aprendizagem de Máquina - Airbnb

## Descrição
Projeto referente a solução do desafio 'Airbnb New User Bookings' no Kaggle. Link: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

## Instalação
- python >= 3.x
```
$ git clone https://github.com/AlexandreSGV/cin_am_airbnb
$ cd cin_am_airbnb && sudo pip3 install -rrequirements.txt
```
## Datasets
 - Devido ao tamanho dos dados do airbnb, é necessário baixar manualmente através da página do desafio: [airbnb-recruiting-new-user-bookings/data](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data) e adicioná-los na pasta `input`
 - Nesta solução utilizamos os arquivos: train_users_2.csv, test_users.csv e sessions.csv

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
