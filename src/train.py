import pandas as pd

from sklearn.preprocessing import LabelEncoder

# usar ou não dados de session, estes dados podem ser gerados com o script clean_sessions.py
sessions = False

dataset = pd.read_csv("../output/users_clean.csv", low_memory=False)

if(sessions == True):
	data_sessions = pd.read_csv("../output/sessions_clean.csv")
	data_sessions.set_index('id')
	
	dataset = pd.concat([dataset.set_index('id'), data_sessions.set_index('id')], axis=1, sort=False)
	dataset.index.name = 'id'

	dataset.to_csv('users_sessions_clean.csv')

print('removendo os dados de teste...')
dataset = dataset.dropna() # remove linhas que tem algum valor nulo; como para os dados de teste o atributo 'country_destination' estará vazio, os dados de testes serão removidos
print('Shape dataset: ', dataset.shape)

if(sessions == False):
	dataset.set_index('id', inplace=True) # define id como index do frame
id_train = dataset.index.values

print('separando dados e rótulos para treinamento...')
labels = dataset['country_destination']
le = LabelEncoder()
y = le.fit_transform(labels)
X = dataset.drop('country_destination', axis=1)

########################################

import xgboost as xgb
from sklearn import model_selection
import pickle

# Treinando o classificador
XGB_model = xgb.XGBClassifier(objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0) #definição do modelo
param_grid = {'max_depth': [3, 4, 5], 'learning_rate': [0.1, 0.2, 0.3 ], 'n_estimators': [30,40,50]} #grid de parâmetros
model = model_selection.GridSearchCV(estimator=XGB_model, param_grid=param_grid, scoring='accuracy', verbose=10, n_jobs=4, iid=True, refit=True, cv=3)

model.fit(X, y)
print("melhor score: %0.3f" % model.best_score_)
print("melhores parâmetros:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
     print("\t%s: %r" % (param_name, best_parameters[param_name]))

# armazenar o modelo e o codificador do rótulo em um pickle
pickle.dump(model, open('../output/model.p', 'wb'))
pickle.dump(le, open('../output/labelencoder.p', 'wb'))
