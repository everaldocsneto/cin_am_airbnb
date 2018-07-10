import pandas as pd
import numpy as np
import pickle

# users_df = pd.read_csv("users.csv", index_col=False, low_memory=False)
users_df = pd.read_csv("users_plus_session.csv", index_col=False, low_memory=False)
#pd.read_csv("users.csv", low_memory=False)

test_users = pd.read_csv("../input/test_users.csv")[['id']]

model = pickle.load(open('model.p','rb'))
le = pickle.load(open('labelencoder.p', 'rb'))


#Inner join com todos os data frames deve manter apenas os usuários de teste
test_df = pd.merge(test_users, users_df, on='id')
test_df.set_index('id', inplace=True)
test_df.drop('country_destination', axis=1, inplace=True)
id_list = test_df.index.values

y_pred = model.predict_proba(test_df)

#Armazenar a predição de acordo com o formato do Kaggle
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_list)):
    idx = id_list[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

#Gerar submissão
print("Outputting final results...")
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('submission.csv', index=False)
print("finish...")
