import pandas as pd
import numpy as np
import pickle

sessions = False

users_path = '../output/users_clean.csv'
if(sessions == True):
	users_path = '../output/users_sessions_clean.csv'

users_df = pd.read_csv(users_path, index_col=False, low_memory=False)

test_users = pd.read_csv("../input/test_users.csv")[['id']]

model = pickle.load(open('../output/model.p','rb'))
le = pickle.load(open('../output/labelencoder.p', 'rb'))


# Inner join it with the all data frame should only keep test users
test_df = pd.merge(test_users, users_df, on='id')
test_df.set_index('id', inplace=True)
test_df.drop('country_destination', axis=1, inplace=True)
id_list = test_df.index.values

y_pred = model.predict_proba(test_df)

## Store prediction according to Kaggle format
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_list)):
    idx = id_list[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

#Generate submission
print("Outputting final results...")
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('../output/submission.csv', index=False)
print("finish...")