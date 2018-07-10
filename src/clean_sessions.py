import pandas as pd
import numpy as np

# parametro usado para calcular quais actions serão consideradas de acordo com a sua proporção de registros no total
percentagem = 0.5
percentagem = percentagem/100

print('Abrindo ')
df = pd.read_csv('../input/sessions.csv', decimal='.')
df = df.sort_values('user_id')

unique_users_ids = df['user_id'].unique()
unique_actions = df['action'].unique()


# calcula limiar minimo
limiar = df.shape[0]*percentagem

value_counts = pd.DataFrame(df['action'].value_counts())

print('Limiar', limiar)
print(value_counts)

value_counts = value_counts.loc[value_counts['action'] > limiar]

colunas_selecionadas = value_counts.index.values

colunas_selecionadas = sorted(colunas_selecionadas)
 
colunas = ['user_id','sum_secs_elapsed'] + colunas_selecionadas
print(" - - - - - - COLUNAS - - - - - - ")
print('Numero de colunas : ',len(colunas))
print(colunas)

final_df = pd.DataFrame(columns=colunas);

print(" - - - - - - USUÁRIOS DISTINTOS - - - - - - ")

size = len(unique_users_ids)
print(size)

i = 0
for user_id in unique_users_ids:
	
	print(i,' of ',size)
	user_df = df.loc[df['user_id'] == user_id]

	row = [ user_id, user_df['secs_elapsed'].sum()] + np.full(len(colunas_selecionadas), 0).tolist()
	final_df.loc[i] = row
	actions_by_user = user_df['action'].value_counts()


	for action,value in actions_by_user.items():
		if(action not in colunas_selecionadas):
			continue
		
		final_df.loc[i, action] = value
		
	i+=1
	

final_df.to_csv('../output/sessions_clean.csv')


