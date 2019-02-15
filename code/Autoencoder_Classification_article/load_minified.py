import pickle
import numpy as np 
import pandas as pd
data = []


# Load all of the minified pickled data into a dataframe and then save that as a csv.
for i in range(0,180):
	pickleFile = open("minified_data/" + str(i)+ "minified_data.pkl", 'rb')
	output = pickle.load(pickleFile)
	for j in output:
		for k in j:
			for l in k:
				data.append(l)
propaganda = np.zeros((len(data)))
non_propaganda = np.zeros((len(data)))
with open('datasets-v5/task-1/task1.train.txt', 'r') as file:
	for idx,line in enumerate(file):
		if line.split('\t')[2].strip() == 'propaganda':
			propaganda[idx] = 1
		else:
			non_propaganda[idx] = 1

index = [str(i) for i in range(1, len(data)+1)]
df = pd.DataFrame(data, index=index)
df['propaganda'] = propaganda
df['non_propaganda'] = non_propaganda
print(df)
df.to_csv('minified_dataframe.csv')
