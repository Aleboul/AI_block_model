import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample

january = pd.read_csv('data/january.csv')
february = pd.read_csv('data/february.csv')
mars = pd.read_csv('data/mars.csv')
april = pd.read_csv('data/april.csv')
may = pd.read_csv('data/may.csv')
july = pd.read_csv('data/july.csv')
june = pd.read_csv('data/june.csv')
august = pd.read_csv('data/aout.csv')
september = pd.read_csv('data/september.csv')
october = pd.read_csv('data/october.csv')
november = pd.read_csv('data/novembre.csv')
december = pd.read_csv('data/december.csv')

data = pd.concat([january, february, mars, april, may, june, july, august, september, october, november, december])
data['Historical_times'] = pd.to_datetime(data['Historical_times'])
data = data.reset_index(drop = True)
print(data)
#data = data.groupby(pd.Grouper(key="Historical_times",
#                    freq="1Y")).max()
#data = data.dropna()
#
#print(data)

index = data.idxmax()

ind = index[0]

month = []
for ind in index:
    month.append(data['Historical_times'].dt.month[ind])

print(month)

unique, counts = np.unique(month, return_counts=True)

print(np.asarray((unique, counts)).T)