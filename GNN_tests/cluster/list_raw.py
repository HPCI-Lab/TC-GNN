# List years of cyclones in raw/ directory and the amount of cyclones per year

import os
import numpy as np

raws = os.listdir('../data/bsc/raw/')#../../data/gnn_records/raw/')

years = []
cyclones = {}

for r in raws:
	y = r.split('_')[1]
	years.append(y)

years = np.unique(years)

for y in years:
	cyclones[y] = 0

for r in raws:
	y = r.split('_')[1]
	c = r.split('_')[3].split('.')[0]
	if cyclones[y] < int(c):
		cyclones[y] = int(c)

print(cyclones)

