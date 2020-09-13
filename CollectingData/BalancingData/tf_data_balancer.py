#!/usr/bin/env python
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import time
import cv2

training_data = np.load("training.npy", allow_pickle=True)
lefts = []
rights = []
forwards = []

df = pd.DataFrame(training_data)  
print(df.head())  
print(Counter(df[1].apply(str)))

shuffle(training_data)

for data in training_data:
	img = data[0]
	output = data[1]
	if output == [1,0,0]:
		lefts.append([img, output])
	elif output == [0,1,0]:
		forwards.append([img, output])
	elif output == [0,0,1]:
		rights.append([img, output])
	else: 
		pass

forwards = forwards[:len(lefts)][:len(rights)]
rights = rights[:len(forwards)]
final_data = forwards + lefts + rights

shuffle(final_data)
print(len(training_data))
print(len(final_data))
np.save('training_data_balanced_test.npy',final_data)

