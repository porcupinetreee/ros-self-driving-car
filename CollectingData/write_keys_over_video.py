#!/usr/bin/env python
import numpy as np
import os
import cv2
from getch import getch
'''
This script collects training data contains current image data versus current key press data. This will be run in the Pi.
'''

# size of the frame
W = 80
H = 60

cap = cv2.VideoCapture(0)


def key_in():
        key = getch()
        return key
       


def callback(keys):
    if keys == 'a':
        output = [1, 0, 0]
    elif keys == 'w':
        output = [0, 1, 0]
    elif keys == 'd':
        output = [0, 0, 1]
    else:
        output = [0, 0, 0]
    training_data.append([frame, output])
    print(output)
    if len(training_data) % 1 == 0:
	np.save(file_name, training_data)
	print('*****SAVED*****')
    return None
    

    
    
file_name = 'training.npy'

if os.path.isfile(file_name):
    print('File exist, loading previous...')
    training_data = list(np.load(file_name, allow_pickle=True))
else:
    print('File does not exist, starting fresh...')
    training_data = []
    
if __name__ == '__main__':
    
    while(cap.isOpened()):
        ret, frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.resize(frame,(W,H))
    	cv2.imshow('frame',frame)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break
	keys = key_in()
    	callback(keys)
	
cap.release()
cv2.destroyAllWindows()
    	





