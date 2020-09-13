import numpy as np
from alexnet import alexnet

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'self-driving-car-BALANCED-{}-{}-{}.model'.format(LR, 'alexnet', EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load('training_data_balanced.npy', allow_pickle=True)
test_data = np.load('test_data_balanced.npy', allow_pickle=True)

train = train_data[:]
test = test_data[:] #will be changed

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = np.array([i[1] for i in test])

model.fit({'input':X}, {'targets': Y}, batch_size = 32, n_epoch = EPOCHS,
          validation_set = ({'input':test_x}, {'targets': test_y}),
          snapshot_step = 10, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)

# tensorboard --logdir=/home/turan/Desktop/version2/log
