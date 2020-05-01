import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import wandb
from wandb.keras import WandbCallback
wandb.init(project="smart-torque")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM, LeakyReLU, Flatten, BatchNormalization, SimpleRNN, GRU, BatchNormalization, TimeDistributed, Lambda
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import os
import seaborn as sns
from utils.st_support import STSupport
from utils.BASEDIR import BASEDIR
from utils.tokenizer import split_list, tokenize
import ast
import matplotlib.gridspec as gridspec
# from keras.callbacks.tensorboard_v1 import TensorBoard

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))
# sns.distplot(data_here)

support = STSupport()
support.init()
os.chdir(BASEDIR)

# fig, ax = plt.subplots(2, 2)
# ax = ax.flatten()


def show_pred_new(epoch=0, sample_idx=None, figure_idx=0):
  if sample_idx is None:
    sample_idx = random.randrange(len(x_test))
  x = x_test[sample_idx]
  y = y_test[sample_idx]

  pred = model.predict(np.array([x]))[0]

  y = support.unnorm(y, 'output_steer') * 1500
  pred = support.unnorm(pred, 'output_steer') * 1500

  plt.figure(figure_idx)
  plt.clf()
  plt.plot(y, label='eps_torque ground')
  plt.plot(pred, label='eps_torque pred')
  plt.legend()
  plt.show()
  plt.pause(0.01)
  # plt.savefig('models/model_imgs/{}'.format(epoch))


class ShowPredictions(tf.keras.callbacks.Callback):
  def __init__(self):
    super().__init__()
    self.every = 1
    self.sample_idx1 = random.randrange(len(x_test))
    self.sample_idx2 = random.randrange(len(x_test))
    self.sample_idx3 = random.randrange(len(x_test))
    # self.sample_idx1 = 0
    # self.sample_idx2 = 4
    # self.sample_idx3 = 8

  def on_epoch_end(self, epoch, logs=None):
    if not (epoch + self.every) % self.every:
      show_pred_new(epoch, self.sample_idx1, 0)
      show_pred_new(epoch, self.sample_idx2, 1)
      show_pred_new(epoch, self.sample_idx3, 2)


print("Loading data...", flush=True)
x_train = np.load('model_data/x_train.npy')
y_train = np.load('model_data/y_train.npy')
with open("model_data/scales", "rb") as f:
  scales = pickle.load(f)

samples = 1000000000000000000
# x_train = np.array(x_train[:samples])
# y_train = np.array(y_train[:samples])

x_train = np.array([i.flatten() for i in x_train[:samples]])
y_train = np.array([i.flatten() for i in y_train[:samples]])
# y_train = helper.unnorm(y_train, 'eps_torque')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)
print(x_train.shape)
print(y_train.shape)

# opt = keras.optimizers.Adadelta(lr=2) #lr=.000375)
opt = keras.optimizers.RMSprop(lr=0.001)#, decay=1e-5)
# opt = keras.optimizers.Adagrad(lr=0.00025)
opt = keras.optimizers.SGD(lr=0.1, momentum=0.3)
opt = keras.optimizers.Adagrad()
# opt = 'rmsprop'
# opt = keras.optimizers.Adadelta()
opt = keras.optimizers.Ftrl(0.1)
opt = keras.optimizers.Adam(amsgrad=True)
# opt = 'adam'

recurrent = True
a_function = "relu"
dropout = 0.1

model = Sequential()
# model.add(Dropout(0.2))
# model.add(GRU(128, return_sequences=True, input_shape=x_train.shape[1:]))
# model.add(GRU(64, return_sequences=True))
# model.add(GRU(64, return_sequences=True))
# model.add(GRU(64, return_sequences=True))
# model.add(GRU(y_train.shape[1], return_sequences=False))
# model.add(Lambda(lambda x: x[:,0,:,:], output_shape=(1, 50, 1) + x_train.shape[2:]))

model.add(Dense(512, activation=a_function, input_shape=x_train.shape[1:]))
model.add(Dense(256, activation=a_function))
model.add(Dense(128, activation=a_function))
model.add(Dense(128, activation=a_function))
model.add(Dense(64, activation=a_function))


model.add(Dense(y_train.shape[1]))

model.compile(loss='mse', optimizer=opt, metrics=['mae'])
# tensorboard = TensorBoard(log_dir="C:/Git/dynamic-follow-tf-v2/train_model/logs/{}".format("final model"))
# callbacks = [ShowPredictions(), WandbCallback()]
show_predictions = ShowPredictions()
w_and_b = WandbCallback()
callbacks = [show_predictions, w_and_b]
model.fit(x_train, y_train,
          shuffle=True,
          batch_size=8,
          epochs=10000,
          validation_data=(x_test, y_test),
          # sample_weight=np.full((len(y_train)), 100),
          callbacks=callbacks)


preds = model.predict(x_test).reshape(1, -1)
diffs = [abs(pred - ground) for pred, ground in zip(preds[0], y_test[0])]

print("Test accuracy: {}%".format(round(np.interp(sum(diffs) / len(diffs), [0, 1], [1, 0]) * 100, 4)))

for i in range(20):
  c = random.randint(0, len(x_test))
  print('Ground truth: {}'.format(support.unnorm(y_test[c][0], 'eps_torque')))
  print('Prediction: {}'.format(support.unnorm(model.predict(np.array([x_test[c]]))[0][0], 'eps_torque')))
  print()


def save_model(name='model'):
  model.save('models/h5_models/{}.h5'.format(name))
