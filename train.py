import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import wandb
# from wandb.keras import WandbCallback
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
from utils.auto_df_support import AutoDFSupport
from utils.BASEDIR import BASEDIR
from utils.tokenizer import split_list, tokenize
import ast
import matplotlib.gridspec as gridspec
# from keras.callbacks.tensorboard_v1 import TensorBoard

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))
# sns.distplot(data_here)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  tf.config.experimental.set_virtual_device_configuration(
      gpus[0],
      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
  logical_gpus = tf.config.experimental.list_logical_devices('GPU')


support = AutoDFSupport()
support.init()
os.chdir(BASEDIR)

# fig, ax = plt.subplots(2, 2)
# ax = ax.flatten()


def show_pred_new(epoch=0, sample_idx=None, figure_idx=0):
  if sample_idx is None:
    sample_idx = random.randrange(len(auto.x_test))
  x = auto.x_test[sample_idx]
  y = auto.y_test[sample_idx]
  pred = auto.model.predict(np.array([x]))[0]

  plt.figure(figure_idx)
  plt.clf()
  plt.bar(range(3), y, label='ground')
  plt.bar(range(3), pred, label='pred')
  plt.legend()
  plt.show()
  plt.pause(0.01)
  # plt.savefig('models/model_imgs/{}'.format(epoch))


class ShowPredictions(tf.keras.callbacks.Callback):
  def __init__(self):
    super().__init__()
    self.every = 1
    self.sample_idx1 = random.randrange(len(auto.x_test))
    self.sample_idx2 = random.randrange(len(auto.x_test))
    self.sample_idx3 = random.randrange(len(auto.x_test))
    # self.sample_idx1 = 0
    # self.sample_idx2 = 4
    # self.sample_idx3 = 8

  def on_epoch_end(self, epoch, logs=None):
    if not (epoch + self.every) % self.every:
      show_pred_new(epoch, self.sample_idx1, 0)
      show_pred_new(epoch, self.sample_idx2, 1)
      show_pred_new(epoch, self.sample_idx3, 2)


class AutoDynamicFollow:
  def __init__(self):
    self.test_size = 0.15

  def start(self):
    self.load_data()
    self.train()

  def load_data(self):
    print("Loading data...", flush=True)
    self.x_train = np.load('model_data/x_train.npy')
    self.y_train = np.load('model_data/y_train.npy')
    with open("model_data/scales", "rb") as f:
      self.scales = pickle.load(f)

    samples = 'all'
    if samples != 'all':
      self.x_train = np.array(self.x_train[:samples])
      self.y_train = np.array(self.y_train[:samples])
    self.x_train = np.array([i.flatten() for i in self.x_train])
    # self.y_train = np.array([i.flatten() for i in self.y_train])

    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train, self.y_train, test_size=self.test_size)
    print(self.x_train.shape)
    print(self.y_train.shape)

  def train(self):
    # opt = keras.optimizers.Adadelta(lr=2) #lr=.000375)
    # opt = keras.optimizers.RMSprop(lr=0.001)#, decay=1e-5)
    # opt = keras.optimizers.Adagrad(lr=0.00025)
    # opt = keras.optimizers.SGD(lr=0.1, momentum=0.3)
    # opt = keras.optimizers.Adagrad()
    # opt = 'rmsprop'
    # opt = keras.optimizers.Adadelta()
    # opt = 'adam'
    opt = keras.optimizers.Adam(amsgrad=True)

    a_function = "relu"

    self.model = Sequential()
    # model.add(Dropout(0.2))
    # model.add(GRU(128, return_sequences=True, input_shape=x_train.shape[1:]))
    # model.add(GRU(64, return_sequences=True))
    # model.add(GRU(64, return_sequences=True))
    # model.add(GRU(64, return_sequences=True))
    # model.add(GRU(y_train.shape[1], return_sequences=False))
    # model.add(Lambda(lambda x: x[:,0,:,:], output_shape=(1, 50, 1) + x_train.shape[2:]))

    denses = [128, 128, 64, 64]

    for idx, n in enumerate(denses):
      if idx == 0:
        self.model.add(Dense(n, activation=a_function, input_shape=self.x_train.shape[1:]))
      else:
        self.model.add(Dense(n, activation=a_function))
      self.model.add(BatchNormalization())

    self.model.add(Dense(self.y_train.shape[1], activation='softmax'))

    self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    name = ', '.join(['{}'.format(n) for n in denses])
    print(name)
    # wandb.init(project="auto-df", name=name)
    # w_and_b = WandbCallback()
    callbacks = [ShowPredictions()]
    self.model.fit(self.x_train, self.y_train,
                   shuffle=True,
                   batch_size=16,
                   epochs=10000,
                   validation_data=(self.x_test, self.y_test),
                   # sample_weight=np.full((len(y_train)), 100),
                   callbacks=callbacks)


auto = AutoDynamicFollow()
auto.start()

# preds = model.predict(x_test).reshape(1, -1)
# diffs = [abs(pred - ground) for pred, ground in zip(preds[0], y_test[0])]
#
# print("Test accuracy: {}%".format(round(np.interp(sum(diffs) / len(diffs), [0, 1], [1, 0]) * 100, 4)))
#
# for i in range(20):
#   c = random.randint(0, len(x_test))
#   print('Ground truth: {}'.format(support.unnorm(y_test[c][0], 'eps_torque')))
#   print('Prediction: {}'.format(support.unnorm(model.predict(np.array([x_test[c]]))[0][0], 'eps_torque')))
#   print()


def save_model(name='model'):
  auto.model.save('models/h5_models/{}.h5'.format(name))
