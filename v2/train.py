import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam, Adadelta
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import seaborn as sns
import ast
from selfdrive.config import Conversions as CV

# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
# tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

os.chdir(os.getcwd())

hyperparameter_defaults = dict(
  dense_one=8,
  dense_two=8,
  dense_three=8,

  activation='leakyrelu',

  batch_size=32,
  learning_rate=0.001
)
wandb.init(project="auto-df-v2", config=hyperparameter_defaults)
config = wandb.config

test_size = 0.25


def show_pred(sample_idxs, ax, VIS_PREDS):
  x = x_train[sample_idxs]
  y = y_train[sample_idxs]
  pred = model.predict(x)

  for idx in range(VIS_PREDS):
    ax[idx].cla()
    ax[idx].plot(range(len(prediction_time_steps)), y[idx], label='ground')
    ax[idx].plot(range(len(prediction_time_steps)), pred[idx], label='pred')
    y_scale = abs(min(y[idx]) - max(y[idx])) * .75  # + .2
    ax[idx].set_ylim(min(y[idx]) - y_scale, max(y[idx]) + y_scale)
  # ax[figure_idx].legend()

  plt.legend()
  plt.show()
  plt.pause(0.01)


class ShowPredictions(tf.keras.callbacks.Callback):
  def __init__(self):
    super().__init__()
    fig, self.ax = plt.subplots(3, 3)
    self.ax = self.ax.flatten()
    self.VIS_PREDS = len(self.ax)
    self.every = 1
    self.sample_idxs = np.random.choice(range(len(x_train)), self.VIS_PREDS)

  def on_epoch_end(self, epoch, logs=None):
    if not (epoch + self.every) % self.every:
      # for idx, sample_idx in enumerate(self.sample_idxs):
      show_pred(self.sample_idxs, self.ax, self.VIS_PREDS)


def n_grams(input_list, n):
  return list(zip(*[input_list[i:] for i in range(n)]))


print('Loading data...')
with open('df_data') as f:
  df_data = f.read()
print('Processing data...')
df_data = [ast.literal_eval(line) for line in df_data.split('\n')[:-1]]
df_keys, df_data = df_data[0], df_data[1:]
df_data = [dict(zip(df_keys, line)) for line in df_data]  # create list of dicts of samples

scale_to = [0, 1]
rate = 20
# time_in_future = int(.5 * rate)  # in seconds
prediction_time_steps = [int(t * rate) for t in [0.25, 0.5, 0.75, 1.0]]  # in seconds -> sample indexes
print('Prediction time steps: {}'.format(prediction_time_steps))

# Filter data
print('Total samples from file: {}'.format(len(df_data)))
df_data = [line for line in df_data if line['v_ego'] > CV.MPH_TO_MS * 5.]  # samples above x mph
df_data = [line for line in df_data if None not in [line['v_lead'], line['a_lead'], line['x_lead']]]  # samples with lead
for line in df_data:  # add TR key to each sample
  line['TR'] = line['x_lead'] / line['v_ego']
df_data = [line for line in df_data if line['TR'] < 5]  # TR less than x
print('Filtered samples: {}'.format(len(df_data)))

df_data_split = [[]]  # split sections where user engaged (din't gather data)
for idx, line in enumerate(df_data):
  if not idx:
    continue
  if line['time'] - df_data[idx - 1]['time'] > 1 / rate * 2:  # rate is 20hz
    df_data_split.append([])
  df_data_split[-1].append(line)

df_data_sequences = []  # now tokenize each disengaged section
for section in df_data_split:
  tokenized = n_grams(section, max(prediction_time_steps) + 1)
  if len(tokenized):  # removes sections not longer than max timestep for pred
    for section in tokenized:
      df_data_sequences.append(section)  # flattens into one list holding all sequences

TRAIN = True
if TRAIN:
  print('\nTraining on {} samples.'.format(len(df_data_sequences)))

  inputs = ['v_lead', 'a_lead', 'x_lead', 'v_ego', 'a_ego']

  # Build model data
  # x_train is all of the inputs in the first item of each sequence
  x_train = [[line[0][key] for key in inputs] for line in df_data_sequences]
  # y_train is multiple timesteps of TR in the future
  y_train = [[line[ts]['TR'] for ts in prediction_time_steps] for line in df_data_sequences]

  # x_train = [[line[key] for key in inputs] for line in df_data]  # todo: old
  # y_train = [[line['TR']] for line in df_data]  # todo: old

  x_train, y_train = np.array(x_train), np.array(y_train)
  print('x_train, y_train shape: {}, {}'.format(x_train.shape, y_train.shape))

  scales = {}
  for idx, inp in enumerate(inputs):
    _inp_data = x_train.take(indices=idx, axis=1)
    scales[inp] = np.min(_inp_data), np.max(_inp_data)

  x_train_normalized = []
  for idx, inp in enumerate(inputs):
    x_train_normalized.append(np.interp(x_train.take(indices=idx, axis=1), scales[inp], scale_to))
  x_train = np.array(x_train_normalized).T

  x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size)

  # sns.distplot(y_train.reshape(-1))

  model = Sequential()
  model.add(Dense(config.dense_one, input_shape=x_train.shape[1:], activation=LeakyReLU() if config.activation == 'leakyrelu' else config.activation))
  model.add(Dense(config.dense_two, activation=LeakyReLU() if config.activation == 'leakyrelu' else config.activation))
  model.add(Dense(config.dense_three, activation=LeakyReLU() if config.activation == 'leakyrelu' else config.activation))
  model.add(Dense(y_train.shape[1]))

  opt = Adam(lr=config.learning_rate, amsgrad=True)

  model.compile(opt, loss='mae', metrics=['mse'])

  callbacks = [WandbCallback()]
  SHOW_PRED = False
  if SHOW_PRED:
    callbacks.append(ShowPredictions())

  try:
    model.fit(x_train, y_train,
              epochs=50,
              batch_size=config.batch_size,
              validation_data=(x_test, y_test),
              callbacks=callbacks)
  except KeyboardInterrupt:
    print('Training stopped! Save model as df_model_v2.h5?')
    # affirmative = input('[Y/n]: ').lower().strip()
    # if affirmative in ['yes', 'ye', 'y']:
    #   model.save('df_model_v2.h5')
