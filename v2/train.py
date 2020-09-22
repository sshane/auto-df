import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam, Adadelta, SGD
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import seaborn as sns
import ast
from v2.conversions import Conversions as CV

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

wandb.init(project="auto-df-v2")

# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
# tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

os.chdir(os.getcwd())

test_size = 0.001


def show_pred(sample_idxs, ax, VIS_PREDS):
  x = x_train[sample_idxs]
  y = y_train[sample_idxs]
  pred = model.predict(x)

  pred = np.interp(pred, scale_to, scales[model_output])
  y = np.interp(y, scale_to, scales[model_output])

  for idx in range(VIS_PREDS):
    ax[idx].cla()
    ax[idx].plot(range(len(prediction_time_steps)), y[idx], label='ground')
    ax[idx].plot(range(len(prediction_time_steps)), pred[idx], label='pred')
    if model_output == 'TR':
      y_scale = abs(min(y[idx]) - max(y[idx])) * .75 + .05
    else:
      y_scale = abs(min(y[idx]) - max(y[idx])) * .75
    ax[idx].set_ylim(min(y[idx]) - y_scale, max(y[idx]) + y_scale)


class ShowPredictions(tf.keras.callbacks.Callback):
  def __init__(self):
    super().__init__()
    fig, self.ax = plt.subplots(4, 3)
    self.ax = self.ax.flatten()
    self.VIS_PREDS = len(self.ax)
    self.every = 1
    self.sample_idxs = np.random.choice(range(len(x_train)), self.VIS_PREDS)

  def on_epoch_end(self, epoch, logs=None):
    # pass
    self.visualize(epoch)

  def visualize(self, step):
    if not (step + self.every) % self.every:
      # for idx, sample_idx in enumerate(self.sample_idxs):
      show_pred(self.sample_idxs, self.ax, self.VIS_PREDS)
      plt.legend()
      plt.show()
      plt.pause(0.01)

  def on_batch_end(self, batch, logs=None):
    # self.visualize(batch)
    pass


def n_grams(input_list, n):
  return list(zip(*[input_list[i:] for i in range(n)]))


df_data = []
print('Loading data...')
for file in os.listdir('data'):
  with open('data/{}'.format(file)) as f:
    file_data = f.read()
  print('Processing {}'.format(file))
  file_data = [ast.literal_eval(line) for line in file_data.split('\n')[:-1]]
  df_keys, file_data = file_data[0], file_data[1:]
  df_data += [dict(zip(df_keys, line)) for line in file_data]  # create list of dicts of samples

print(df_keys)
scale_to = [0, 1]
rate = 20
# time_in_future = int(.5 * rate)  # in seconds
prediction_time_steps = [int(t * rate) for t in [0, 1, 2, 3]]  # in seconds -> sample indexes
print('Prediction time steps: {}'.format(prediction_time_steps))

# Filter data
print('Total samples from file: {}'.format(len(df_data)))
df_data = [line for line in df_data if line['v_ego'] > CV.MPH_TO_MS * 5.]  # samples above x mph
df_data = [line for line in df_data if line['lead_status']]  # samples with lead
df_data = [line for line in df_data if not any([line['left_blinker'], line['right_blinker']])]  # samples without blinker

for line in df_data:  # add TR key to each sample
  line['TR'] = line['x_lead'] / line['v_ego']

df_data = [line for line in df_data if line['TR'] < 2.7]  # TR less than x
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

total_train_time = sum([len(sec) for sec in df_data_split if len(sec) >= max(prediction_time_steps) + 1]) / rate / 60
print('Training on {} minutes of data!'.format(round(total_train_time, 2)))

TRAIN = True
if TRAIN:
  print('\nTraining on {} samples.'.format(len(df_data_sequences)))

  model_inputs = ['v_lead', 'a_lead', 'x_lead', 'v_ego', 'a_ego']
  model_output = 'x_lead'

  # Build model data
  # x_train is all of the inputs in the first item of each sequence
  x_train = [[line[0][key] for key in model_inputs] for line in df_data_sequences]
  # y_train is multiple timesteps of TR in the future
  y_train = [[line[ts][model_output] for ts in prediction_time_steps] for line in df_data_sequences]

  # x_train = [[line[key] for key in inputs] for line in df_data]  # todo: old
  # y_train = [[line['TR']] for line in df_data]  # todo: old

  x_train, y_train = np.array(x_train), np.array(y_train)
  print('x_train, y_train shape: {}, {}'.format(x_train.shape, y_train.shape))

  scales = {}
  for idx, inp in enumerate(model_inputs):
    _inp_data = x_train.take(indices=idx, axis=1)
    scales[inp] = np.min(_inp_data), np.max(_inp_data)
  scales[model_output] = np.amin(y_train), np.amax(y_train)

  x_train_normalized = []
  for idx, inp in enumerate(model_inputs):
    x_train_normalized.append(np.interp(x_train.take(indices=idx, axis=1), scales[inp], scale_to))
  x_train = np.array(x_train_normalized).T
  y_train = np.interp(y_train, scales[model_output], scale_to)

  x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size)

  # sns.distplot(y_train.reshape(-1))

  model = Sequential()
  model.add(Dense(48, input_shape=x_train.shape[1:], activation='relu'))
  model.add(Dropout(0.07))
  model.add(Dense(96, activation='relu'))
  model.add(Dropout(0.1))
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.15))
  model.add(Dense(y_train.shape[1]))

  # opt = Adam(lr=config.learning_rate, amsgrad=True)
  opt = Adam(lr=0.001, amsgrad=True)
  # opt = Adadelta(1)
  # opt = SGD(lr=0.5, momentum=0.9, decay=0.0001)

  model.compile(opt, loss='mse', metrics=['mae'])

  callbacks = [WandbCallback()]
  SHOW_PRED = True
  if SHOW_PRED:
    callbacks.append(ShowPredictions())

  try:
    model.fit(x_train, y_train,
              epochs=1000000,
              batch_size=8,
              # validation_data=(x_test, y_test),
              callbacks=callbacks)
  except KeyboardInterrupt:
    print('\nTraining stopped! Save model as df_model_v2.h5?')
    affirmative = input('[Y/n]: ').lower().strip()
    if affirmative in ['yes', 'ye', 'y']:
      model.save('models/df_model_v2.h5')
