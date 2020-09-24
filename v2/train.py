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
from utils.conversions import Conversions as CV

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

hyperparameter_defaults = dict(
  batch_size=32,
  learning_rate=0.0006
)
wandb.init(project="auto-df-v2", config=hyperparameter_defaults)
config = wandb.config

# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
# tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

os.chdir(os.getcwd())

test_size = 0.25


def show_pred(sample_idxs, ax, VIS_PREDS):
  x = x_test[sample_idxs]
  y = y_test[sample_idxs]
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
    self.sample_idxs = np.random.choice(range(len(x_test)), self.VIS_PREDS)

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
  if '.txt' in file:
    continue
  with open('data/{}'.format(file)) as f:
    file_data = f.read()
  print('Processing {}'.format(file))
  file_data = [ast.literal_eval(line) for line in file_data.split('\n')[:-1]]
  df_keys, file_data = file_data[0], file_data[1:]
  df_data += [dict(zip(df_keys, line)) for line in file_data]  # create list of dicts of samples

print('Keys: {}'.format(df_keys))
scale_to = [0, 1]
rate = 20
scales = {}
# time_in_future = int(.5 * rate)  # in seconds
# prediction_time_steps = [int(t * rate) for t in [0, 1, 2, 3]]  # in seconds -> sample indexes
prediction_time_steps = [int(t * rate) for t in np.linspace(0, 2, 5)]  # in seconds -> sample indexes
print('Prediction time steps: {}'.format(prediction_time_steps))

# Filter data
print('Total samples from file: {}'.format(len(df_data)))
df_data = [line for line in df_data if line['v_ego'] >= CV.MPH_TO_MS * 1.]  # samples above x mph
df_data = [line for line in df_data if line['lead_status']]  # samples with lead
df_data = [line for line in df_data if not any([line['left_blinker'], line['right_blinker']])]  # samples without blinker

for line in df_data:  # add TR key to each sample
  line['TR'] = line['x_lead'] / line['v_ego']

df_data = [line for line in df_data if line['TR'] < 2.7]  # TR less than x
print('Filtered samples: {}'.format(len(df_data)))


REMOVE_LEAD_TRACKS = True  # remove lead from middle_lane speeds and dists
if REMOVE_LEAD_TRACKS:
  distance_epsilon = 2.5  # if diff of track dist and x_lead is larger than this, keep track
  for line in df_data:
    if len(line['middle_lane_speeds']) == 0 or not line['lead_status']:
      continue
    middle_lane = [(spd, dst) for spd, dst in zip(line['middle_lane_speeds'], line['middle_lane_distances']) if abs(line['x_lead'] - dst) > distance_epsilon]
    if len(middle_lane) == 0:  # map doesn't like empty lists, so fill them manually
      line['middle_lane_speeds'], line['middle_lane_distances'] = [], []
    else:
      line['middle_lane_speeds'], line['middle_lane_distances'] = map(list, zip(*middle_lane))


# For scales to normalize lane data
lane_speeds = [line['left_lane_speeds'] + line['middle_lane_speeds'] + line['right_lane_speeds'] for line in df_data]
lane_speeds = [item for sublist in lane_speeds for item in sublist]
lane_dists = [line['left_lane_distances'] + line['middle_lane_distances'] + line['right_lane_distances'] for line in df_data]
lane_dists = [item for sublist in lane_dists for item in sublist]
scales['lane_speeds'] = min(lane_speeds), max(lane_speeds)
scales['lane_distances'] = min(lane_dists), max(lane_dists)

# Find maxes for padding
scales['l_lane_max'] = max([len(line['left_lane_speeds']) for line in df_data])
scales['m_lane_max'] = max([len(line['middle_lane_speeds']) for line in df_data])
scales['r_lane_max'] = max([len(line['right_lane_speeds']) for line in df_data])
print('Scales: {}'.format(scales))

for line in df_data:
  l_distances  = np.interp(line['left_lane_distances'], scales['lane_distances'], scale_to)  # now normalize
  m_distances = np.interp(line['middle_lane_distances'], scales['lane_distances'], scale_to)
  r_distances = np.interp(line['right_lane_distances'], scales['lane_distances'], scale_to)
  l_speeds = np.interp(line['left_lane_speeds'], scales['lane_speeds'], scale_to)
  m_speeds = np.interp(line['middle_lane_speeds'], scales['lane_speeds'], scale_to)
  r_speeds = np.interp(line['right_lane_speeds'], scales['lane_speeds'], scale_to)

  left_data = [[0, 0] for _ in range(scales['l_lane_max'])]
  middle_data = [[0, 0] for _ in range(scales['m_lane_max'])]
  right_data = [[0, 0] for _ in range(scales['r_lane_max'])]

  for idx, car in enumerate(zip(l_distances, l_speeds)):
    left_data[idx] = list(car)
  for idx, car in enumerate(zip(m_distances, m_speeds)):
    middle_data[idx] = list(car)
  for idx, car in enumerate(zip(r_distances, r_speeds)):
    right_data[idx] = list(car)

  lane_data = left_data + middle_data + right_data
  line['lane_data'] = [item for sublist in lane_data for item in sublist]  # normalized and flattened


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

  model_inputs = ['v_lead', 'a_lead', 'v_ego', 'a_ego']
  model_output = 'x_lead'

  # Build model data
  # x_train is all of the inputs in the first item of each sequence
  x_train = [[seq[0][key] for key in model_inputs] for seq in df_data_sequences]
  # y_train is multiple timesteps of TR in the future
  y_train = [[seq[ts][model_output] for ts in prediction_time_steps] for seq in df_data_sequences]

  # x_train = [[line[key] for key in inputs] for line in df_data]  # todo: old
  # y_train = [[line['TR']] for line in df_data]  # todo: old

  x_train, y_train = np.array(x_train), np.array(y_train)

  for idx, inp in enumerate(model_inputs):
    _inp_data = x_train.take(indices=idx, axis=1)
    scales[inp] = np.min(_inp_data), np.max(_inp_data)
  scales[model_output] = np.amin(y_train), np.amax(y_train)

  x_train_normalized = []
  for idx, inp in enumerate(model_inputs):
    x_train_normalized.append(np.interp(x_train.take(indices=idx, axis=1), scales[inp], scale_to))
  x_train = np.array(x_train_normalized).T
  y_train = np.interp(y_train, scales[model_output], scale_to)

  ADD_LANE_SPEED_DATA = True
  if ADD_LANE_SPEED_DATA:
    x_train = x_train.tolist()
    for idx, line in enumerate(x_train):
      line += df_data_sequences[idx][0]['lane_data']
    x_train = np.array(x_train)
  print('x_train, y_train shape: {}, {}'.format(x_train.shape, y_train.shape))

  x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size)

  # sns.distplot(y_train.reshape(-1))

  model = Sequential()
  model.add(Dense(68, input_shape=x_train.shape[1:], activation=LeakyReLU()))
  # model.add(Dropout(0.07))
  model.add(Dense(72, activation=LeakyReLU()))
  model.add(Dense(128, activation=LeakyReLU()))
  # model.add(Dropout(0.1))
  model.add(Dense(164, activation=LeakyReLU()))
  # model.add(Dropout(0.15))
  model.add(Dense(y_train.shape[1]))

  opt = Adam(lr=config.learning_rate, amsgrad=True)
  # opt = Adam(lr=0.001, amsgrad=True)
  # opt = Adadelta(1)
  # opt = SGD(lr=0.5, momentum=0.9, decay=0.0001)

  model.compile(opt, loss='mse', metrics=['mae'])

  callbacks = [WandbCallback()]
  SHOW_PRED = True
  if SHOW_PRED:
    callbacks.append(ShowPredictions())

  try:
    model.fit(x_train, y_train,
              epochs=100,
              batch_size=config.batch_size,
              validation_data=(x_test, y_test),
              callbacks=callbacks)
  except KeyboardInterrupt:
    print('\nTraining stopped! Save model as df_model_v2.h5?')
    affirmative = input('[Y/n]: ').lower().strip()
    if affirmative in ['yes', 'ye', 'y']:
      model.save('models/df_model_v2_leakyrelu_relu.h5')
