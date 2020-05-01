import ast
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import time
import pickle
import numpy as np
from threading import Thread, Lock
from utils.tokenizer import tokenize, split_list
from utils.BASEDIR import BASEDIR

os.chdir(BASEDIR)


class STSupport:
  def __init__(self):
    self.driving_data = []
    self.scales = {}
    self.keys = None
    self.ignored_keys = ['angle_steers_rate', 'rate_desired', 'driver_torque', 'angle_offset']

    # self.model_inputs = ['delta_desired', 'rate_desired', 'angle_steers', 'angle_steers_rate', 'v_ego']
    self.model_inputs = ['delta_desired', 'angle_steers', 'v_ego']
    # self.model_outputs = ['eps_torque', 'rate_desired', 'angle_steers']
    self.model_outputs = ['output_steer']
    self.one_output_feature = bool(len(self.model_outputs) == 1)
    self.one_sample = False
    self.save_path = 'model_data'

    # self.needs_to_be_degrees = ['delta_desired', 'rate_desired']
    self.needs_to_be_degrees = []
    self.sR = 17.8

    self.scale_to = [0, 1]
    self._avg_time = 0.01  # openpilot runs latcontrol at 100hz, so this makes sense
    self.x_lenth = round(2 / self._avg_time)  # how long in seconds for input sample
    self.y_future = round(0.5 / self._avg_time)  # how far into the future we want to be predicting, in seconds (0.01 is next sample)
    self.lock = Lock()
    self.n_threads = 0
    self.split_between = 512
    self.max_threads = 256

    self.seq_len = self.x_lenth + self.y_future  # how many seconds should the model see at any one time

    print(f'y_future: {self.y_future}')
    print(f'seq_len: {self.seq_len}\n')

  def init(self, process_data=False):
    if process_data:
      self.start()
    else:
      self.load_scales()

  def start(self):
    self.setup_dirs()
    self.load_data()
    self.normalize_data()
    self.split_data()
    self.tokenize_data()
    self.format_data()
    self.dump()

  def load_scales(self):
    if not self.setup_dirs:
      raise Exception('Error, need to run process.py first!')
    with open('model_data/scales', 'rb') as f:
      self.scales = pickle.loads(f.read())

  def setup_dirs(self):
    if not os.path.exists(self.save_path):
      os.mkdir(self.save_path)
      return True
    return False

  def load_data(self):
    _data = []
    for data_file in os.listdir('data/'):
      if 'smart_torque_data-output-steer' in data_file:
        print('Loading: {}'.format(data_file))
        with open('data/{}'.format(data_file), 'r') as f:
          data = f.read().split('\n')

        keys, data = ast.literal_eval(data[0]), data[1:]

        for line in data:
          try:
            line = dict(zip(keys, ast.literal_eval(line)))
            line = {key: line[key] for key in line if key not in self.ignored_keys}
            # if abs(line['eps_torque']) > 900:
            #   continue
            _data.append(line)
          except:
            print('Error parsing line: `{}`'.format(line))

        keys = [key for key in keys if key not in self.ignored_keys]

        if self.keys is not None:
          if keys != self.keys:
            raise Exception('Keys in files do not match each other!')
        self.keys = keys


    for sample in _data:
      for key in sample:
        if key in self.needs_to_be_degrees:
          sample[key] = math.degrees(sample[key] * self.sR)
      self.driving_data.append(sample)

  def normalize_data(self):
    print('Normalizing data...', flush=True)
    data = np.array([[sample[key] for key in self.keys] for sample in self.driving_data])
    data_t = data.T
    data = []
    for idx, inp in enumerate(self.keys):
      if inp not in ['time']:
        self.scales[inp] = [np.amin(data_t[idx]), np.amax(data_t[idx])]
        data.append(self.norm(data_t[idx], inp))
    del data_t
    self.scales = {key: self.scales[key] for key in self.keys if key in self.model_inputs + self.model_outputs}

    data = [dict(zip(self.keys, i)) for i in np.array(data).T]
    times = [i['time'] for i in self.driving_data]
    assert len(data) == len(times) == len(self.driving_data), 'Length of data not equal'

    self.driving_data = []
    for idx, (t, sample) in enumerate(zip(times, data)):
      sample['time'] = t
      self.driving_data.append(sample)
    print(f'Samples: {len(self.driving_data)}')

  def split_data(self):
    print('Splitting data by time...', flush=True)
    data_split = [[]]
    counter = 0
    for idx, line in enumerate(self.driving_data):
      if idx > 0:
        time_diff = line['time'] - self.driving_data[idx - 1]['time']
        if abs(time_diff) > 0.05:  # account for lag when writing data (otherwise we would use 0.01)
          counter += 1
          data_split.append([])
      data_split[counter].append(line)
    self.driving_data = data_split

  def tokenize_data(self):
    print('Tokenizing data...', flush=True)
    data_sequences = []
    for idx, seq in enumerate(self.driving_data):
      # todo: experiment with splitting list instead. lot less training data, but possibly less redundant data
      data_sequences += tokenize(seq, self.seq_len)
    self.driving_data = data_sequences

  def format_data(self):
    print('Formatting data for model...', flush=True)
    self.x_train = []
    self.y_train = []

    # seq_x = map(lambda seq: seq[:-self.y_future], self.driving_data)
    # seq_y = map(lambda seq: seq[-self.y_future:], self.driving_data)
    # self.x_train = [[[sample[des_key] for des_key in self.model_inputs] for sample in seq] for seq in seq_x]
    # self.y_train = [[[sample[des_key] for des_key in self.model_outputs] for sample in seq] for seq in seq_y]

    print(self.keys)
    self.pbar = tqdm(total=len(self.driving_data))
    sections = split_list(self.driving_data, round(len(self.driving_data) / self.split_between), False)
    for idx, section in enumerate(sections):
      while self.n_threads >= self.max_threads:
        time.sleep(1/10)
      Thread(target=self.format_thread, args=(section, idx)).start()

    while self.n_threads != 0:
      # with self.lock:
      # print(f'\nWaiting for {self.n_threads} threads to complete...', flush=True, end='\r')
      #   print(f'Percentage: {round(len(self.x_train) / len(self.driving_data) * 100, 1)}%')
      time.sleep(3)


  def format_thread(self, section, idx):
    self.n_threads += 1
    x_train = []
    y_train = []
    for idx, seq in enumerate(section):
      # if idx % 25000 == 1:
      #   print(f'Percentage: {round(idx / len(section) * 100, 1)}%')
      # if abs(self.unnorm(seq[-1]['angle_steers'], 'angle_steers')) > 80:
      #   continue
      seq_x = seq[:-self.y_future]
      seq_y = seq[-self.y_future:]
      x = [[sample[des_key] for des_key in self.model_inputs] for sample in seq_x]
      y = [[sample[des_key] for des_key in self.model_outputs] for sample in seq_y]

      x_train.append(x)
      if self.one_sample:
        y_train.append(y[0])
      else:
        y_train.append(y)

    with self.lock:
      self.pbar.update(len(x_train))
      self.x_train += x_train
      self.y_train += y_train
    del x_train
    del y_train
    self.n_threads -= 1


  def dump(self):
    self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
    print('\nDumping data...', flush=True)
    np.save('model_data/x_train', self.x_train)
    np.save('model_data/y_train', self.y_train)
    with open('model_data/scales', 'wb') as f:
      pickle.dump(self.scales, f)

  def unnorm(self, x, name):
    return np.interp(x, self.scale_to, self.scales[name])

  def norm(self, x, name):
    return np.interp(x, self.scales[name], self.scale_to)


if __name__ == '__main__':
  support = STSupport()
  support.init(process_data=True)
