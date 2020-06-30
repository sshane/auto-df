import ast
import os
from tqdm import tqdm
import math
import time
import pickle
import numpy as np
from threading import Thread, Lock
from utils.tokenizer import tokenize, split_list
from utils.BASEDIR import BASEDIR

os.chdir(BASEDIR)


SAVE_PATH = 'model_data'


# noinspection PyTypeChecker
class DataProcessor:
  def __init__(self):
    self.driving_data = []
    self.scales = {}
    self.all_keys = ['v_ego', 'a_ego', 'a_lead', 'v_lead', 'x_lead', 'left_lane_speeds', 'middle_lane_speeds', 'right_lane_speeds', 'left_lane_distances', 'middle_lane_distances', 'right_lane_distances', 'profile', 'time']

    self.input_keys = ['v_ego', 'left_lane_speeds', 'middle_lane_speeds', 'right_lane_speeds', 'left_lane_distances', 'middle_lane_distances', 'right_lane_distances']
    self.output_key = 'profile'

    self.speed_keys = ['left_lane_speeds', 'middle_lane_speeds', 'right_lane_speeds']
    self.distance_keys = ['left_lane_distances', 'middle_lane_distances', 'right_lane_distances']

    self.profile_map = {'traffic': 0, 'relaxed': 1, 'roadtrip': 2, 'auto': 3}

    self.data_file_name = 'df_data_lane_speed'

    self.scale_to = [0, 1]
    self._data_rate = 1 / 20.
    self.x_length = round(45 / self._data_rate)  # how long in seconds for input sample (default 35s)
    self.y_future = round(1.25 / self._data_rate)  # how far into the future we want to be predicting, in seconds (0.01 is next sample) (default 2.5s)
    self.to_skip = True
    self.skip_every = round(0.25 / self._data_rate)  # how many seconds to skip between timesteps (default 0.2s)
    self.seq_len = self.x_length + self.y_future  # how many seconds should the model see at any one time

  def start(self):
    self._setup_dirs()
    self._load_data()
    self._normalize_data()
    self._split_data()
    self._tokenize_data()
    self._format_data()
    self._dump()

  def _load_data(self):
    for data_file in os.listdir('data/'):
      if self.data_file_name not in data_file:
        print('Skipping: {}'.format(data_file))
        continue

      print('Loading: {}'.format(data_file))
      with open('data/{}'.format(data_file), 'r') as f:
        data = f.read().split('\n')

      keys, data = ast.literal_eval(data[0]), data[1:]
      assert all([k in keys for k in self.all_keys]), 'Missing keys'

      last_line = None
      for idx, line in enumerate(data):
        try:
          line = dict(zip(keys, ast.literal_eval(line)))
        except Exception as e:
          if line != '':
            print(e)
            print('Error parsing line (skipping): {}'.format(line))
          continue

        if None in [line['a_lead'], line['v_lead'], line['x_lead']]:  # skip samples without lead
          continue

        if line['profile'] == self.profile_map['auto']:  # set auto mode samples to last mode profile
          if last_line is not None:
            line['profile'] = last_line['profile']
          else:
            continue  # skip if auto and first sample

        last_line = dict(line)
        self.driving_data.append(line)

  def _normalize_data(self):
    print('\nNormalizing data...', flush=True)

    v_egos = [line['v_ego'] for line in self.driving_data]  # get scale for v_ego
    self.scales['v_ego'] = [np.min(v_egos), np.max(v_egos)]

    for idx in range(len(self.driving_data)):  # normalize v_ego
      line = self.driving_data[idx]
      line['v_ego'] = self.norm(line['v_ego'], 'v_ego')
      self.driving_data[idx] = line

    # Get scale of lane speeds and distances
    all_speeds = []
    all_distances = []
    for line in self.driving_data:
      for speed_key in self.speed_keys:
        all_speeds += line[speed_key]  # get a flat list of all speeds present in lane speeds
      for distance_key in self.distance_keys:
        all_distances += line[distance_key]  # get a flat list of all distances present in lane distances
    self.scales['lane_speeds'] = [np.min(all_speeds), np.max(all_speeds)]
    self.scales['lane_distances'] = [np.min(all_distances), np.max(all_distances)]

    # Now normalize lane speeds and distances
    for idx in range(len(self.driving_data)):
      line = self.driving_data[idx]
      for speed_key in self.speed_keys:  # norm all lane speeds
        line[speed_key] = self.norm(line[speed_key], 'lane_speeds').tolist()
      for distance_key in self.distance_keys:  # norm all lane distances
        line[distance_key] = self.norm(line[distance_key], 'lane_distances').tolist()
      self.driving_data[idx] = line

    # Remove unused keys
    for idx in range(len(self.driving_data)):
      line = self.driving_data[idx]
      line = {key: value for key, value in line.items() if key in self.input_keys + [self.output_key, 'time']}
      self.driving_data[idx] = line

    print(f'Samples: {len(self.driving_data)}')
    print(f'Scales: {self.scales}')

  def _split_data(self):
    print('\nSplitting data by time...', flush=True)
    data_split = [[]]
    for idx, line in enumerate(self.driving_data):
      if idx > 0:
        time_diff = line['time'] - self.driving_data[idx - 1]['time']
        if abs(time_diff) > self._data_rate * 2:  # account for lag when writing data (otherwise we would use 0.01)
          data_split.append([])
      data_split[-1].append(line)
    self.driving_data = data_split
    print('Concurrent sequences from splitting: {}'.format(len(self.driving_data)))

  def _tokenize_data(self):
    print('\nTokenizing data...', flush=True)
    data_sequences = []
    for idx, seq in enumerate(self.driving_data):
      # todo: experiment with splitting list instead. lot less training data, but possibly less redundant data
      data_sequences += tokenize(seq, self.seq_len)
    self.driving_data = data_sequences

    print('{} sequences of {} length'.format(len(self.driving_data), self.seq_len))

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
        time.sleep(1/100)
      Thread(target=self.format_thread, args=(section, idx)).start()

    while self.n_threads != 0:
      # with self.lock:
      print(f'\nWaiting for {self.n_threads} threads to complete...', flush=True, end='\r')
      # print(f'Percentage: {round(len(self.x_train) / len(self.driving_data) * 100, 1)}%')
      time.sleep(1)

  def format_thread(self, section, idx):
    self.n_threads += 1
    x_train = []
    y_train = []
    for idx, seq in enumerate(section):
      seq_x = seq[:self.x_length]
      seq_y = seq[self.x_length:]
      x = [[sample[des_key] for des_key in self.model_inputs] for sample in seq_x]
      y = [[self.one_hot(sample[des_key]) for des_key in self.model_outputs] for sample in seq_y]

      if self.to_skip:
        x = x[::self.skip_every]

      x_train.append(x)
      if self.one_sample:
        y_train.append(y[-1][0])
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

  def _setup_dirs(self):
    if not os.path.exists(SAVE_PATH):
      os.mkdir(SAVE_PATH)
      return True
    return False

  def unnorm(self, x, name):
    return np.interp(x, self.scale_to, self.scales[name])

  def norm(self, x, name):
    return np.interp(x, self.scales[name], self.scale_to)


if __name__ == '__main__':
  proc = DataProcessor()
  proc.start()
