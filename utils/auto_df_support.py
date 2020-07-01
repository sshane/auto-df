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


class Track:
  def __init__(self, speed, distance):
    self.speed = speed
    self.distance = distance


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
    self.left_lane = [self.speed_keys[0], self.distance_keys[0]]
    self.middle_lane = [self.speed_keys[1], self.distance_keys[1]]
    self.right_lane = [self.speed_keys[2], self.distance_keys[2]]

    self.profile_map = {'traffic': 0, 'relaxed': 1, 'roadtrip': 2, 'auto': 3}
    self.data_file_name = 'df_data_lane_speed'

    # self.max_tracks_per_lane = 6  # this is the end of a dropoff before leveling out. most samples have 2 tracks per lane

    self.scale_to = [0, 1]
    self._data_rate = 1 / 20.
    self.x_length = round(30 / self._data_rate)  # how long in seconds for input sample (default 35s)
    self.y_future = round(1.5 / self._data_rate)  # how far into the future we want to be predicting, in seconds (0.01 is next sample) (default 2.5s)
    self.to_skip = True
    self.skip_every = round(0.4 / self._data_rate)  # how many seconds to skip between timesteps (default 0.2s)
    self.seq_len = self.x_length + self.y_future  # how many seconds should the model see at any one time

  def start(self):
    self._setup_dirs()
    self._load_data()
    self._normalize_data()
    self._flatten_lanes()
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
    print('- Normalizing data...', flush=True)

    v_egos = [line['v_ego'] for line in self.driving_data]  # get scale for v_ego
    self.scales['v_ego'] = [np.min(v_egos), np.max(v_egos)]

    for idx in range(len(self.driving_data)):  # normalize v_ego
      line = self.driving_data[idx]
      line['v_ego'] = self.norm(line['v_ego'], 'v_ego')
      self.driving_data[idx] = line

    # Get scale of lane speeds and distances
    all_speeds = []
    self.all_speeds = []
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

  def _flatten_lanes(self):
    print('- Flattening lane data...', flush=True)
    # Calculate max tracks in input data
    lane_data = []
    for idx, line in enumerate(self.driving_data):
      # 0 is left, 1 is middle, etc. just so model knows which lane track is in
      left_tracks = [[s, d, 0] for s, d in zip(*[line[l] for l in self.left_lane])]
      middle_tracks = [[s, d, 1] for s, d in zip(*[line[l] for l in self.middle_lane])]
      right_tracks = [[s, d, 2] for s, d in zip(*[line[l] for l in self.right_lane])]
      lane_data.append(left_tracks + middle_tracks + right_tracks)
    max_tracks = max([len(i) for i in lane_data])

    # Now pad
    for idx, line in enumerate(lane_data):
      builder = line
      to_pad = max_tracks - len(builder)
      builder += [[0, 0, 0] for _ in range(to_pad)]  # now pad
      self.driving_data[idx]['flat_lanes'] = np.array(builder).flatten()

  def _split_data(self):
    print('- Splitting data by time...', flush=True)
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
    print('- Tokenizing data...', flush=True)
    data_sequences = []
    for idx, seq in enumerate(self.driving_data):
      # todo: experiment with splitting list instead. lot less training data, but possibly less redundant data
      data_sequences += tokenize(seq, self.seq_len)
    self.driving_data = data_sequences

    print('{} sequences of {} length'.format(len(self.driving_data), self.seq_len))

  def _format_data(self):
    # TODO: figure out how to supply up to 16 items per lane * 2 (speed and dist) that's 16 * 6 * number of timesteps per sample. yikes!
    print('- Formatting data for model...', flush=True)
    self.x_train = []
    self.y_train = []

    for idx, seq in enumerate(self.driving_data):
      seq_x = seq[:self.x_length]
      seq_y = seq[self.x_length:]
      # v_ego = [sample['v_ego'] for sample in seq_x]
      x = [[sample['v_ego']] + sample['flat_lanes'].tolist() for sample in seq_x]
      y = self.one_hot(seq_y[-1]['profile'])

      if self.to_skip:
        x = x[::self.skip_every]

      self.x_train.append(np.array(x).flatten())
      self.y_train.append(y)

  def _dump(self):
    print('- Dumping data...', flush=True)
    self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
    np.save('model_data/x_train', self.x_train)
    np.save('model_data/y_train', self.y_train)
    with open('model_data/scales', 'wb') as f:
      pickle.dump(self.scales, f)
    print('Final data input shape: {}'.format(self.x_train[0].shape))
    print('Done!')

  def _setup_dirs(self):
    if not os.path.exists(SAVE_PATH):
      os.mkdir(SAVE_PATH)
      return True
    return False

  def unnorm(self, x, name):
    return np.interp(x, self.scale_to, self.scales[name])

  def norm(self, x, name):
    return np.interp(x, self.scales[name], self.scale_to)

  def one_hot(self, idx):
    o = [0 for _ in range(len(self.profile_map) - 1)]  # removes auto
    o[idx] = 1
    return o


if __name__ == '__main__':
  proc = DataProcessor()
  proc.start()
