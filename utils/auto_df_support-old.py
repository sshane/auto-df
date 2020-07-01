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


class AutoDFSupport:
  def __init__(self):
    self.driving_data = []
    self.scales = {}
    self.keys = ['v_ego', 'a_lead', 'v_lead', 'x_lead', 'live_tracks', 'profile', 'time']
    self.live_tracks_keys = ['v_rel', 'a_rel', 'd_rel', 'y_rel']  # all keys
    self.live_track_input_keys = ['v_rel', 'd_rel']  # only keep these keys

    self.model_inputs = ['v_ego', 'v_lead', 'a_lead', 'x_lead']
    self.model_outputs = ['profile']

    self.one_output_feature = len(self.model_outputs) == 1
    self.one_sample = True
    self.save_path = 'model_data'
    self.data_file_name = 'df_data'
    self.profile_map = {'traffic': 0, 'relaxed': 1, 'roadtrip': 2}

    self.scale_to = [0, 1]
    self._avg_time = 1 / 20.
    self.x_length = round(45 / self._avg_time)  # how long in seconds for input sample (default 35s)
    self.y_future = round(1.25 / self._avg_time)  # how far into the future we want to be predicting, in seconds (0.01 is next sample) (default 2.5s)
    self.to_skip = True
    self.skip_every = round(0.25 / self._avg_time)  # how many seconds to skip between timesteps (default 0.2s)
    self.lock = Lock()
    self.n_threads = 0
    self.split_between = 512
    self.max_threads = 512

    self.seq_len = self.x_length + self.y_future  # how many seconds should the model see at any one time

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
    for data_file in os.listdir('data/'):
      if self.data_file_name not in data_file:
        print('Skipping: {}'.format(data_file))
        continue

      print('Loading: {}'.format(data_file))
      data = self.read_file(data_file)
      keys, data = ast.literal_eval(data[0]), data[1:]
      assert all([k in keys for k in self.keys]), 'Missing keys'

      _sample1 = dict(zip(keys, ast.literal_eval(data[0])))
      profile_is_string = isinstance(_sample1['profile'], str)

      last_line = None
      for idx, line in enumerate(data):
        try:
          line = dict(zip(keys, ast.literal_eval(line)))
          if profile_is_string:
            line['profile'] = self.profile_map[line['profile']]

          # if data_file == 'df_data2' and line['v_ego'] < 60 * 0.44704:  # only keep high speed samples since low speed samples differ
          #   continue

          # if data_file == 'df_data2':  # only keep high speed samples since low speed samples differ
          #   if line['v_ego'] < 63 * 0.44704:
          #     if np.random.random() < 0.85:  # only keep roughly 15 percent of samples under 60 mph
          #       continue

          if None in [line['a_lead'], line['v_lead'], line['x_lead']]:  # skip samples without lead
            continue

          if line['profile'] == 3:  # set auto mode samples to last mode profile
            if last_line is not None:
              line['profile'] = last_line['profile']
            else:
              continue  # skip if auto and first sample

          last_line = dict(line)
          self.driving_data.append(line)
        except Exception as e:
          print(e)
          raise Exception('Error parsing line: `{}`'.format(line))

  def normalize_data(self):
    print('Normalizing data...', flush=True)
    data_t = np.array([[sample[key] for key in self.keys] for sample in self.driving_data]).T
    data_rest = []
    for idx, inp in enumerate(self.keys):
      if inp != 'live_tracks':
        to_append = data_t[idx]
        if inp not in ['time', 'profile']:
          # self.test = to_append
          self.scales[inp] = [np.amin(to_append), np.amax(to_append)]
          to_append = self.norm(to_append.astype(np.float64), inp)
        data_rest.append(to_append)

    # data_t = np.array([sample['live_tracks'] for sample in self.driving_data]).T  # this normalizes live tracks
    # data_live_tracks = []
    # for idx, inp in enumerate(self.live_tracks_keys):
    #   if inp in self.live_track_input_keys:
    #     self.scales[inp] = [np.amin(data_t[idx]), np.amax(data_t[idx])]
    #     data_live_tracks.append(self.norm(data_t[idx], inp))
    keys = [k for k in self.keys if k != 'live_tracks']
    data_rest = [dict(zip(keys, i)) for i in np.array(data_rest).T]
    times = [i['time'] for i in self.driving_data]
    assert len(data_rest) == len(times) == len(self.driving_data), 'Length of data not equal'

    self.driving_data = []
    for idx, (t, sample) in enumerate(zip(times, data_rest)):
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
        if abs(time_diff) > self._avg_time * 2:  # account for lag when writing data (otherwise we would use 0.01)
          counter += 1
          data_split.append([])
      data_split[counter].append(line)
    self.driving_data = data_split
    print('Sequences: {}'.format(len(self.driving_data)))

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

  def read_file(self, data_file):
    with open('data/{}'.format(data_file), 'r') as f:
      data = f.read().split('\n')
    return data

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
    self.x = x
    return np.interp(x, self.scales[name], self.scale_to)

  def one_hot(self, idx):
    o = [0 for _ in range(len(self.profile_map))]
    o[idx] = 1
    return o



if __name__ == '__main__':
  support = AutoDFSupport()
  support.init(process_data=True)
