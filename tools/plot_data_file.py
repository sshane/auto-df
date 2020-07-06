import os
import ast
import numpy as np
import matplotlib.pyplot as plt
from utils.BASEDIR import BASEDIR


os.chdir(BASEDIR + '/data')

data_file = 'df_data_lane_speed2-mmmkaaay'
with open(data_file, 'r') as f:
  data = f.read().split('\n')

keys, data = data[0], data[1:]
keys = ast.literal_eval(keys)

data_parsed = []
for line in data:
  try:
    p = dict(zip(keys, ast.literal_eval(line)))
    if p['v_lead'] is not None:
      data_parsed.append(p)
  except:
    pass

# p = {'traffic': 0, 'relaxed': 1, 'roadtrip': 2}

time = [i['time'] for i in data_parsed]
v_egos = [i['v_ego'] for i in data_parsed]
v_leads = [i['v_lead'] for i in data_parsed]
# profiles = [p[i['profile']] for i in data_parsed]
profiles = [i['profile'] for i in data_parsed]

plt.plot(v_egos, label='v_ego')
plt.plot(v_leads, label='v_lead')
plt.plot(profiles, label='profile')
plt.legend()
plt.show()
