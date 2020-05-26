from tensorflow.keras.models import load_model
import numpy as np
from utils.BASEDIR import BASEDIR
import os
from models.auto_df_v2 import predict

os.chdir(BASEDIR)

model = load_model('models/auto_df_v2.h5')

sample = np.random.rand(800).astype(np.float32)
ker = model.predict([[sample]]).tolist()[0]

kon = predict(sample).tolist()
print(ker)
print(kon)
