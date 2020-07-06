import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

x_train = np.random.rand(10000, 50)
y_train = x_train.take(axis=1, indices=-1) * np.pi

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=x_train.shape[1:]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(Adam(amsgrad=True), 'mse')
model.fit(x_train, y_train, batch_size=32, validation_split=0.5, epochs=100)
