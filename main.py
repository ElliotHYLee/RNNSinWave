import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python import SKCompat
from sklearn.metrics import mean_squared_error

from DataGen import genData
from Model import lstm_model

LOG_DIR = 'logs/'
TIMESTEPS = 50
# RNN_LAYERS = [{'TimeSteps': TIMESTEPS, 'keep_prob': 1},
#               {'TimeSteps': TIMESTEPS, 'keep_prob': 1}]
RNN_LAYERS = [{'TimeSteps': TIMESTEPS, 'keep_prob': 1}]


exp = 3
fcDim = [10**exp, 10**exp, 10**exp]#, 10**exp, 10**exp]
TRAINING_STEPS = 10**3
PRINT_STEPS = TRAINING_STEPS / 10
BATCH_SIZE = 100


x, y = genData(np.sin, np.linspace(0, 10**2, 10**4, dtype=np.float32), TIMESTEPS)

# create a lstm instance and validation monitor
validation_monitor = learn.monitors.ValidationMonitor(x['val'], y['val'],)

# Get the RNN model
model = SKCompat(learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, fcDim)))

# TRAIN START HERE ===========================================
model.fit(x['train'], y['train'], monitors=[validation_monitor], batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

# TEST START HERE ===========================================
predicted = np.asmatrix(model.predict(x['test']), dtype = np.float32)

# Analyse Test Result
rmse = np.sqrt((np.asarray((np.subtract(predicted, y['test']))) ** 2).mean())
score = mean_squared_error(predicted, y['test'])
nmse = score / np.var(y['test']) # should be variance of original data and not data from fitted model, worth to double check

print("RSME: %f" % rmse)
print("NSME: %f" % nmse)
print("MSE: %f" % score)


# Plot the result
plot_test1, = plt.plot(y['test'][:,0], label='y0_actual')
plot_predicted1, = plt.plot(predicted[:,0], label='y0_predicted')
plot_test2, = plt.plot(y['test'][:,1], label='y1_actual')
plot_predicted2, = plt.plot(predicted[:,1], label='y1_predicted')

plt.legend(handles=[plot_predicted1, plot_test1, plot_predicted2, plot_test2])
plt.grid(True)
plt.ylim((-200,200))
plt.show()
