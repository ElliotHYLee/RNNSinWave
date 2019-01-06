import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, LSTM
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU


def getSeq(x,y, ts):
    input = []
    for i in range(0, x.shape[0]-ts):
        input.append(x[i:i+ts])
    input = np.array(input)
    input = np.reshape(input, (-1, ts, 1))
    y = y[ts-1:-1]
    print(input.shape)
    print(y.shape)
    return input, y

def addNoise(x, sigma):
    noise = sigma * np.random.rand(x.shape[0])
    return x + noise

def getGT(delay = 0, numPeriod = 3):
    x = np.linspace(0, np.pi*2*numPeriod, 3000)
    gt = np.sin(x+delay) + 2*np.sin(3*x+delay)
    return gt

def getModel(sh, ts):
     x_input = Input(shape=sh, name = 'x_input')
     temp = LSTM(ts, activation=LeakyReLU(), return_sequences = True)(x_input)
     temp = LSTM(ts, activation=LeakyReLU(), return_sequences = True)(temp)
     # temp = LSTM(ts, activation=LeakyReLU(), return_sequences = True)(temp)
     temp = LSTM(ts, activation=LeakyReLU(), return_sequences = False)(temp)
     temp = Dense(10, activation=LeakyReLU())(temp)
     temp = Dense(10, activation=LeakyReLU())(temp)
     out = Dense(1)(temp)
     model = Model(inputs = [x_input], outputs=[out])
     model.compile(loss='mse', optimizer='adam', loss_weights=[1])
     return model

def main():
    gt = getGT()
    x = addNoise(getGT(np.pi/3), 1)/4
    y = gt
    ts = 100
    seqX, seqY = getSeq(x,y,ts)

    m = getModel(seqX.shape[1:], ts)
    m.fit([seqX], [seqY], epochs = 30, verbose = 1, batch_size = 32)
    pred = np.zeros((3000, 1))
    seqX_test, _ = getSeq(addNoise(getGT(np.pi/3), 1)/4,y,ts)
    pred[ts-1:-1, :] = m.predict(seqX)


    plt.figure()
    plt.plot(x*4, 'black', linestyle='-.')
    plt.plot(gt, 'r')
    plt.plot(pred, 'b')
    plt.title('y = sin(x) + 2sin(3x), sequence = ' + str(ts))
    plt.legend(['Input', 'GT', 'Predicted'])
    plt.show()


if __name__ == '__main__':
    main()
