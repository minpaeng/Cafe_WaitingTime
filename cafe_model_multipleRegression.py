import pandas as pd
import numpy as np
import tensorflow.keras as keras
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt


def get_xy():
    cafe = pd.read_csv('data/cafe2.csv')
    cafe = cafe.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]]
    # print(cafe)

    scaler = preprocessing.MinMaxScaler()  # 최소, 최대 범위를 0~1로
    values = scaler.fit_transform(cafe.values)
    print(values.shape)     # (500, 12)

    x = np.float32(values[:, :-1])
    y = np.float32(values[:, -1])
    print(x.shape, y.shape)  # (500, 11) (500,)

    return x, y, scaler.data_min_[-1], scaler.data_max_[-1]


def model_cafe():
    x, y, data_min, data_max = get_xy()  # 최대 최소값을 이용, 계산해 원래의 값으로 복구시킴

    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x_train.shape[1:]))
    model.add(keras.layers.Dense(16, activation="ReLU"))
    model.add(keras.layers.Dense(32, activation="ReLU"))
    model.add(keras.layers.Dense(16, activation="ReLU"))
    model.add(keras.layers.Dense(8, activation="ReLU"))
    model.add(keras.layers.Dense(1))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss=keras.losses.mse,
                  metrics='mse')

    checkpoint = keras.callbacks.ModelCheckpoint('model/cafe_multiple_regression_{epoch:02d}_{val_loss:.2f}.h5',
                                                 save_best_only=True)

    model.fit(x_train, y_train, epochs=500, verbose=2, batch_size=32,
              validation_data=(x_test, y_test),
              callbacks=[checkpoint])

    p = model.predict(x_test)
    print((data_max - data_min) * p + data_min)

    plt.subplot(1, 2, 1)
    plt.plot(y_test, 'r', label='target')
    plt.plot(p, 'g', label='prediction')
    plt.legend()  # label 값을 표에 표시

    p = (data_max - data_min) * p + data_min

    y_test = (data_max - data_min) * y_test + data_min

    plt.subplot(1, 2, 2)
    plt.plot(y_test, 'r')
    plt.plot(p, 'g')
    plt.show()


model_cafe()
