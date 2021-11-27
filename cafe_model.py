import pandas as pd
import nltk
import numpy as np
import tensorflow.keras as keras
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt


# 퀴즈 1
# stock_daily.csv 파일로부터 x, y를 반환하는 함수를 만드세요
# batch_size, seq_length, n_features = 32, 7, 5

# 퀴즈 2
# 80%의 데이터로 학습하고 20%의 데이터에 대해 결과를 예측하세요
def get_xy():
    cafe = pd.read_csv('data/cafe2.csv')
    cafe = cafe.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]]
    print(cafe)

    scaler = preprocessing.MinMaxScaler()  # 최소, 최대 범위를 0~1로
    values = scaler.fit_transform(cafe.values)

    grams = list(nltk.ngrams(values, 2))
    grams = np.float32(grams)
    # print(grams.shape)

    x = np.float32([g[:-1] for g in grams])
    y = np.float32([g[-1, -1:] for g in grams])
    # print(x.shape, y.shape)  # (725, 7, 5) (725, 1)

    print(x.shape, y.shape)
    return x, y, scaler.data_min_[-1], scaler.data_max_[-1]


# 퀴즈
# 앞에서 만든 데이터에 대해 모델을 구축하세요
def modef_stock():
    x, y, data_min, data_max = get_xy()  # 최대 최소값을 이용, 계산해 원래의 값으로 복구시킨다

    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x_train.shape[1:]))
    model.add(keras.layers.LSTM(32, return_sequences=True, activation="ReLU"))
    model.add(keras.layers.LSTM(16, return_sequences=True, activation="ReLU"))
    model.add(keras.layers.LSTM(8, return_sequences=False, activation="ReLU"))
    model.add(keras.layers.Dense(1))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss=keras.losses.mse,
                  metrics='mse')

    model.fit(x_train, y_train, epochs=100, verbose=2, batch_size=32)
    model.evaluate(x_test, y_test, verbose=2)

    p = model.predict(x_train)
    print(p.shape, x_train.shape)
    print((data_max - data_min) * p + data_min)

    plt.subplot(1, 2, 1)
    plt.plot(y_train, 'r', label='target')  # 데이터를 섞어서 시각화가 제대로 되지 않음 # 셔플옵션  false로 주고오기
    plt.plot(p, 'g', label='prediction')
    plt.legend()  # label 값을 표에 표시할수있다

    p = (data_max - data_min) * p + data_min
    # print((data_max-data_min)*p+data_min)
    y_train = (data_max - data_min) * y_train + data_min

    plt.subplot(1, 2, 2)
    plt.plot(y_train, 'r')  # 데이터를 섞어서 시각화가 제대로 되지 않음 # 셔플옵션  false로 주고오기
    plt.plot(p, 'g')
    # plt.ylim(2650, 3000)
    plt.show()


modef_stock()
