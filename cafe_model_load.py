import pandas as pd
import numpy as np
import tensorflow.keras as keras
from sklearn import preprocessing


def get_xy():
    cafe = pd.read_csv('data/cafe2.csv')
    cafe = cafe.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]]
    # print(cafe)

    scaler = preprocessing.MinMaxScaler()  # 최소, 최대 범위를 0~1로
    values = scaler.fit_transform(cafe.values)
    print(values.shape)  # (500, 12)

    x = np.float32(values[:, :-1])
    y = np.float32(values[:, -1])
    print(x.shape, y.shape)  # (500, 11) (500,)

    return scaler, x, y, scaler.data_min_[-1], scaler.data_max_[-1]


model = keras.models.load_model("model/cafe_multiple_regression_60_0.01.h5")
scaler, _, _, data_min, data_max = get_xy()
values = scaler.transform([[1, 0, 4, 4, 4, 0, 1, 0, 14, 2, 1, 0]])
p = model.predict(values[:, :-1])

print((data_max - data_min) * p + data_min)
