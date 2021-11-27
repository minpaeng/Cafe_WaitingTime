import pandas as pd
import numpy as np
import math

cafe = pd.read_csv('data/cafe1.csv')
wait_time = np.float16(cafe['대기시간'].values)
order_cnt = np.int16(cafe['밀린 주문 수'].values)
# print(wait_time.shape, order_cnt.shape)

final_wait_time = np.zeros(shape=(500,), dtype=float)

# 버림 적용 대기시간
floor_wait_time = np.floor(wait_time)
# print(floor_wait_time)

for i in range(500):
    final_wait_time[i] = wait_time[i]
    if order_cnt[i] != 0:
        final_wait_time[i] += np.sum([floor_wait_time[i - cnt] for cnt in range(1, order_cnt[i]+1)])

final_wait_time = 0.1*np.trunc(10*final_wait_time)
print(final_wait_time)

cafe['대기시간 (+밀린 주문 수)'] = final_wait_time

print(cafe.describe)

cafe.to_csv("cafe2.csv")









