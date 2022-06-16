import tensorflow as tf
# numpy import
import numpy as np
# tqdm import
# tqdm은 optional이다
# tqdm은 시간이 걸리는 작업일 경우 progress bar를 표시해주는 모듈이다.
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

# random seed 설정
np.random.seed(0)

# 데이터 로드
# 해당 데이터는 다섯가지의 폰트를 가진 글자 이미지 데이터셋
# data에는 픽셀이 0와 1 값으로 가지고 있 
data = np.load('data_with_labels.npz')
train = data['arr_0']/255
labels = data['arr_1']

# 데이터를 살펴보기 위해 출력
print(train[0])
print(labels[0])

# matplotlib가 설치되었다면 figure를 필요할 때 가져올 수 있다.
# figure
import matplotlib.pyplot as plt
plt.ion()

fig = plt.figure()
plt.show()
# 
