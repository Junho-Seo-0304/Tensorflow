# TensorFlow import
import tensorflow as tf

# 상수 선언
a = tf.constant(1)
b = tf.constant(2)

c = a + b
d = a * b

print(a, b, c, d)

# TensorFlow에서 숫자를 tensor라는 것에 저장한다.
# .을 붙이면서 float형 선언 가능
V1 = tf.constant([1., 2.])              # Vector, 1-dimensinal
V2 = tf.constant([3., 4.])              # Vector, 1-dimensinal
M = tf.constant([[1.,2.]])              # Matrix, 2d
N = tf.constant([[1.,2.],[3.,4.]])      # Matrix, 2d
K = tf.constant([[[1.,2.],[3.,4.]]])    # Tensor, 3d+

# tensor의 값들도 연산이 가능
V3 = V1 + V2
M2 = M * M
# tensor의 값을 곱하기 위해선 아래와 같이 사용
NN = tf.matmul(N,N)

print(V3)
print(NN)

# TensorFlow 2.x 버젼으로 오면서 Session을 선언하고 run하는 과정이 생략
# sess = tf.Session()
# output = sess.run(NN)
# print(output)
# 아래와 같이 print하면 Session, run 과 같은 역할을 할 수 있다.
tf.print(NN)

# Session을 이용했다면 Session을 닫아주어야한다.
# sess.close()

# TensorFlow에서 Session은 일종의 실행창이다
# Tensor의 내용물과 연산 결과를 확인하고 싶을 때 Session을 이용한다.
# Session은Session()과 InteractiveSession()로 크게 두가지가 존재
# InteractiveSession()은 자동으로 터미널에 default Session을 할당하지만 Session()은 그렇지 않기 때문에 with절과 사용해야한다.

# Session 선언
# sess = tf.Session()
# sess = tf.InteractiveSession()

# Session 실행
# sess.run(tensor 혹은 연산)

# Session 종료
# sess.close()
# 단, with 절에서는 불필요

# TensorFlow 1.x 버젼일 경우 eval()을 이용해 값을 array로 가져 올 수 있다.
# print(M2.eval())
# Session()으로 세션을 선언했다면
# with tf.Session() as sess:
#     sess.run(M2)
#     print(M2.eval())
# 와 같이 with 절로 사용하지 않는다면 default session이 없기 때문에 오류가 발생
# InteractiveSession()으로 세션을 선언했다면
# print(M2.eval())
# 와 같이 그냥 선언해도 오류가 발생하지 않음
# TensorFlow 2.x 버젼일 경우 numpy()를 이용해 값을 array로 가져온다.
print(M2.numpy())

# 변수 선언
W = tf.Variable(0, name="weight")

# TensorFlow 1.x 버젼일 경우 변수를 사용하기 전에 initalize 해주어야 한다.
# init_op = tf.initialize_all_variables()
# sess.run(init_op)
# TensorFlow 2.x 버젼일 경우 그냥 선언만 해줘도 된다.
print("W is :")
print(W.numpy())

# 변수 연산
# TensorFlow 1.x 버젼일 경우 그냥 연산을 하면 되지만
# TensorFlow 2.x 버젼일 경우 variable.assign(variable 연산 value)로 연산한다.

# TensorFlow 1.x에선 W += a
W.assign(W+a)
print("W after adding a:")
print(W.numpy())

# TensorFlow 1.x에선 W *= 2
W.assign(W*2)
print("W after multiplying 2:")
print(W.numpy())

E = d + b   # 1*2 + 2 = 4
print("E as defined:")
print(E.numpy())

# E와 d를 list로 동시 출력
print("E and d:")
# TensorFlow 1.x 에 경우
# print(sess.run([E,d]))
print([E.numpy(),d.numpy()])

# E와 변경된 d를 동시 출력
# TensorFlow에서는 변경된 값으로 node가 동시에 변경된다
print("E with custom d:4.")
# TensorFlow 1.x 에 경우 placeholder로 값을 변경할 수 있었다.
# placeholder란 처음에 변수를 선언할 때 값을 바로 주는 것이 아니라, 나중에 값을 던져주는 공간을 만들어주는 것
# print(sess.run(E, feed_dict = {d:4.}))
# 와 같이 사용할 수 있었다.
# 하지만 Tensorflow 2.x로 변경되면서 placeholder가 사라지고 함수를 통해 간결하게 처리
@tf.function
def add_b(x):
    return x + b
E = add_b(tf.constant(4))
print(E.numpy())
