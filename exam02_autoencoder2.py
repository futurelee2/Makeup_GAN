import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

#codec = encoder+decoder   (암호화로 바꾸는 규칙을 가지고 있음)
#(왜군: 시그널/ 봉화: 코드)
#coder:시그널을 코드로 변환
#decoder: 코드를 신호로 변환
#코드를 암호화하면서 압축 기술 필요

input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='sigmoid')(encoded)
decoded = Dense(128, activation='sigmoid')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
autoencoder = Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train/255
x_test = x_test/255

flatted_x_train = x_train.reshape(-1,784)
flatted_x_test = x_test.reshape(-1,784)

fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train, epochs=50,
                           batch_size=256, validation_data=(flatted_x_test, flatted_x_test))

decoded_img = autoencoder.predict(flatted_x_test[:10])

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,10,i+1)
    plt.imshow(x_test[i]) #입력이미지 출력
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,10,i+1+n)
    plt.imshow(decoded_img[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show() #압축하는 과정에서 손실이 있었으나 복원이 됨(노이즈가 있으면 노이즈 제거 후 복원) 위:원본/ 아래:복원









