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
encoded = Dense(32, activation='relu')
encoded = encoded(input_img) #input 을 받은 dense레이어의 출력
decoded = Dense(784, activation='sigmoid') #입력데이터를 0~1사이로 정규화
decoded = decoded(encoded)
autoencoder = Model(input_img, decoded) #처음부터 끝까지 시퀀셜하게 이어짐

autoencoder.summary()
# autoencoder :암호화 규칙을 자기가 만듦(학습 후에 weight와 bias로 규칙 만듦)/(새로운 신호 만들때마다 )학습된 신호만 압축&보관 가능

encoder = Model(input_img, encoded) #하나의 모델> 앞부분만 떼어냄(모델 두개 아님)
encoder.summary()

encoder_input = Input(shape=(32,)) #인풋은 레이어 아님/ 입력되는 빈 공간일뿐
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoder_input, decoder_layer(encoder_input))
decoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data() #y_train 필요없음: 라벨 필요없음 > 자기지도학습 (비지도학습)
# (답이없으면 비지도학습/ 있으면 지도학습)
x_train = x_train/255 # (x_test/=255) 와 같음
x_test = x_test/255

flatted_x_train = x_train.reshape(-1,784)
flatted_x_test = x_test.reshape(-1,784)

fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train, epochs=50,
                           batch_size=256, validation_data=(flatted_x_test, flatted_x_test))

encoded_img = encoder.predict(x_test[:10].reshape(-1,784))
decoded_img = decoder.predict(encoded_img)

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
plt.show() #압축하는 과정에서 손실이 있었으나 복원이 됨(노이즈가 있으면 노이즈 제거 후 복원)








