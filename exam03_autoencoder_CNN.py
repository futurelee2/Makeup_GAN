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

input_img = Input(shape=(28,28,1)) #이미지 사이즈 28*28 / 모노칼라 = 1
x = Conv2D(16,(3,3), activation='relu',padding='same')(input_img) # 필터 16장/ 겹치는 이미지 사이즈 3*3
x = MaxPool2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3), activation='relu',padding='same')(x)
x = MaxPool2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3), activation='relu',padding='same')(x)
encoded = MaxPool2D((2,2), padding='same')(x)

x = Conv2D(8,(3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x) #4 의 두배 8
x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x) #8의 두배 16
x = Conv2D(16,(3,3), activation='relu')(x) # 입출력 사이즈 28로 맞추기 위해서
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1,(3,3), activation='sigmoid', padding='same')(x) #최종출력 1장 / 0~1사이값 출력

autoencoder = Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train/255
x_test = x_test/255

conv_x_train = x_train.reshape(-1,28,28,1)
conv_x_test = x_test.reshape(-1,28,28,1)

noise_factor = 0.5
conv_x_train_noisy = conv_x_train + np.random.normal(0,1, size=conv_x_train.shape)*noise_factor # 평균0 표준편차 1인 표준정규분포를 따르는 noise만들기
conv_x_train_noisy = np.clip(conv_x_train_noisy,0.0,1.0) # 최대 최소값 제한(0 보다 작으면 0, 1보다 크면 1)
conv_x_test_noisy = conv_x_test + np.random.normal(0,1, size=conv_x_test.shape)*noise_factor
conv_x_test_noisy = np.clip(conv_x_test_noisy,0.0,1.0)


fit_hist = autoencoder.fit(conv_x_train_noisy, conv_x_train, epochs=50,
                           batch_size=256, validation_data=(conv_x_test_noisy, conv_x_test))

autoencoder.save('./models/autoencoder_noisy.h5')

decoded_img = autoencoder.predict(conv_x_test[:10])

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,10,i+1)
    plt.imshow(x_test[i]) #입력이미지 출력
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,10,i+1+n)
    plt.imshow(conv_x_test_noisy[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show() #압축하는 과정에서 손실이 있었으나 복원이 됨(노이즈가 있으면 노이즈 제거 후 복원) 위:원본/ 아래:복원









