import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist
import os

OUT_DIR = './DNN_out'
img_shape = (28,28,1)
epochs = 100000
batch_size = 128
noise = 100
sample_interval = 100 #노이즈 100개

(x_train, _), (_,_) = mnist.load_data()
print(x_train.shape)

x_train = x_train/127.5 - 1
x_train = np.expand_dims(x_train, axis=3)
print(x_train.shape)

generator = Sequential()
generator.add(Dense(128, input_dim=noise))
generator.add(LeakyReLU(alpha=0.01)) #음수에서의 기울기
#흐릿한 부분에 흐릿하게, 진한부분에는 강하게 반응하기
generator.add(Dense(784, activation='tanh'))
generator.add(Reshape(img_shape))
generator.summary()

lrelu = LeakyReLU(alpha=0.01)

discriminater = Sequential()
discriminater.add(Flatten(input_shape=img_shape))
discriminater.add(Dense(128, activation=lrelu))
discriminater.add(Dense(1, activation='sigmoid'))
discriminater.summary()
discriminater.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
discriminater.trainable = False  # discriminater 학습 안되게 막아줌


gan_model = Sequential()
gan_model.add(generator)
gan_model.add(discriminater)
gan_model.summary()
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

#이진분류기 타겟
real = np.ones((batch_size,1)) #1로 채워진 행렬을 만들어줌/ 개수는 배치사이즈만큼
print(real)
fake = np.zeros((batch_size,1))
print(fake)

for epoch in range(epochs):
    idx = np.random.randint(0,x_train.shape[0],batch_size) #0~6사이의 인트값을 랜덤하게 뽑는다/ 배치사이즈 128
    real_imgs = x_train[idx]

    z = np.random.normal(0,1,(batch_size, noise))
    fake_imgs = generator.predict(z)

    d_hist_real = discriminater.train_on_batch(real_imgs, real) #이것만 학습하고 끝남(1 epoch)
    d_hist_fake = discriminater.train_on_batch(fake_imgs, fake)

    d_loss, d_acc = np.add(d_hist_fake, d_hist_real)*0.5 #평균값


    if epoch %2 == 0:
        z = np.random.normal(0,1,(batch_size, noise))
        gan_hist = gan_model.train_on_batch(z, real) #1이라고 답하게 학습 (generator만 학습됨)
        #학습 시켯을때 어떤게 학습이 잘 되는지 보고 균형이 안맞을경우 맞춰줘야함> GAN모델이 학습이 너무 잘 되어서 절반만 학습되도록


    if epoch % sample_interval == 0: #100에폭마다 한번씩만 프린트하고, 이미지 그려서 저장해줌 (학습하고 관계없음)
        print('%d, [D loss: %f, acc.: %.2f%%], [G loss: %f]'
              %(epoch, d_loss, d_loss, gan_hist))
        row = col = 4
        z = np.random.normal(0,1,(row*col, noise))
        fake_imgs = generator.predict(z)
        fake_imgs = 0.5*fake_imgs + 0.5
        _, axs = plt.subplots(row, col, figsize=(5,5),sharey=True, sharex=True) # X,Y축을 서로 공유(True)하면 스케일 같아짐
        cont = 0
        for i in range(row):
            for j in range(col):
                axs[i,j].imshow(fake_imgs[cont, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                cont += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(epoch+1))
        plt.savefig(path)
        plt.close()