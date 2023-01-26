#exam 07만 아나콘다 가상환경 만들어줌 (아나콘다 필요한 패키지들 깔기)
# > 가상환경 아나콘다로 설정 후
# 터미널 : conda install -c conda-forge dlib
import dlib #얼굴찾을때 사용
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf #버전1에 맞춤
tf.disable_v2_behavior() #버전 2 비활성화
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

detector = dlib.get_frontal_face_detector() #앞 얼굴을 찾아주는 탐지기 역할
shape = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat') #얼굴의 다섯가지 랜드마크로 찾음 (5개 =눈의 양쪽 끝점, 인중)

# img = dlib.load_rgb_image('./imgs/02.jpg')
# plt.figure(figsize=(16,10))
# plt.imshow(img)
# plt.show()
#
# img_result = img.copy()
# dets = detector(img,1)  #탐지된 것 들
#
# if len(dets) == 0:
#     print('Not find faces')
#
# else: #얼굴 개수 만큼 그리기 (사각형 시작점: 항상 왼쪽 위 0.0)
#     fig, ax = plt.subplots(1, figsize=(10,16))
#     for det in dets:
#         x, y, w, h = det.left(), det.top(), det.width(), det.height()
#         rect = patches.Rectangle((x,y),w,h, linewidth=2, edgecolor='b', facecolor='None')
#         #사각형(Rectangle) 그리기/ facecolor = 사각형 안(면)의 색깔
#         ax.add_patch(rect)
#
# ax.imshow(img_result)
# plt.show()
# fig, ax = plt.subplots(1, figsize=(10,6))
# obj = dlib.full_object_detections()
#
# for detection in dets:
#     s = shape(img, detection)
#     obj.append(s)
#
#     for point in s.parts():
#         circle = patches.Circle((point.x, point.y), radius=3, edgecolor='b', facecolor='b')
#         ax.add_patch(circle)
#     ax.imshow(img_result)
# plt.show()
#
#
def align_face(img):
    dets = detector(img,1)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = shape(img, detection)
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)
    return faces
#
#
# test_faces = align_face(img)
# fig, axes = plt.subplots(1,len(test_faces)+1, figsize=(10,8))
# axes[0].imshow(img)
# for i, face in enumerate(test_faces):
#     axes[i+1].imshow(face)
# plt.show()

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.global_variables_initializer())
sess.run(init_op)

saver = tf.train.import_meta_graph('./models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

def preprocess(img):
    return img / 127.5-1
def depricess(img):
    return 0.5 * img +0.5 #생성된 이미지를 되돌릴때?

img1 = dlib.load_rgb_image('./imgs/no_makeup/01.jpg')
img1_faces = align_face(img1)

img2 = dlib.load_rgb_image('./imgs/makeup/XMY-136.png')
img2_faces = align_face(img2)

# fig, axes = plt.subplots(1, 2, figsize=(8,5))
# axes[0].imshow(img1_faces[0])
# axes[1].imshow(img2_faces[0])
# plt.show()


src_img = img1_faces[0]
ref_img = img2_faces[0]

X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0)

Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict = {X:X_img, Y:Y_img})
output_img = depricess(output[0])

fig, axes = plt.subplots(1, 3, figsize=(8,5))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
axes[2].imshow(output_img)
plt.show()
