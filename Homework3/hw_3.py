import _pickle as pickle
import numpy as np
import visdom
import matplotlib.pyplot as plt
import cv2
import os


def m_II(image):
    a = np.cumsum(image, axis=1)
    h, w = np.shape(a)
    A = np.zeros((h + 1, w + 1))
    for i in range(0, h):
        for j in range(0, w):
            A[i + 1, j + 1] = np.sum(a[0:i + 1, j], axis=0)
    return A

#A = np.r_[[np.zeros(w)], A]
#A = np.c_[np.zeros(h+1), A]

### Маска №1
def mask_1(A, h, w):
    x_i = []
    for i_h in range(1, h+1):
        for j_w in range(1, (w // 2) + 1):
            for i in range(i_h, h+1):
                #print('_____')
                #print('ширина масски ', j_w)
                for j in range(j_w, w-j_w+1):
                    #print('++++++++++++')
                    #print('номер столбца ', j)
                    white = A[i,j] - A[i-i_h, j] - A[i, j-j_w] + A[i-i_h, j-j_w]
                    black = A[i, j+j_w] - A[i-i_h, j+j_w] - A[i, j] + A[i-i_h, j]
                    w_b = 2*A[i, j] - 2*A[i-i_h, j] - A[i, j-j_w] + A[i-i_h, j-j_w] - A[i, j+j_w] + A[i-i_h, j+j_w]
                    #print(w_b, i_h, j_w, i, j)
                    x_i.append(w_b)
    return x_i

### Маска №2
def mask_2(A, h, w):
    x_i = []
    for i_h in range(1, h+1):
        for j_w in range(1, (w // 3) + 1):
            for i in range(i_h, h+1):
                #print('_____')
                #print('ширина масски ', j_w)
                for j in range(j_w, w - 3*j_w + j_w + 1):
                    #print('++++++++++++')
                    #print('номер столбца ', j)
                    white = A[i,j] - A[i-i_h, j] - A[i, j-j_w] + A[i-i_h, j-j_w]
                    black = A[i, j+j_w] - A[i-i_h, j+j_w] - A[i, j] + A[i-i_h, j]
                    w_b = 2*A[i, j] - 2*A[i-i_h, j] - 2*A[i, j+j_w] + 2*A[i-i_h, j+j_w] +A[i-i_h, j-j_w] - A[i, j-j_w] + A[i, j + j_w*2] - A[i-i_h, j+j_w*2]
                    #print(w_b, i_h, j_w, i, j)
                    x_i.append(w_b)
    return x_i

###  Маска №3
def mask_3(A, h, w):
    x_i = []
    for i_h in range(1, (h // 2) + 1):
        for j_w in range(1, w + 1):
            for i in range(i_h, h - i_h + 1):
                #print('_____')
                #print('ширина масски ', j_w)
                for j in range(j_w, w + 1):
                    #print('++++++++++++')
                    #print('номер столбца ', j)
                    w_b = 2*A[i, j] - 2*A[i, j-j_w] + A[i-i_h, j-j_w] - A[i-i_h, j] - A[i+i_h, j] + A[i+i_h, j-j_w]
                    #print(w_b, i_h, j_w, i, j)
                    x_i.append(w_b)
    return x_i

###  Маска №4
def mask_4(A, h, w):
    x_i = []
    for i_h in range(1, (h // 3) + 1):
        for j_w in range(1, w + 1):
            for i in range(i_h, h - 3*i_h + i_h + 1):
                #print('_____')
                #print('ширина масски ', j_w)
                for j in range(j_w, w + 1):
                    #print('++++++++++++')
                    #print('номер столбца ', j)
                    w_b = 2*A[i, j] - 2*A[i, j-j_w] - 2*A[i+i_h, j] + 2*A[i+i_h, j-j_w] + \
                          A[i-i_h, j-j_w] - A[i-i_h, j] + A[i+i_h*2, j] - A[i+i_h*2, j-j_w]
                    #print(w_b, i_h, j_w, i, j)
                    x_i.append(w_b)
    return x_i

###  Маска №5
def mask_5(A, h, w):
    x_i = []
    for i_h in range(1, (h // 2) + 1):
        for j_w in range(1, (w // 2) + 1):
            for i in range(i_h, h - i_h + 1):
                #print('_____')
                #print('ширина масски ', j_w)
                for j in range(j_w, w - j_w + 1):
                    #print('++++++++++++')
                    #print('номер столбца ', j)
                    w_b = 4*A[i, j] - 2*A[i, j-j_w] - 2*A[i-i_h, j] - 2*A[i+i_h, j] - 2*A[i, j+j_w] +\
                          A[i-i_h, j-j_w] + A[i+i_h, j+j_w] + A[i-i_h, j+j_w] + A[i+i_h, j-j_w]
                    #print(w_b, i_h, j_w, i, j)
                    x_i.append(w_b)
    return x_i

train_data = os.listdir('D:/CV_HW/train/face')
print(train_data[0])
matrix_feature = []
for i in range(0, len(train_data)):
    img = cv2.imread('D:/CV_HW/train/face/' + train_data[i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('img',gray)
    #cv2.waitKey(0)
    II = m_II(gray)
    height, width = np.shape(gray)[0], np.shape(gray)[1]
    mask1 = mask_1(II, height, width)
    mask2 = mask_2(II, height, width)
    mask3 = mask_3(II, height, width)
    mask4 = mask_4(II, height, width)
    mask5 = mask_5(II, height, width)
    print(len(mask1), len(mask2), len(mask3), len(mask4), len(mask5))
    haar_like_feature = np.concatenate((mask1, mask2, mask3, mask4, mask5))
    print(haar_like_feature)
    matrix_feature.append(haar_like_feature)
print(np.array(matrix_feature))
matrix_feature = np.array(matrix_feature)
with open('feature.pickle', 'wb') as f:  # сохраняю модель
    pickle.dump({'matrix_feature' : matrix_feature}, f)

