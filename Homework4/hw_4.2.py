import _pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from math import sqrt, radians, sin, cos

train_data = os.listdir('D:/CV_HW/Coins')

for i in range(0, len(train_data)):
    img = cv2.imread('D:/CV_HW/Coins/' + train_data[i])
    img_orig = cv2.imread('D:/CV_HW/Coins/' + train_data[i])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    edges = cv2.Canny(gray, 100, 200) # бинарное изображение

    R = round(sqrt(height ** 2 + width ** 2))

    max_R = round(min(height, width) / 2)
    cumalative_arr = np.zeros((height, width, max_R))
    img_cum = np.zeros((height, width)) #  для рисования фазовых окружностей

    color = np.array([255, 0, 0])
    for b in range(0, height):
        for a in range(0, width):
            #print(edges[y][x])
            if edges[b][a] > 0:
                for y in range(0, height):
                    for x in range(0, width):
                        r = round(sqrt((x-a)**2 + (y-b)**2))
                        if r < max_R:
                            cumalative_arr[y][x][r] += 1
                            img_cum[y][x] += 1
                            #print('r = ', r, ', a = ', a, ', b = ', b)

    non_max = np.zeros((height, width, max_R))

    for y in range(0, height):
        for x in range(0, width):
            maxp = np.max(cumalative_arr[y][x])
            index_max = np.argmax(cumalative_arr[y][x])
            non_max[y][x][index_max] = maxp
    max_RR = np.max(non_max)

    for b in range(0, height):
        for a in range(0, width):
            for r in range(0, max_R):
                if non_max[b][a][r] > max_RR/1.5:
                    for y in range(0, height):
                        for x in range(0, width):
                            if r == round(sqrt((x - a) ** 2 + (y - b) ** 2)):
                                img[y][x] = color



    plt.subplot(2, 2, 1), plt.imshow(img_orig, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2), plt.imshow(edges, cmap='gray')
    plt.title('Edge'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 3), plt.imshow(img_cum, cmap='gray')
    plt.title('Cum_arr'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 4), plt.imshow(img, cmap='gray')
    plt.title('End'), plt.xticks([]), plt.yticks([])

    plt.savefig('D:/CV_HW/img_circl/img_' + str(i) + '.png')
    #plt.show()
