import _pickle as pickle
import numpy as np
import visdom
import matplotlib.pyplot as plt
import cv2
import os
import sys
from math import sqrt, radians, sin, cos


def non_maximum_supression(image, n, d): # n - для размера окошка, d - минимальная длина линии
    h, w = image.shape
    max_arr = np.zeros((h+n*2, w+n*2))
    max_arr[n:h+n, n:w+n] = image

    for i in range(n, h+n):
        for j in range(n, w+n):
            max_r = max(np.max(max_arr[i - n:i + 1, j - n:j]), np.max(max_arr[i - n:i, j:j + n + 1]))
            max_l = max(np.max(max_arr[i:i + n + 1, j+1:j+n+1]), np.max(max_arr[i + 1:i + n + 1, j-n:j + 1]))
            if max_r > max_arr[i, j] or max_l > max_arr[i, j]:
                max_arr[i, j] = 0
    for i in range(n, h + n):
        for j in range(n, w + n):
            if max_arr[i, j] < d:
                max_arr[i, j] = 0
    return max_arr[n:h+n,n:w+n]


train_data = os.listdir('D:/CV_HW/Lines')
for i in range(0, len(train_data)):
    img = cv2.imread('D:/CV_HW/Lines/' + train_data[i])
    img_orig = cv2.imread('D:/CV_HW/Lines/' + train_data[i])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    edges = cv2.Canny(gray, 100, 200) # бинарное изображение

    diagonal = int(sqrt(height**2 + width**2))
    Theta = np.linspace(-(np.pi) / 2, np.pi, 1000)
    #cumalative_arr = np.zeros((diagonal, 1000))
    cumalative_arr = pickle.load(open('D:/CV_HW/img/cumalative_array_'+str(i), 'rb'))['c_arr'] # загрузка массивов
    print(edges.shape)
    print(cumalative_arr.shape)

# это часть закомментированного кода для новых изображений, в целях экономии времени, я сохраняла для изображений их кумулятивные массивы,
# потому, когда начала пробывать менять параметры (см.далее), закомментировала эту часть
    """for y in range(0, height):
        for x in range(0, width):
            #print(edges[y][x])
            if edges[y][x] > 0:
                for f in range(0, 1000):
                    #theta = radians(f)
                    theta = Theta[f]
                    r = round(x*cos(theta) + y*sin(theta))
                    #print(r, f, x, y)
                    if r < diagonal and r > 0:
                        cumalative_arr[r][f] += 1
                        #print(cumalative_arr[r][f])
                        #print(r, y, x, theta)
                #print(cumalative_arr)
    
    with open('D:/CV_HW/img/cumalative_array_'+str(i), 'wb') as f:  # сохраняю модель
        pickle.dump({'c_arr': cumalative_arr}, f)"""

    blur = cv2.GaussianBlur(cumalative_arr, (5,5), 1)

    color = np.array([255, 0, 0])
    # Я пыталась менять максимальную длину найденных линий и размер окошка для поиска максимумов, пробовала получить результаты получше
    for dd in range(3, 6):
        for max_line in range(1, 5):
            non_max_sup = non_maximum_supression(blur, dd, np.max(blur) / max_line) # Первые эксперименты были с dd = 3, а max_line = 1.5
            for x in range(0, width):
                for f in range(0, 1000):
                    for r in range(0, diagonal):
                        if non_max_sup[r][f] != 0:
                            if sin(Theta[f]) != 0:
                                y = (r - x * cos(Theta[f])) / sin(Theta[f])
                                if round(y) < height and round(y) > 0:
                                    img[round(y)][x] = color
                                    print(y)
            for y in range(0, height):
                for f in range(0, 1000):
                    for r in range(0, diagonal):
                        if non_max_sup[r][f] != 0:
                            if cos(Theta[f]) != 0:
                                x = (r - y * sin(Theta[f])) / cos(Theta[f])
                                if round(x) < width and round(x) > 0:
                                    img[y][round(x)] = color
                                    print(x)
            plt.subplot(2, 2, 1), plt.imshow(img_orig, cmap='gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])

            plt.subplot(2, 2, 2), plt.imshow(edges, cmap='gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

            plt.subplot(2, 2, 3), plt.imshow(blur, cmap='gray')
            plt.title('Blur'), plt.xticks([]), plt.yticks([])
            plt.subplot(2, 2, 4), plt.imshow(img, cmap='gray')
            plt.title('Lines'), plt.xticks([]), plt.yticks([])
            plt.savefig('D:/CV_HW/img2/img_' + str(i) + '_' + str(dd) + '_' + str(max_line) + '.png')
            # plt.show()


