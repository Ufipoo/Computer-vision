import _pickle as pickle
import numpy as np
import visdom
import matplotlib.pyplot as plt

if __name__ == '__main__':
    vis = visdom.Visdom()
    train_data, train_labels = [], []
    for i in range(1, 6):
        f = open('D:/CV-2019/cifar-10-batches-py/data_batch_' + str(i), 'rb')
        d = pickle.load(f, encoding='latin1')
        train_data.append(d['data'])
        train_labels.append(np.array(d['labels']))

    train_data = np.float32(np.array(train_data).reshape(50000, 3072))
    train_labels = np.array(train_labels).reshape(50000)

    valid_data, valid_labels = [], []
    test_data, test_labels = [], []
    f = open('D:/CV-2019/cifar-10-batches-py/test_batch', 'rb')
    d = pickle.load(f, encoding='latin1')
    valid_data = np.array(d['data'])
    valid_labels = np.array(d['labels'])
    test_data = np.array(d['data'])
    test_labels = np.array(d['labels'])

    classes = pickle.load(open('D:/CV-2019/cifar-10-batches-py/batches.meta', 'rb'), encoding='latin1')['label_names']
    #print(train_data)
    #print(train_labels)
    #print(valid_data)
    #print(valid_labels)
    #print(classes)
    #print(np.shape(valid_data))

    train_size = np.shape(train_data)[0] # размер обуч. выборки = 50 000
    valid_size = np.shape(valid_data)[0] # размер валид. выборке = 10 000
    x_size = np.shape(train_data)[1] # размер вход. вектора = 3072
    num_class = len(classes) # количество классов = 10
    print(train_size, valid_size, x_size, num_class)

    def normalization(x_j):
        x_j = x_j.reshape((int(x_size/3), 3))
        x_mean = np.mean(x_j, 0)
        x_std = np.std(x_j, 0)
        return ((x_j - x_mean)/x_std).reshape(x_size)

    def softmax(y_i):
        e_y = np.exp(y_i - np.max(y_i, 1).reshape(batch_size,1))
        e_y = e_y / e_y.sum(axis=1).reshape(batch_size,1)
        return e_y

    def one_hot_encoding(label):
        vec_ohe = np.zeros((np.size(label), num_class))
        for i in range(0, np.size(label)):
            vec_ohe[i][label[i]] = 1
        return vec_ohe

    for i in range(0, train_size):
        train_data[i] = normalization(train_data[i])
    for i in range(0, valid_size):
        valid_data[i] = normalization(valid_data[i])

    w = np.random.normal(0, 1/x_size, size=(num_class, x_size)) # На лекциях в прошлом году Евгений Викторович говорил,
    # что лучше всего инициализировать веса таким образом, сигма = 0, дисперсия - 1/М, где М - размерность вход.ветора
    bias = np.random.randn(num_class, 1)
    print(np.shape(w))


    global_iteration = 0
    global_iteration2 = 0
    lr = 0.001 # пробывались значения больше, но по графику точность сильно скакала, а это дала более менее нормальные рез-ты (скорость для работы с минибатчами)
    accuracy2 = 0 # для валидации
    batch_size = 10 # размер минибатчей
    max_accuracy_valid = 0


    for epoch in range(0, 300): # обучение останавливалось, когда точность переставала рсти и держалась примерно на одном уровне
        dw = 0 # для градиента весов
        db = 0 # для градиента смещенеия
        accuracy = 0 # точность на тренировчной выборке
        for i in range(0, int(train_size/batch_size)):
            y = np.dot(w, train_data[i*batch_size:i*batch_size+batch_size].T) + bias
            y_softmax = softmax(y.T)

            y_ohe = one_hot_encoding(np.argmax(y_softmax, 1))
            for j in range(0, batch_size):
                if all(y_ohe[j] == one_hot_encoding(train_labels[i * batch_size:i * batch_size + batch_size])[j]):
                    accuracy += 1

            global_iteration += 1
            if global_iteration % ((train_size/batch_size)/5) == 0: # промежуточные точности
                print('iteration ' + str(i + 1) + ' | accuracy = ' + str(accuracy / ((i + 1)*batch_size)))

            dw += np.dot((y_softmax - one_hot_encoding(train_labels[i * batch_size:i * batch_size + batch_size])).T,
                         train_data[i * batch_size:i * batch_size + batch_size])
            db += np.sum(y_softmax - one_hot_encoding(train_labels[i * batch_size:i * batch_size + batch_size]),
                         0).reshape(num_class, 1)


        print('TRAIN ' + str(global_iteration) + ' | accuracy = ' + str(accuracy / train_size))
        vis.line(Y=np.array([accuracy / train_size]), X=np.array([global_iteration]), win='plot_acc',
                 update='append', opts=dict(title='Accuracy train', xlabel='Iteration', ylabel='Accuracy'))

        w = w - lr * (dw / train_size)
        bias = bias - lr * (db / train_size)

    # ВАЛИДАЦИЯ
        # после каждого полного прохода по обуч. выборке - прохожу по валид и считаю точность
        for i in range(0, int(valid_size/batch_size)):
            y = np.dot(w, valid_data[i * batch_size:i * batch_size + batch_size].T) + bias
            y_softmax = softmax(y.T)
            y_ohe = one_hot_encoding(np.argmax(y_softmax, 1))
            for j in range(0, batch_size):
                if all(y_ohe[j] == one_hot_encoding(valid_labels[i * batch_size:i * batch_size + batch_size])[j]):
                    accuracy2 += 1

            global_iteration2 += 1
            if global_iteration2 % ((valid_size/batch_size)*4) == 0: # каждые 4 эпохи валидации считаю среднюю точность и вывожу график
                vis.line(Y=np.array([accuracy2 / (valid_size*4)]), X=np.array([global_iteration2]), win='plot_acc_val',
                         update='append',
                         opts=dict(title='Accuracy validation', xlabel='Iteration', ylabel='Accuracy'))
                print('VALIDATION ' + str(global_iteration2) + ' | accuracy = ' + str(accuracy2 / (valid_size*4)))
                with open('w_2/itr_' + str(global_iteration2) + '.pickle', 'wb') as f: # сохраняю модель
                    pickle.dump({
                        'w': w,
                        'b':bias,
                        'global_iteration': global_iteration2}, f)
                if (accuracy2 / (valid_size*4)) > max_accuracy_valid:
                    max_accuracy_valid = global_iteration2
                accuracy2 = 0

    """________ПРОВЕРКА НА ВАЛИДАЦИОННОЙ/ТЕСТОВОЙ ВЫБОРКЕ____________"""
    f = open('w_2/itr_' + str(max_accuracy_valid) + '.pickle', 'rb')
    d = pickle.load(f, encoding='latin1')
    best_model_w = d['w']
    best_model_b = d['b']

    confusion_matrix = np.zeros((num_class, num_class))
    valid_accuracy = 0
    max_accuracy_true, max_accuracy_false = [], []
    class_true, class_false = [], []
    num_true, num_false = [], []

    for i in range(0, int(valid_size / batch_size)):
        y = np.dot(best_model_w, valid_data[i * batch_size:i * batch_size + batch_size].T) + best_model_b
        y_softmax = softmax(y.T)
        y_ohe = one_hot_encoding(np.argmax(y_softmax, 1))
        for j in range(0, batch_size):
            if all(y_ohe[j] == one_hot_encoding(valid_labels[i * batch_size:i * batch_size + batch_size])[j]):
                valid_accuracy += 1
                confusion_matrix[valid_labels[i * batch_size +j]][valid_labels[i * batch_size + j]] += 1
                max_accuracy_true.append(np.max(y_softmax[j]))
                class_true.append(np.argmax(y_softmax[j]))
                num_true.append(i * batch_size +j)

            else:
                confusion_matrix[valid_labels[i * batch_size +j]][np.argmax(y_softmax[j])] += 1
                max_accuracy_false.append(np.max(y_softmax[j]))
                class_false.append(np.argmax(y_softmax[j]))
                num_false.append(i * batch_size +j)

    print('ACCURACY TEST = ' + str(valid_accuracy / 10000))
    print('CONFUSION MATRIX')
    print(confusion_matrix)
    sort_true = [x for y, x in sorted(zip(max_accuracy_true, zip(class_true, num_true)))]
    sort_false = [x for y, x in sorted(zip(max_accuracy_false, zip(class_false, num_false)))]

    for i in range(1, 4):
        img = test_data[sort_true[len(sort_true) - i][1]].reshape((3, 32, 32))
        img = img.transpose(1, 2, 0)
        plt.figure()
        plt.title(classes[sort_true[len(sort_true) - i][0]] + '- true')
        plt.imshow(img)
        plt.show()
    print('best results | ' + str(sort_true[len(sort_true) - 3:]))
    print('bad results | ' + str(sort_false[len(sort_false) - 3:]))
    for i in range(1, 4):
        img = test_data[sort_false[len(sort_false) - i][1]].reshape((3, 32, 32))
        img = img.transpose(1, 2, 0)
        plt.figure()
        plt.title(classes[sort_false[len(sort_false) - i][0]] + '- false,' + classes[
            test_labels[sort_false[len(sort_false) - i][1]]] + ' - true')
        plt.imshow(img)
        plt.show()







