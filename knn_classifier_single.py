import cv2
import numpy as np

with np.load('knn_data.npz') as data:
    print( data.files )
    train_data = data['train_data']
    train_labels = data['train_labels']

    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    
    imgdigit = cv2.imread("digit_61.jpg")
    gray_img = cv2.cvtColor(imgdigit, cv2.COLOR_BGR2GRAY)

    imgdigit = cv2.resize(gray_img, (20,20))

    #cv2.imshow("digit is", imgdigit)
    arraa = np.array(imgdigit).reshape(-1,400).astype(np.float32)
    print(arraa.shape)

    ret , result , neighbours, dist = knn.findNearest( arraa,k= 5)
    print(type(result))
    print(type(ret))
    print(result[0])