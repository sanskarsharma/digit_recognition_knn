import cv2
import numpy as np

img = cv2.imread("digits.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cells = [ np.hsplit(row, 100) for row in np.vsplit(gray_img, 50)]

arr = np.array(cells)

train_data = arr[:, :90]
train_data = train_data.reshape(-1,400).astype(np.float32)

test_data = arr[:, 90:100]
test_data = test_data.reshape(-1,400).astype(np.float32)

k = np.arange(10)
train_labels = np.repeat(k,450)[:, np.newaxis]

test_labels = np.repeat(k,50)[:, np.newaxis]

knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)


imgdigit = cv2.imread("digit_4.jpg")
gray_img_dig = cv2.cvtColor(imgdigit, cv2.COLOR_BGR2GRAY)
imgdigit = cv2.resize(gray_img_dig, (20,20))
arraa = np.array(imgdigit).reshape(-1,400).astype(np.float32)
print(arraa.shape)

ret , result , neighbours, dist = knn.findNearest( test_data,k= 5)
print(type(result))
print(type(ret))
print(result[0])

matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print(accuracy)

np.savez('knn_data.npz',train_data=train_data, train_labels=train_labels)

#increase contrast and sharpness for accuracy