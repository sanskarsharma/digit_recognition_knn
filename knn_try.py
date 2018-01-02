
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from matplotlib import pyplot as plt


# In[2]:


img = cv2.imread('digits.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[3]:


cells = [np.hsplit(row,100) for row in np.vsplit(gray_img,50)]
type(cells)


# In[4]:


arr = np.array(cells)
print(type(arr))
print(arr.shape)


# In[16]:


train_data = arr[:, :90]   # vertically spliiting image array
print(train_data.shape)
train_data = train_data.reshape( -1, 400).astype(np.float32)# reshaping 
train_data.shape


# In[17]:


test_data = arr[:, 90:100]
test_data = test_data.reshape(-1, 400).astype(np.float32)
test_data.shape
test_data


# In[18]:


# creating lables for our data
k = np.arange(10)
train_labels = np.repeat(k, 450)[:, np.newaxis]
train_labels


# In[20]:


test_labels= np.repeat(k, 50)[:, np.newaxis]
test_labels


# In[22]:


knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)


# In[23]:


imgdigit = cv2.imread("digit_7.png")
gray_img_dig = cv2.cvtColor(imgdigit, cv2.COLOR_BGR2GRAY)
imgdigit = cv2.resize(gray_img_dig, (20,20))
arraa = np.array(imgdigit).reshape(-1,400).astype(np.float32)
print(arraa.shape)

ret , result , neighbours, dist = knn.findNearest( test_data,k= 5)
print(type(result))
print(type(ret))
#print(result[0])


# In[24]:


matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print(accuracy)

