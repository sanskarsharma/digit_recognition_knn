import cv2
import numpy as np

# Now load the data
with np.load('knn_data.npz') as data:
    print( data.files )
    train_data = data['train_data']
    train_labels = data['train_labels']

    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    video = cv2.VideoCapture(1)     #loading vid from secondary camera
    framecount = 0
    while True:
        checksuccess, frame = video.read()
        
        # converting frame to grayscale              
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 480 by 640

        biggeredge = grayframe.shape[1]
        smalleredge = grayframe.shape[0]
        if grayframe.shape[1] >= grayframe.shape[0] :
            biggeredge = grayframe.shape[1]
            smalleredge = grayframe.shape[0]
        else:
            biggeredge = grayframe.shape[0]
            smalleredge = grayframe.shape[1]

        cropx1 = int((biggeredge-smalleredge)/2)
        cropx2 = cropx1 + smalleredge
        cropy1 = 0
        cropy2 = smalleredge

        grayframe = grayframe[cropx1:cropx2, cropy1:cropy2]
        
        (thresh, im_bw) = cv2.threshold(grayframe, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = 127
        grayframe = cv2.threshold(grayframe, thresh, 255, cv2.THRESH_BINARY)[1]

        grayframe = np.invert(grayframe)
        #print(grayframe.shape)
        # showing frame in a window
        cv2.imshow("capturing video ...", grayframe)

        # to find which key was pressed
        keypress = cv2.waitKey(1)   # waiting for 1 millisecond before moving ahead 
                                # this runs in while loop and generates frames continuously which we see as video in our window

        if keypress==ord('q'):
            break
        if keypress==ord('m'):
            gray_img_dig = grayframe
            imgdigit = cv2.resize(gray_img_dig, (20,20))
            cv2.imshow("digit is", imgdigit)
            arraa = np.array(imgdigit).reshape(-1,400).astype(np.float32)
            print(arraa.shape)

            ret , result , neighbours, dist = knn.findNearest( arraa,k= 3)
            print(type(result))
            print(type(ret))
            print(framecount)
            print(result[0])
            framecount = framecount+1

    video.release()
    cv2.destroyAllWindows
    #imgdigit = cv2.imread("digit_7.png")
    
    