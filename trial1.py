import cv2
import numpy as np
from matplotlib import pyplot as plt
#cap = cv2.VideoCapture("gif.mp4")

#ret, frame1 = cap.read()
list_names=['frame' + str(i+25) + '.jpg' for i in range(10)]

# Read in the first frame
frame1 = cv2.imread(list_names[0])
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
counter=1
while(counter<len(list_names)):
    #ret, frame2 = cap.read()
    frame2 = cv2.imread(list_names[counter])
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    hist=cv2.calcHist([rgb],[0],None,[256],[0,256])
    plt.figure()
    plt.title("GrayScale Hist")
    plt.xlabel("magnitude")
    plt.ylabel("angle")
    plt.plot(hist)
    plt.xlim([0,256])
    plt.show()
    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next.copy()
    counter+=1

#cap.release()
cv2.destroyAllWindows()
