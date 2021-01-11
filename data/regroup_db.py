import os
import numpy as np
import cv2




i=0
for root, dirs, file in os.walk("Published_database_FV-USM_Dec2013/2nd_session/extractedvein/"):
    for dir in dirs:
        for file in os.walk(root+dir):
            for each in file[2]:
                print(root+dir+each)
                x = cv2.imread(root+dir+"/"+each)
                # cv2.imshow("image", x)
                # cv2.waitKey(0)
                cv2.imwrite("fv2bright02fv/testA/finger{}.png".format(i), x)
                i+=1
    