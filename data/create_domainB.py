import os
import numpy as np
import cv2
import tensorflow as tf



l = np.random.randint(0, 2951, 100)
print(l)

for i in l:
    path = "fv2bright02fv/testA/finger{}.png".format(i)
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=1)

    tmp = tf.image.adjust_brightness(img, 0.2)
    # print(tmp.numpy())
    cv2.imwrite("fv2bright02fv/testB/finger{}.png".format(i), tmp.numpy())
    os.remove(path)

# i=0
# for root, dirs, file in os.walk("Published_database_FV-USM_Dec2013/2nd_session/extractedvein/"):
#     for dir in dirs:
#         for file in os.walk(root+dir):
#             for each in file[2]:
#                 print(root+dir+each)
#                 x = cv2.imread(root+dir+"/"+each)
#                 # cv2.imshow("image", x)
#                 # cv2.waitKey(0)
#                 cv2.imwrite("fv2bright02fv/testA/finger{}.png".format(i), x)
#                 i+=1
    