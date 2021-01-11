import cv2
from tensorflow.python.ops.gen_array_ops import concat
from lib.biometrics import extract_features
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

path = "../data/template/trainA/finger0.png"
img = tf.io.read_file(path)
img = tf.io.decode_image(img, channels=1)

# print(img)





# # tmp = tf.image.resize_with_pad(img, 256, 256)

# plt.imshow(img.numpy())
# plt.show()

tmp1 = tf.image.adjust_brightness(img, 0.1)
tmp2 = tf.image.adjust_brightness(img, 0.2)
tmp3 = tf.image.adjust_brightness(img, 0.3)
tmp4 = tf.image.adjust_brightness(img, -0.1)
tmp5 = tf.image.adjust_brightness(img, -0.2)

# final = tf.concat([img, tmp1, tmp2, tmp3, tmp4, tmp5], axis=1)
# print(final)
# plt.imshow(final.numpy(), cmap="gray")
# plt.show()


final2 = np.concatenate((extract_features(img.numpy()[:,:,0]), extract_features(tmp2.numpy()[:,:,0])), axis=1)

plt.imshow(final2, cmap="gray")
plt.show()
mae = tf.keras.losses.MeanAbsoluteError()
print(mae(extract_features(img.numpy()[:,:,0]), extract_features(tmp2.numpy()[:,:,0])))

# ##################################333


# tmp1 = tf.image.adjust_contrast(img, 1)
# tmp2 = tf.image.adjust_contrast(img, 3)
# tmp3 = tf.image.adjust_contrast(img, 5)
# tmp4 = tf.image.adjust_contrast(img, 7)
# tmp5 = tf.image.adjust_contrast(img, 9)

# final = tf.concat([img, tmp1, tmp2, tmp3, tmp4, tmp5], axis=1)
# print(final)
# plt.imshow(final.numpy())
# plt.show()

# tmp1 = tf.image.adjust_saturation(img, 1)
# tmp2 = tf.image.adjust_saturation(img, 10)
# tmp3 = tf.image.adjust_saturation(img, 20)
# tmp4 = tf.image.adjust_saturation(img, 30)
# tmp5 = tf.image.adjust_saturation(img, 200)

# final = tf.concat([img, tmp1, tmp2, tmp3, tmp4, tmp5], axis=1)
# print(final)
# plt.imshow(final.numpy())
# plt.show()

# feat1 = extract_features(img.numpy()[:,:,1])*255
# import matplotlib.pyplot as plt
# import numpy as np
# plt.imshow(feat1)
# plt.show()
# print(".")
# tmp = np.concatenate((feat1, img), axis=1)

# plt.imshow(tmp, cmap="gray")
# plt.show()

# print(tmp)
