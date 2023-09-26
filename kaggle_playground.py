import cv2
import matplotlib.pyplot as plt

image = cv2.imread("my_left_eye.jpg")

fig, ax = plt.subplots(3, 1)

ax[0].imshow(image[:, :, 0])
ax[1].imshow(image[:, :, 1])
ax[2].imshow(image[:, :, 2])

plt.show()
