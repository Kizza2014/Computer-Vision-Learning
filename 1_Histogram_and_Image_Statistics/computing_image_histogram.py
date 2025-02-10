import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)


# manual
values = img.flatten()
h = [0] * 256
for value in values:
    h[value] += 1

plt.figure(figsize=(10, 4))

plt.subplot(1,2,1)
plt.title(f'Original image ({img.shape[0]} x {img.shape[1]})')
plt.imshow(img, cmap='gray')
plt.axis('off')


plt.subplot(1,2,2)
plt.bar(list(range(256)), h)
plt.title('Image histogram')

plt.savefig('histogram.png')
plt.show()

# using openCV function
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.bar(hist)
plt.show()