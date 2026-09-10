import cv2 as cv
import numpy as np
image= cv.imread("sessionn1/demo.png",0)
cv.imshow("black",image)
cv.waitKey(0)
cv.destroyAllWindows()
image=cv.resize(image,(400,500))
cv.imshow("black",image)
cv.waitKey(0)
cv.destroyAllWindows()
print("height and width",image.shape)
print("number of pixel",image.size)
print(image.dtype)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(image)
print("Maximum pixel intensity: ",max_val)
print("Maximum pixel intensity: ",min_val)


mean_val=cv.mean(image)
print(mean_val)
print("pixel intensity of 50 , 50 loc ", image[50,50])
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# 1. Load the image directly
image = mpimg.imread("sessionn1/demo.png")

# 2. Display the image
plt.imshow(image)
plt.axis("off")  # Optional: Hides the pixel coordinate grid
plt.show()
