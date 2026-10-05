
import cv2 as cv
import os
import numpy as np

# -----------------------------------------
# Read the same image
# -----------------------------------------

image_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "demo.png"
)

image = cv.imread(image_path)

if image is None:
    print("Error: demo.png not found!")
    exit()

# Resize image
image = cv.resize(image, (400, 500))


# =========================================
# 1. INCREASE BRIGHTNESS
# =========================================

# Add 50 to every pixel
bright_image = cv.add(image, np.ones(image.shape, dtype=np.uint8) * 50)

cv.imshow("Increased Brightness", bright_image)
cv.waitKey(0)
cv.destroyAllWindows()


# =========================================
# 2. DECREASE BRIGHTNESS
# =========================================

# Subtract 50 from every pixel
dark_image = cv.subtract(image, np.ones(image.shape, dtype=np.uint8) * 50)

cv.imshow("Decreased Brightness", dark_image)
cv.waitKey(0)
cv.destroyAllWindows()


# =========================================
# 3. INCREASE CONTRAST
# =========================================

# Multiply pixel values by 1.5
increased_contrast = cv.convertScaleAbs(
    image,
    alpha=1.5,
    beta=0
)

cv.imshow("Increased Contrast", increased_contrast)
cv.waitKey(0)
cv.destroyAllWindows()


# =========================================
# 4. DECREASE CONTRAST
# =========================================

# Multiply pixel values by 0.5
decreased_contrast = cv.convertScaleAbs(
    image,
    alpha=0.5,
    beta=0
)

cv.imshow("Decreased Contrast", decreased_contrast)
cv.waitKey(0)
cv.destroyAllWindows()


# =========================================
# 5. NEGATIVE IMAGE
# =========================================

# Negative = 255 - original pixel
negative_image = 255 - image

cv.imshow("Negative Image", negative_image)
cv.waitKey(0)
cv.destroyAllWindows()


# =========================================
# 6. BINARY THRESHOLDING
# =========================================

# Convert image to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply binary thresholding
threshold_value = 127

ret, binary_image = cv.threshold(
    gray,
    threshold_value,
    255,
    cv.THRESH_BINARY
)

cv.imshow("Binary Threshold Image", binary_image)
cv.waitKey(0)
cv.destroyAllWindows()


# =========================================
# Print information
# =========================================

print("Image Shape:", image.shape)
print("Image Size:", image.size)
print("Data Type:", image.dtype)
print("Threshold Value:", threshold_value)
