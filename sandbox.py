import cv2

treshold = 100
segment_size = (100, 100)
kernel_size = (3, 3)
img = cv2.imread("./medium_image.png", cv2.IMREAD_GRAYSCALE)
blured = cv2.GaussianBlur(img, kernel_size, 0)
h, w = blured.shape
cv2.imshow("Image", img)
print(img)
print(img.shape)
cv2.waitKey(0)

cv2.destroyAllWindows()