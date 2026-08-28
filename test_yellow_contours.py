import cv2

bin_img = cv2.imread("output/test_yellow_binary.png", cv2.IMREAD_GRAYSCALE)
c, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print("Binary contours:", len(c))

canny_img = cv2.imread("output/test_yellow_canny.png", cv2.IMREAD_GRAYSCALE)
c2, _ = cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print("Canny contours:", len(c2))
