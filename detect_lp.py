import cv2
import imutils
import pytesseract
import numpy as np

# Param
from PIL import Image

max_size = 15000
min_size = 100
im = cv2.imread("test.jpg")
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
noise_removal = cv2.bilateralFilter(im_gray,9,75,75)
cv2.imshow('noise remove', noise_removal)
equal_histogram = cv2.equalizeHist(noise_removal)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=20)
sub_morp_image = cv2.subtract(equal_histogram,morph_image)
ret,thresh_image = cv2.threshold(sub_morp_image, 0,255,cv2.THRESH_OTSU)
canny_image = cv2.Canny(thresh_image,250,255)
kernel = np.ones((3,3), np.uint8)
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
cv2.imshow('thresh image', thresh_image)
cv2.imshow('dilated image', dilated_image)

cnts = cv2.findContours(thresh_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

# loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    rect = cv2.boundingRect(approx)
    width = rect[2]
    height = rect[3]
    tmp = abs( width / height)
    if tmp > 1:
        tmp = 1 / tmp
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    cv2.rectangle(im, rect, (0, 255, 0))
    print(tmp, cv2.contourArea(c));
    cv2.imshow('input', im)
    cv2.waitKey(0)
    if len(approx) == 4 and (0.13 < tmp < 0.4 or 0.45 < tmp < 0.8) and cv2.contourArea(c) < 26000:
        screenCnt = rect;
        break

if screenCnt is None:
    detected = 0
    print ("No plate detected")
else:
    detected = 1

if detected == 1:
    images = []
    cropped = im_gray[screenCnt[1]: screenCnt[1] + screenCnt[3], screenCnt[0]: screenCnt[0] + screenCnt[2]]
    cropped2 = thresh_image[screenCnt[1]: screenCnt[1] + screenCnt[3], screenCnt[0]: screenCnt[0] + screenCnt[2]]
    cv2.imshow('License plate', cropped2)
    cnts = cv2.findContours(cropped2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        rect = cv2.boundingRect(c)
        x = rect[2]
        y = rect[3]
        if 0.2 < x / y < 0.6 and x * y < 10000 :
            images.append(rect)
            print(rect)

    imagesSorted = sorted(images, key=lambda x: (x[1] // 20, x[0]));
    i = 1
    result = ''
    for image in imagesSorted:
        charImg = cropped[image[1] - 3 : image[1] + image[3] + 3 , image[0] - 4 : image[0] + image[2] + 4]
        cv2.imshow('char ' + str(i), charImg)
        result += pytesseract.image_to_string(Image.fromarray(charImg), config='--psm 8')
        i +=1
    print('Bien so: ' + str(result))
#
#     # Display image
cv2.imshow('Input image', im)
cv2.waitKey(0)
cv2.destroyAllWindows()



