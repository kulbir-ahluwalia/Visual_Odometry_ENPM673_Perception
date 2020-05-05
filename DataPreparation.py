import cv2
import os
from Oxford_dataset.ReadCameraModel import ReadCameraModel
from Oxford_dataset.UndistortImage import UndistortImage

images = []
fx, fy , cx , cy , Gcamera_image , LUT = ReadCameraModel('./model')
for fileName in os.listdir('./stereo/centre'):
    print(fileName)
    img = cv2.imread(os.path.join('./stereo/centre',fileName),0)
    color_image = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    undistorted_image = UndistortImage(color_image, LUT)
    greyScale = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
    greyScale = cv2.equalizeHist(greyScale)
    cv2.imwrite(os.path.join('./data',fileName), greyScale)

print("done")