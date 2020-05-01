import cv2
import os
import numpy as np
import random

from Oxford_dataset.ReadCameraModel import ReadCameraModel


def generateSIFTKeyPts(img1, img2):
    BEST_MATCH_PERCENT = 0.15
    sift = cv2.ORB_create(500) #Generates 500 Max features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    keypts1, dscptr1 = sift.detectAndCompute(img1, None)
    keypts2, dscptr2 = sift.detectAndCompute(img2, None)

    matches = matcher.match(dscptr1, dscptr2, None)

    matches.sort(key=lambda x: x.distance, reverse=False) #sort the matches based on score
    bestPoints = int(len(matches) * BEST_MATCH_PERCENT)
    matches = matches[:bestPoints]

    #imMatches = cv2.drawMatches(img1, keypts1, img2, keypts2, matches, None)
    #cv2.imwrite("matches.jpg", imMatches)

    SIFTpts1 = np.zeros((len(matches), 2), dtype=np.float32)
    SIFTpts2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        SIFTpts1[i, :] = keypts1[match.queryIdx].pt
        SIFTpts2[i, :] = keypts2[match.trainIdx].pt

    return SIFTpts1, SIFTpts2


def EstimateFundmentalMatrix(pts1, pts2):
    #applying the SVD concept
    A = np.zeros((8, 9), dtype=np.float32)
    for i in range(0, 8):
        A[i][0] = pts1[i][0]*pts2[i][0]
        A[i][1] = pts1[i][0]*pts2[i][1]
        A[i][2] = pts1[i][0]
        A[i][3] = pts1[i][1]*pts2[i][0]
        A[i][4] = pts1[i][1]*pts2[i][1]
        A[i][5] = pts1[i][1]
        A[i][6] = pts2[i][0]
        A[i][7] = pts2[i][1]
        A[i][8] = 1

    U,S,V = np.linalg.svd(A)
    f = V[-1].reshape((3,3))
    U1, S1, V1 = np.linalg.svd(f)
    S2 = np.array([[S1[0], 0, 0], [0, S1[1], 0], [0, 0, S1[2]]])

    F = np.dot(U1,np.dot(S2, V1))
    return F



def RANSAC(kp1, kp2):
    maximumInliers = 0
    maxF = np.zeros((3,3))
    for i in range(0,10): #can make it 50
        randomList = random.sample(range(0, len(kp1)), 8)
        rndpts1 = []
        rndpts2 = []
        for j in randomList:
            rndpts1.append(kp1[j])
            rndpts2.append(kp2[j])
        F = EstimateFundmentalMatrix(rndpts1, rndpts2)
        count = 0
        for k in range(0, len(kp1)):
            X1 = np.array([kp1[k][0], kp1[k][1], 1])
            X2 = np.transpose(np.array([kp2[k][0], kp1[k][1], 1]))
            if abs(np.dot(X2, np.dot(F, X1))) < 0.01:
                count += 1

        # code to find the inliers for the points
        if count > maximumInliers:
            maximumInliers = count
            maxF = F

    return maxF


imagesList = os.listdir('Oxford_dataset/data')
fx, fy, cx, cy, Gcamera_image, LUT = ReadCameraModel('Oxford_dataset/model')
KMatrix = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])

for index in range(0,1):#len(imagesList)-2):
    img1 = cv2.imread(os.path.join('Oxford_dataset/data',imagesList[index]))
    img2 = cv2.imread(os.path.join('Oxford_dataset/data',imagesList[index+1]))

    keypts1, keypts2 = generateSIFTKeyPts(img1, img2) #each is a list of best points which match the images (75)

    fundamentalMatrix = RANSAC(keypts1, keypts2)
    print("fundamental Matrix is", fundamentalMatrix)




