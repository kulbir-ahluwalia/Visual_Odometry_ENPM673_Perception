import cv2
import os
import random
from csv import writer, reader
import numpy as np
import math
from numpy import linalg as LA
from Oxford_dataset.ReadCameraModel import ReadCameraModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getCorrectPose(Rset, Cset, points1, points2, KMatrix):

	maxDepth = 0
	correct = -1
	numPoints = len(points1)

	for i in range(4):
		depth = 0
		points = []
		RtC = -np.dot(Rset[i].T, Cset[i])
		PMatrix2 = np.array([[Rset[i][0][0], Rset[i][0][1], Rset[i][0][2], RtC[0]], \
				[Rset[i][1][0], Rset[i][1][1], Rset[i][1][2], RtC[1]], \
				[Rset[i][2][0], Rset[i][2][1], Rset[i][2][2], RtC[2]]])
		PMatrix2 = np.dot(KMatrix, PMatrix2)
		PMatrix1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])

		for j in range(numPoints):
			A = np.array([points1[j][1]*PMatrix1[2] - PMatrix1[1], PMatrix1[0] - \
			points1[j][0]*PMatrix1[2], points2[j][1]*PMatrix2[2] - PMatrix2[1],\
			PMatrix2[0] - points2[j][0]])

			u, s, vt = np.linalg.svd(A)
			temp = vt[-1]
			temp = temp/temp[-1]
			points.append(temp[:3])

		for p in points:
			r3 = Rset[i][2,:]
			depth += np.matmul(r3, p - Cset[i])

		if depth > maxDepth:
			maxDepth = depth
			correct = i

	points = []
	RtC = -np.dot(Rset[i].T, Cset[correct])
	PMatrix2 = np.array([[Rset[correct][0][0], Rset[correct][0][1], Rset[correct][0][2], RtC[0]], \
				[Rset[correct][1][0], Rset[correct][1][1], Rset[correct][1][2], RtC[1]], \
				[Rset[correct][2][0], Rset[correct][2][1], Rset[correct][2][2], RtC[2]]])
	PMatrix2 = np.dot(KMatrix, PMatrix2)
	for j in range(numPoints):
		A = np.array([points1[j][1]*PMatrix1[2] - PMatrix1[1], PMatrix1[0] - \
			points1[j][0]*PMatrix1[2], points2[j][1]*PMatrix2[2] - PMatrix2[1],\
			PMatrix2[0] - points2[j][0]])

		u, s, vt = np.linalg.svd(A)
		#print(A)
		temp = vt[-1]
		temp = temp/temp[-1]
		points.append(temp[:3])
	
	return Rset[correct], Cset[correct], points


def generateSIFTKeyPts(img1, img2):
	BEST_MATCH_PERCENT = 1
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

def svd(A):
	A_transpose = np.transpose(A)
	intermediate_matrix = np.matmul(A_transpose, A)

	# compute eigen vectors and eigen values
	w, v = LA.eig(intermediate_matrix)
	# print(w) #w is the list for Eigen values
	# print(v) #v is the Eigen vector matrix where columns of V correspond to
	# eigen vectors of (A_transpose*A)

	# Define the matrix V
	V_eigen_vector_matrix = v
	V_transpose = np.transpose(V_eigen_vector_matrix)

	# Make an empty list called eigen_square_root
	eigen_square_root = []

	# take the square root of eigen values
	for i in range(0, len(w)):
		eigen_square_root.append(math.sqrt(abs(w[i])))

	# Generate eigen value matrix
	Sigma_eigen_value_matrix = np.diag(eigen_square_root)

	# Find inverse of eigen value matrix
	Sigma_inverse = np.linalg.inv(Sigma_eigen_value_matrix)

	# Find the U matrix
	U_matrix = np.matmul(A, np.matmul(V_eigen_vector_matrix, Sigma_inverse))

	# # Print all the matrices after SVD decomposition
	# print(f'SVD decomposition of E is:\nU: \n {U_matrix}')
	# print(f'Sigma :\n {Sigma_eigen_value_matrix}')
	# print(f'V transpose :\n {V_transpose}')

	return U_matrix, Sigma_eigen_value_matrix, V_transpose

	# Uncomment the following section to multiply U, sigma and V_tranpose
	# You'll get the original matrix A again which shows SVD is correct
	# SVD_check = np.matmul(U_matrix, np.matmul(Sigma_eigen_value_matrix, V_transpose))
	# print(f'SVD check matrix:\n {SVD_check}')

def append_rows_in_csv_file(file_name, between_photos_index, list_of_elem1, list_of_elem2, list_of_elem3, list_of_elem4):
	# Open file in append mode
	with open(file_name, 'a+', newline='') as write_obj:
		# Create a writer object from csv module
		csv_writer = writer(write_obj)
		# Add contents of list as last row in the csv file
		# csv_writer.writerow(list(between_photos_index))
		write_obj.write('%d \n' % between_photos_index)
		csv_writer.writerow(list_of_elem1)
		csv_writer.writerow(list_of_elem2)
		csv_writer.writerow(list_of_elem3)
		csv_writer.writerow(list_of_elem4)


imagesList = os.listdir('Oxford_dataset/data')
fx, fy, cx, cy, Gcamera_image, LUT = ReadCameraModel('Oxford_dataset/model')
KMatrix = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
PMatrix = np.array([[fx, 0, cx, 0],[0, fy, cy, 0],[0, 0, 1, 0]])

cameraX = [0]
cameraY = [0]
cameraZ = [0]

prevH = np.eye(4)
prevT = 0
origin = np.array([[0, 0, 0, 1]]).T
numPoints = 123

lk_params = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
feature_detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

stepSize = 1

for index in range(0, len(imagesList)-stepSize, stepSize):
	print(index)
	img1 = cv2.imread(os.path.join('Oxford_dataset/data',imagesList[index]))
	#img1 = np.array(img1, np.uint8)
	#print(img1.shape)
	#img1 = cv2.equalizeHist(img1)
	img2 = cv2.imread(os.path.join('Oxford_dataset/data',imagesList[index+stepSize]))
	#img1 = np.array(img1, np.uint8)
	#img2 = cv2.equalizeHist(img2)
	cv2.imshow('', img1)
	#keypts1, keypts2 = generateSIFTKeyPts(img1, img2) #each is a list of best points which match the images (75)

	temp = []
	prev_keypoint = feature_detector.detect(img1, None)
	for i in range(len(prev_keypoint)):
		temp.append([prev_keypoint[i].pt[0], prev_keypoint[i].pt[1]])
	keypts1 = np.array(temp, np.float32)
	keypts2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, keypts1, None, **lk_params)
	st = st.reshape((st.shape[0]))
	keypts1 = keypts1[st>0]
	keypts2 = keypts2[st>0]

	#################################
	# Code for generating csv files #

	# fundamentalMatrix = RANSAC(keypts1, keypts2)
	# # print("\nFundamental Matrix is:\n", fundamentalMatrix, "\n")
	# # print("Length of fundamental matrix is: ", len(fundamentalMatrix))
	# # print(fundamentalMatrix[1][2])

	# #Estimate Essential Matrix from Fundamental Matrix
	# # E = K_t.F.K
	# #essential_matrix = E
	# k = KMatrix
	# k_transpose = np.transpose(KMatrix)

	E, mask = cv2.findEssentialMat(keypts1, keypts2, KMatrix, cv2.RANSAC, 0.999, 1, None)
	mask = mask.reshape(mask.shape[0])
	keypts1 = keypts1[mask>0]
	keypts2 = keypts2[mask>0]

	# #E = np.matmul(k_transpose, np.matmul(fundamentalMatrix,k))
	# # print("Essential matrix is: \n", E, "\n")

	# if not (np.linalg.det(E) == 0):
	#     # SVD decomposition of the fundamental matrix E
	#     # E__Umatrix, E__Sigma_eigen_value_matrix, E__Vtranspose = np.linalg.svd(E)

	#     # # Replace Sigma matrix with (1,1,0)
	#     # correction_matrix = np.array([[1,0,0],
	#     #                               [0,1,0],
	#     #                               [0,0,0]])
	#     # # print(correction_matrix)
	#     # E = np.matmul(E__Umatrix, np.matmul(correction_matrix,E__Vtranspose))
	#     # print("\nCorrected essential matrix is: \n", E_corrected, "\n")

	#     calculate pose configurations
	#     w = np.array([[0,-1,0],
	#                   [1,0,0],
	#                   [0,0,1]])

	#     w_transpose = np.transpose(w)
	#     # print(w[:,2])
	#     c1 = E__Umatrix[:,2]
	#     c2 = -E__Umatrix[:,2]
	#     c3 = E__Umatrix[:,2]
	#     c4 = -E__Umatrix[:,2]

	#     # print("c1 is : ", c1)
	#     # print(len(c1))
	#     # print(c1[0])
	#     # print(c1.flatten())

	#     r1 = np.matmul(E__Umatrix, np.matmul(w,E__Vtranspose))
	#     r2 = np.matmul(E__Umatrix, np.matmul(w,E__Vtranspose))
	#     r3 = np.matmul(E__Umatrix, np.matmul(w_transpose,E__Vtranspose))
	#     r4 = np.matmul(E__Umatrix, np.matmul(w_transpose,E__Vtranspose))

	#     # print(r4)
	#     # print(np.linalg.det(r4))

	#     #check r, det(r) should be 1
	#     #If det(R)=−1, the camera pose must be corrected i.e. C=−C and R=−R.
	#     r1_det = round(np.linalg.det(r1))
	#     r2_det = round(np.linalg.det(r2))
	#     r3_det = round(np.linalg.det(r3))
	#     r4_det = round(np.linalg.det(r4))

	#     if r1_det == -1:
	#         c1 = -c1
	#         r1 = -r1

	#     if r2_det == -1:
	#         c2 = -c2
	#         r2 = -r2

	#     if r3_det == -1:
	#         c3 = -c3
	#         r3 = -r3

	#     if r4_det == -1:
	#         c4 = -c4
	#         r4 = -r4

	#     # print(r4)
	#     # print(np.linalg.det(r4))

	#     # print("r1 is: \n",r1, "\n")
	#     # print(r1[2][2])

	#     r1_flat = r1.flatten()
	#     r2_flat = r2.flatten()
	#     r3_flat = r3.flatten()
	#     r4_flat = r4.flatten()

	#     # print("r1_flat is: \n",r1_flat,"\n")

	#     config1 = np.concatenate((c1, r1_flat), axis=0)
	#     config2 = np.concatenate((c2, r2_flat), axis=0)
	#     config3 = np.concatenate((c3, r3_flat), axis=0)
	#     config4 = np.concatenate((c4, r4_flat), axis=0)
##
	    # print("camera config1 flat is: (c1,r1) = \n", config1, "\n")

	    # print("\nConfiguration 1 is: \n", config1)
	    # print("\nConfiguration 2 is: \n", config2)
	    # print("\nConfiguration 3 is: \n", config3)
	    # print("\nConfiguration 4 is: \n", config4)
	    #
	#    append_rows_in_csv_file('camera_poses.csv',index, config1, config2, config3, config4)
	    # append_rows_in_csv_file('camera_poses.csv',index, [config1], [config2], [config3], [config4])

	# Code for generating csv files #
	#################################

	
	#Rcorr, tcorr, points3D = getCorrectPose(Rset, Cset, keypts1, keypts2, KMatrix)
	ret, Rcorr, tcorr, mask = cv2.recoverPose(E, keypts1, keypts2)
	if abs(prevT - tcorr[2][0]) > 1:
		tcorr[2][0] = -tcorr[2][0]
	prevT = tcorr[2][0]

	points3D = []
	PMatrix2 = np.array([[Rcorr[0][0], Rcorr[0][1], Rcorr[0][2], tcorr[0][0]], \
				[Rcorr[1][0], Rcorr[1][1], Rcorr[1][2], tcorr[1][0]], \
				[Rcorr[2][0], Rcorr[2][1], Rcorr[2][2], tcorr[2][0]]])
	PMatrix2 = np.dot(KMatrix, PMatrix2)
	PMatrix1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
	for j in range(len(keypts1)):
		A = np.array([keypts1[j][1]*PMatrix1[2] - PMatrix1[1], PMatrix1[0] - \
			keypts1[j][0]*PMatrix1[2], keypts2[j][1]*PMatrix2[2] - PMatrix2[1],\
			PMatrix2[0] - keypts2[j][0]])

		u, s, vt = np.linalg.svd(A)
		temp = vt[-1]
		temp = temp/temp[-1]
		points3D.append(temp[:3])

	Cset = []
	Rset = []

	# with open('camera_poses.csv', mode='r') as file:
	# 	csvReader = reader(file, delimiter=',')
	# 	lineNum = -1
		
	# 	for line in csvReader:
	# 		lineNum += 1
	# 		if (lineNum >= 5*index + 1) and (lineNum < 5*(index+1)):
	# 			Cset.append(np.array(line[:3], np.float32))
	# 			Rset.append(np.array(line[3:], np.float32).reshape((3,3)))

	
	#Rcorr, Ccorr, points3D = getCorrectPose(Rset, Cset, keypts1, keypts2, KMatrix)
	A = np.zeros((2*len(keypts2), 12))

	# for i in range(len(keypts2)):
	# 	normalized = np.dot(np.linalg.inv(KMatrix), np.array([keypts1[i][0], keypts1[i][1], 1]))
	# 	#print(points3D)
	# 	A[2*i][0] = points3D[i][0]
	# 	A[2*i][1] = points3D[i][1]
	# 	A[2*i][2] = points3D[i][2]
	# 	A[2*i][6] = -points3D[i][0]*normalized[0]
	# 	A[2*i][7] = -points3D[i][1]*normalized[0]
	# 	A[2*i][8] = -points3D[i][2]*normalized[0]
	# 	A[2*i][9] = 1
	# 	A[2*i][11] = -normalized[0]

	# 	A[2*i+1][3] = points3D[i][0]
	# 	A[2*i+1][4] = points3D[i][1]
	# 	A[2*i+1][5] = points3D[i][2]
	# 	A[2*i+1][6] = -points3D[i][0]*normalized[1]
	# 	A[2*i+1][7] = -points3D[i][1]*normalized[1]
	# 	A[2*i+1][8] = -points3D[i][2]*normalized[1]
	# 	A[2*i+1][10] = 1
	# 	A[2*i+1][11] = -normalized[1]

	# u, s, vt = np.linalg.svd(A)
	# pose = vt[-1]
	# trans = np.array([pose[9:]]).T
	# rot = np.array(pose[:9]).reshape((3,3))
	# ru, rs, rvt = np.linalg.svd(rot)
	# rot = np.dot(ru, rvt)
	newH = np.vstack((np.hstack((Rcorr, tcorr)), np.array([0,0,0,1])))
	print(tcorr)
	prevH = np.matmul(prevH, newH)
	pos = np.matmul(prevH, origin)
	#print(ru)


	with open('cameraPositions.csv', mode='a+', newline='') as file:
		csv_writer = writer(file)
		csv_writer.writerow(pos)
	cameraX.append(pos[0])
	cameraY.append(pos[1])
	cameraZ.append(pos[2])

	plt.plot(pos[0], pos[2], '-ro')
	plt.pause(0.0001)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax = fig.add_subplot(111)
#ax.scatter(cameraX, cameraZ)
#plt.pause(0.1)
print("Done")
plt.show()