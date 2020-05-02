import cv2
import os
import numpy as np
import random
from csv import writer
import numpy as np
import math
from numpy import linalg as LA
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


imagesList = os.listdir('../Oxford_dataset/data')
fx, fy, cx, cy, Gcamera_image, LUT = ReadCameraModel('../Oxford_dataset/model')
KMatrix = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])

# for index in range(0,len(imagesList)-1):
for index in range(0, 4):
    img1 = cv2.imread(os.path.join('../Oxford_dataset/data',imagesList[index]))
    img2 = cv2.imread(os.path.join('../Oxford_dataset/data',imagesList[index+1]))

    keypts1, keypts2 = generateSIFTKeyPts(img1, img2) #each is a list of best points which match the images (75)

    fundamentalMatrix = RANSAC(keypts1, keypts2)
    # print("\nFundamental Matrix is:\n", fundamentalMatrix, "\n")
    # print("Length of fundamental matrix is: ", len(fundamentalMatrix))
    # print(fundamentalMatrix[1][2])

    #Estimate Essential Matrix from Fundamental Matrix
    # E = K_t.F.K
    #essential_matrix = E
    k = KMatrix
    k_transpose = np.transpose(KMatrix)

    E = np.matmul(k_transpose, np.matmul(fundamentalMatrix,k))
    # print("Essential matrix is: \n", E, "\n")

    # SVD decomposition of the fundamental matrix E
    E__Umatrix, E__Sigma_eigen_value_matrix, E__Vtranspose = svd(E)

    # Replace Sigma matrix with (1,1,0)
    correction_matrix = np.array([[1,0,0],
                                  [0,1,0],
                                  [0,0,0]])
    # print(correction_matrix)
    E_corrected = np.matmul(E__Umatrix, np.matmul(correction_matrix,E__Vtranspose))
    # print("\nCorrected essential matrix is: \n", E_corrected, "\n")

    #calculate pose configurations
    w = np.array([[0,-1,0],
                  [1,0,0],
                  [0,0,1]])

    w_transpose = np.transpose(w)
    # print(w[:,2])
    c1 = E__Umatrix[:,2]
    c2 = -E__Umatrix[:,2]
    c3 = E__Umatrix[:,2]
    c4 = -E__Umatrix[:,2]

    # print("c1 is : ", c1)
    # print(len(c1))
    # print(c1[0])
    # print(c1.flatten())

    r1 = np.matmul(E__Umatrix, np.matmul(w,E__Vtranspose))
    r2 = np.matmul(E__Umatrix, np.matmul(w,E__Vtranspose))
    r3 = np.matmul(E__Umatrix, np.matmul(w_transpose,E__Vtranspose))
    r4 = np.matmul(E__Umatrix, np.matmul(w_transpose,E__Vtranspose))

    # print(r4)
    # print(np.linalg.det(r4))

    #check r, det(r) should be 1
    #If det(R)=−1, the camera pose must be corrected i.e. C=−C and R=−R.
    r1_det = round(np.linalg.det(r1))
    r2_det = round(np.linalg.det(r2))
    r3_det = round(np.linalg.det(r3))
    r4_det = round(np.linalg.det(r4))

    if r1_det == -1:
        c1 = -c1
        r1 = -r1

    if r2_det == -1:
        c2 = -c2
        r2 = -r2

    if r3_det == -1:
        c3 = -c3
        r3 = -r3

    if r4_det == -1:
        c4 = -c4
        r4 = -r4

    # print(r4)
    # print(np.linalg.det(r4))

    # print("r1 is: \n",r1, "\n")
    # print(r1[2][2])

    r1_flat = r1.flatten()
    r2_flat = r2.flatten()
    r3_flat = r3.flatten()
    r4_flat = r4.flatten()

    # print("r1_flat is: \n",r1_flat,"\n")

    config1 = np.concatenate((c1, r1_flat), axis=0)
    config2 = np.concatenate((c2, r2_flat), axis=0)
    config3 = np.concatenate((c3, r3_flat), axis=0)
    config4 = np.concatenate((c4, r4_flat), axis=0)

    # print("camera config1 flat is: (c1,r1) = \n", config1, "\n")

    # print("\nConfiguration 1 is: \n", config1)
    # print("\nConfiguration 2 is: \n", config2)
    # print("\nConfiguration 3 is: \n", config3)
    # print("\nConfiguration 4 is: \n", config4)
    #
    append_rows_in_csv_file('camera_poses.csv',index, config1, config2, config3, config4)
    # append_rows_in_csv_file('camera_poses.csv',index, [config1], [config2], [config3], [config4])

