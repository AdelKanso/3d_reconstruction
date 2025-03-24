import cv2
import numpy as np
import glob

# Chessboard size
CHESSBOARD_SIZE = (9, 7)  

# Prepare object points
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  
imgpoints1 = []  
imgpoints2 = []  

# Load images
images1 = glob.glob("firstCam/image*.jpg")  
images2 = glob.glob("secondCam/image*.jpg")  

for fname1, fname2 in zip(images1, images2):
    img1 = cv2.imread(fname1)
    img2 = cv2.imread(fname2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ret1, corners1 = cv2.findChessboardCorners(gray1, CHESSBOARD_SIZE, None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, CHESSBOARD_SIZE, None)

    if ret1 and ret2:
        objpoints.append(objp)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)

# **Camera Calibration for First Camera**
ret1, M1, d1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, gray1.shape[::-1], None, None)
ret2, M2, d2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1], None, None)

print("\nCamera 1 Matrix (Intrinsic Parameters):\n", M1)
print("Camera 1 Distortion Coefficients:\n", d1)
print("Camera 1 Reprojection Error:", ret1)

print("\nCamera 2 Matrix (Intrinsic Parameters):\n", M2)
print("Camera 2 Distortion Coefficients:\n", d2)
print("Camera 2 Reprojection Error:", ret2)

# **Stereo Calibration**
_, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2, M1, d1, M2, d2, gray1.shape[::-1], None, None, None, None,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
)

print("\nRotation Matrix (R):\n", R)
print("\nTranslation Vector (T):\n", T)

# **Rectification**
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(M1, d1, M2, d2, gray1.shape[::-1], R, T)

# **Undistort and rectify images**
map1x, map1y = cv2.initUndistortRectifyMap(M1, d1, R1, P1, gray1.shape[::-1], cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(M2, d2, R2, P2, gray2.shape[::-1], cv2.CV_32FC1)

rectified1 = cv2.remap(gray1, map1x, map1y, cv2.INTER_LINEAR)
rectified2 = cv2.remap(gray2, map2x, map2y, cv2.INTER_LINEAR)

cv2.imshow("Rectified Image 1", rectified1)
cv2.imshow("Rectified Image 2", rectified2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# **Feature Matching using ORB + BFMatcher**
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(rectified1, None)
kp2, des2 = orb.detectAndCompute(rectified2, None)

# BFMatcher (Hamming for ORB)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance (lower = better)
matches = sorted(matches, key=lambda x: x.distance)

# Draw top 50 matches
img_matches = cv2.drawMatches(rectified1, kp1, rectified2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the matched features
cv2.imshow("Feature Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# **Triangulation**
points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

points4D = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
points3D = points4D[:3] / points4D[3]

print("\nTriangulated 3D Points:")
print(points3D.T)

# **Post Processing: Remove Outliers (Optional - Basic Approach)**
mean_z = np.mean(points3D[2])
std_z = np.std(points3D[2])
filtered_points3D = points3D[:, np.abs(points3D[2] - mean_z) < 2 * std_z]

print("\nFiltered 3D Points After Outlier Removal:")
print(filtered_points3D.T)
