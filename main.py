import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

#CAMERA CALIBRATION

# Prepare object points (3D world coordinates)
objp = np.zeros((9 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

# Lists to store object points and image points
objpoints = []  # 3D points
imgpoints = []  # 2D points


cap = cv2.VideoCapture(1)  # Change index if needed
image_count = 0  # Counter for captured images

###Take picture
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     cv2.imshow('Camera', frame)
#
#     key = cv2.waitKey(1) & 0xFF
#
#     if key == ord(' '):  # Space bar to capture images
#         image_count += 1
#         filename = f"cal/image{image_count}.jpg"
#         cv2.imwrite(filename, frame)
#         print(f"Saved {filename}")
#
#     if key == ord('d'):  # Stop after capturing 2 images
#         break
#
# cap.release()
# cv2.destroyAllWindows()


for i in range (1,5):
    img = cv2.imread(f"cal/image{i}.jpg")
    # cv2.imshow(" Image", img)
    # cv2.waitKey(2000)

    if img is None:
        print(f"Could not load image: {i}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Gray Image", gray)

    ret, corners = cv2.findChessboardCorners(gray, (9,7), None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,7), corners, ret)
        # cv2.imshow('Checkerboard Detection', img)
        # cv2.waitKey(1000)
        #cv2.imwrite(f"Corners/cal{i}_corners.jpg", img)
        plt.matshow(img)
        plt.show()

    print(f"Image image{i} - Corners {'found' if ret else 'not found'}")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    reprojection_error = 0
    for j in range(len(objpoints)):

        imgpoints2, _ = cv2.projectPoints(objpoints[j], rvecs[j], tvecs[j], K, dist)
        error = cv2.norm(imgpoints[j], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        reprojection_error += error
    reprojection_error /= len(objpoints)
    print("Reprojection error:", reprojection_error)

cv2.destroyAllWindows()

print("Camera Matrix (Intrinsic Parameters):")

f_x = K[0, 0]
f_y = K[1, 1]
c_x = K[0, 2]
c_y = K[1, 2]

print(f"Focal Lengths: fx = {f_x}, fy = {f_y}")
print(f"Optical Center: cx = {c_x}, cy = {c_y}")

if len(rvecs) > 0:
    print("\nExtrinsic Parameters (for first image):")

    # Convert rotation vector to 3x3 rotation matrix
    R, _ = cv2.Rodrigues(rvecs[0])  # Convert rvecs[0] to 3x3 matrix

    print("Rotation Matrix (R):\n", R)  # Now it’s a proper 3x3 matrix
    print("Translation Vector (t):\n", tvecs[0].T)  # Transposed for readability

img = cv2.imread("Images/left2.jpg")

img1 = cv2.imread("Images/right2.jpg")

def undistort_image(image, K, dist):
    h, w = image.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(image, K, dist, None, new_camera_matrix)
    return undistorted

undistorted_img1 = undistort_image(img, K, dist)
undistorted_img2 = undistort_image(img1, K, dist)
combined_image = np.hstack((undistorted_img1,undistorted_img2))
#  cv2.imshow("Undistorted Images Side-by-Side", combined_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.matshow(combined_image)
plt.show

def match_features(image1, image2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            pts1.append(keypoints1[m.queryIdx].pt)
            pts2.append(keypoints2[m.trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2, good_matches,keypoints1, keypoints2
pts1, pts2, good_matches, keypoints1, keypoints2 = match_features(cv2.cvtColor(undistorted_img1, cv2.COLOR_BGR2GRAY),cv2.cvtColor(undistorted_img2, cv2.COLOR_BGR2GRAY))
result_img = cv2.drawMatches(img, keypoints1, img1, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imshow("SIFT Feature Matching", result_img)
# cv2.imwrite("SIFT_Feature_Matching.jpg", result_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(f"Number of good matches: {len(good_matches)}")
plt.matshow(result_img)
plt.show

def compute_fundamental_matrix(pts1, pts2):
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    return F, pts1, pts2
F, pts1, pts2 = compute_fundamental_matrix(pts1, pts2)
# print("Fundemental Matria",F)
#
# def compute_essential_matrix(pts1, pts2, K):
#     E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
#     _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
#     return E, R, t
# E, R, t = compute_essential_matrix(pts1, pts2, K)
# print("Rotation Matrix (R):\n", R)
# print("Translation Vector (t):\n", t.T)
#
# def triangulate_points(pts1, pts2, K, R, t):
#     P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
#     P2 = np.dot(K, np.hstack((R, t)))
#
#     pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
#
#     # Filter valid points
#     #valid_mask = np.isfinite(pts3D).all(axis=0)
#     valid_mask = (np.abs(pts4D[3]) > 1e-6) & (pts4D[2] > -1.0) & np.isfinite(pts4D).all(axis=0)
#
#     #pts3D = pts3D[:, valid_mask]
#     points_3D = (pts4D[:3, valid_mask] / pts4D[3, valid_mask]).T
#     points_3D = points_3D[np.isfinite(points_3D).all(axis=1)]  # Remove any remaining NaNs
#
#      # Keeps only finite (valid) values
#     valid_mask = np.isfinite(points_3D).all(axis=1)
#     points_clean = points_3D[valid_mask]
#
#     return points_clean
# points_3D = triangulate_points(pts1, pts2, K, R, t)
# print("Final 3D Points Shape:", points_3D.shape)
#
# # Draw keypoints on the image
# def draw_points(image, keypoints):
#     output_image = image.copy()
#     colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
#     for i, kp in enumerate(keypoints):
#         x, y = int(kp.pt[0]), int(kp.pt[1])
#         color = colors[i % len(colors)]  # Cycle through colors
#         cv2.circle(output_image, (x, y), 2, color, -1)  # Smaller dots with varied colors
#     return output_image
#
#
#     # Draw keypoints on one of the images
# keypoint_image = draw_points(img1, keypoints1)
# plt.imshow(cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2RGB))
# plt.title("points on Image")
# plt.show()


