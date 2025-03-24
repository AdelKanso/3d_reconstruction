import cv2
import numpy as np

import matplotlib.pyplot as plt
import open3d as o3d
# CAMERA CALIBRATION
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
# Prepare object points (3D world coordinates)
def calibrate_camera(image_path, camera_name):
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)
    
    objpoints = []  # 3D points
    imgpoints = []  # 2D points
    
    # Load multiple calibration images
    for i in range(1, 5):  # Change range to include more images
        img = cv2.imread(f"{image_path}/image{i}.jpg")

        if img is None:
            print(f"[{camera_name}] Could not load image: {i}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, (9, 7), corners, ret)
            plt.matshow(img)
            plt.show()

        print(f"[{camera_name}] Image {i} - Corners {'found' if ret else 'not found'}")

    if len(objpoints) > 0:
        # Calibrate camera
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Compute reprojection error
        reprojection_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            reprojection_error += error
        reprojection_error /= len(objpoints)

        print(f"\n[{camera_name}] Reprojection Error: {reprojection_error}")
        print(f"\n[{camera_name}] Camera Matrix (Intrinsic Parameters):")
        print(f"fx = {K[0, 0]}, fy = {K[1, 1]}")
        print(f"cx = {K[0, 2]}, cy = {K[1, 2]}")
        cv2.destroyAllWindows()
        return K, dist, imgpoints
    else:
        print(f"[{camera_name}] Calibration failed - Not enough valid images.")
        return None, None, None
def stereo_calibrate(K1, dist1, K2, dist2, imgpoints1, imgpoints2, img_size):
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

    # Make sure imgpoints1 and imgpoints2 are lists of lists of points
    if len(imgpoints1) > 0 and len(imgpoints2) > 0:
        imgpoints1 = [np.array(points, dtype=np.float32).reshape(-1, 2) for points in imgpoints1]
        imgpoints2 = [np.array(points, dtype=np.float32).reshape(-1, 2) for points in imgpoints2]
        
        ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
            [objp] * len(imgpoints1),  # Replicate objp for each image pair
            imgpoints1, imgpoints2, 
            K1, dist1, K2, dist2, img_size, 
            None, None
        )
        print("Stereo Calibration successful!")
        print("Rotation Matrix (R):\n", R)
        print("Translation Vector (T):\n", T)
        return R, T
    else:
        print("Stereo Calibration failed - No valid corner matches.")
        return None, None

K1, dist1, imgpoints1 = calibrate_camera("firstCam", "First Camera")
K2, dist2, imgpoints2 = calibrate_camera("secondCam", "Second Camera")

if K1 is not None and K2 is not None:
    print("Calibration successful!")
else:
    print("Calibration failed for one or both cameras.")

# Load the stereo pair of images
img1 = cv2.imread('Images/left.jpg')  # Replace with your actual image path
img2 = cv2.imread('Images/right.jpg')  # Replace with your actual image path

img_size = img1.shape[1], img1.shape[0]  # (width, height)

# Pass the original image points to stereo calibration
R, T = stereo_calibrate(K1, dist1, K2, dist2, imgpoints1, imgpoints2, img_size)

def compute_fundamental_matrix(pts1, pts2):
    # Convert to numpy arrays and reshape to 2D
    pts1 = np.array(pts1, dtype=np.float32).reshape(-1, 2)
    pts2 = np.array(pts2, dtype=np.float32).reshape(-1, 2)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    return F, pts1, pts2

if R is not None and T is not None:
    # Compute the fundamental matrix using the function, using the image points of the current image.
    F, pts1, pts2 = compute_fundamental_matrix(np.array(imgpoints1[0], dtype=np.float32), 
                                                np.array(imgpoints2[0], dtype=np.float32))  # Ensure pts1 and pts2 are numpy arrays
    print("Fundamental Matrix:\n", F)

    # Rectify the stereo images
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, dist1, K2, dist2, img_size, R, T
    )

    # Create undistort and rectify maps
    map1_x, map1_y = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, img_size, cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(K2, dist2, R2, P2, img_size, cv2.CV_32FC1)

    # Apply the rectification maps to the images
    rectified_img1 = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)
    rectified_img2 = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR)

    print("Stereo rectification complete.")
else:
    print("Stereo calibration failed")

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


# Perform feature matching
pts1, pts2, good_matches, keypoints1, keypoints2 = match_features(cv2.cvtColor(rectified_img1, cv2.COLOR_BGR2GRAY),cv2.cvtColor(rectified_img2, cv2.COLOR_BGR2GRAY))

# Visualize the matched features
result_img = cv2.drawMatches(rectified_img1, keypoints1, rectified_img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Disable axis
plt.show()

def compute_essential_matrix(F, K1, K2):
    # Essential matrix: E = K2^T * F * K1
    E = K2.T @ F @ K1
    return E

# Compute essential matrix from the fundamental matrix
E = compute_essential_matrix(F, K1, K2)

# Recover the camera pose (R and T) from the essential matrix
_, R_rec, T_rec, _ = cv2.recoverPose(E, pts1, pts2, K1)

print("Recovered Rotation Matrix (R):\n", R_rec)
print("Recovered Translation Vector (T):\n", T_rec)

def triangulate_3d_points(K1, K2, R, T, pts1, pts2):
    # Create the projection matrices for both cameras
    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))  # First camera projection matrix
    P2 = np.dot(K2, np.hstack((R, T.reshape(-1, 1))))  # Second camera projection matrix

    pts1_float = np.float32(pts1)
    pts2_float = np.float32(pts2)

    print(f"pts1_float dtype: {pts1_float.dtype}, shape: {pts1_float.shape}")
    print(f"pts2_float dtype: {pts2_float.dtype}, shape: {pts2_float.shape}")

    points_3d_homogeneous = cv2.triangulatePoints(P1, P2, pts1_float.T, pts2_float.T)

    # Convert from homogeneous to 3D coordinates
    points_3d = points_3d_homogeneous[:3] / points_3d_homogeneous[3]

    return points_3d.T

# Perform triangulation for the matched points
points_3d = triangulate_3d_points(K1, K2, R_rec, T_rec, pts1, pts2)

# Visualize the sparse point cloud using Open3D
def visualize_point_cloud(points_3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

# Visualize the 3D reconstructed points
visualize_point_cloud(points_3d)

def post_process_point_cloud(points_3d):
    # Convert 3D points to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Statistical outlier removal (remove noise)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    filtered_pcd = pcd.select_by_index(ind)
    print(f"Remaining points after outlier removal: {len(filtered_pcd.points)}")

    # Voxel downsampling (smoothing)
    voxel_size = 0.05  # You can adjust this value based on your data
    downsampled_pcd = filtered_pcd.voxel_down_sample(voxel_size)
    print(f"Remaining points after downsampling: {len(downsampled_pcd.points)}")

    # Return processed point cloud
    return downsampled_pcd

# Apply post-processing to the point cloud
processed_pcd = post_process_point_cloud(points_3d)

# `# o3d.visualization.draw_geometries([processed_pcd])` is a command that visualizes the processed
# point cloud using Open3D library.
# Visualize the processed point cloud
o3d.visualization.draw_geometries([processed_pcd])

def dense_reconstruction(rectified_img1, rectified_img2, Q):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(rectified_img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rectified_img2, cv2.COLOR_BGR2GRAY)

    # Create StereoSGBM matcher (you can tune parameters for better results)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 6,  # Must be divisible by 16
        blockSize=9,
        P1=8 * 3 * 9 ** 2,
        P2=32 * 3 * 9 ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

    # Compute disparity map
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > disparity.min()
    colors = cv2.cvtColor(rectified_img1, cv2.COLOR_BGR2RGB)
    output_points = points_3D[mask]
    output_colors = colors[mask]

    # Optional: downsample for faster plotting
    sample_idx = np.random.choice(len(output_points), size=10000, replace=False)
    output_points_sampled = output_points[sample_idx]
    output_colors_sampled = output_colors[sample_idx] / 255.0

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(output_points_sampled[:, 0], 
            output_points_sampled[:, 1], 
            output_points_sampled[:, 2], 
            c=output_colors_sampled, s=0.5)
    ax.set_title('Dense 3D Point Cloud')
    plt.show()

    return points_3d, colors

# Call the function
dense_points, dense_colors = dense_reconstruction(rectified_img1, rectified_img2, Q)
