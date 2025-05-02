import numpy as np
import cv2
import glob
import open3d as o3d
import matplotlib.pyplot as plt

pattern_size=(7, 6)
imgL = cv2.imread('Images/L.png')
imgR = cv2.imread('Images/R.png')

# ====================================================================================
# Capture Image
# ====================================================================================
def capture_images(path, cam, position=""):
    # Initialize the camera feed from the specified camera index (cam)
    cap = cv2.VideoCapture(cam)
    image_count = 0
    takenPic = 37  # Default number of pictures to capture

    # If a specific position is given (e.g., "left" or "right"), override to take only 1 picture
    if position != "":
        takenPic = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if the camera frame is not captured properly

        # Display the camera feed in a window, using a custom title based on path
        if path != 0:
            cv2.imshow(f'Camera {path}', frame)
        else:
            cv2.imshow('Camera', frame)

        key = cv2.waitKey(1) & 0xFF

        # Space key is used to capture and save the image
        if key == ord(' '):
            toSavePath = f"Images/cal/{image_count}_{path}.png"
            image_count += 1

            # If a position is specified (e.g., "L" for left), override the save path
            if position != "":
                toSavePath = f"Images/{position[0].upper()}.png"

            cv2.imwrite(toSavePath, frame)

            # If the desired number of images has been taken, exit the loop
            if image_count == takenPic:
                break

        # Pressing 'q' quits the capture loop
        elif key == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# ====================================================================================
# Calibrate cameras
# ====================================================================================
def calibrate_camera(image_pattern):
    # Prepare the 3D object points based on the known chessboard pattern
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    objpoints = []  # Real world coordinates of the chessboard corners
    imgpoints = []  # Corresponding 2D image coordinates
    gray = None
    h, w = None, None

    # Load and sort all images matching the given pattern (e.g., "calib/*.jpg")
    images = sorted(glob.glob(image_pattern))

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Try to find the chessboard corners in the grayscale image
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            # If found, store the corresponding object and image points
            objpoints.append(objp)
            imgpoints.append(corners)

            # For the first few images, display the detected corners for visual verification
            if i < 5:
                cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(6, 4))
                plt.imshow(img_rgb)
                plt.title(f" Corners (Image {i+1})")
                plt.axis("off")
                plt.show()

    # Perform camera calibration to get the intrinsic matrix (K), distortion coefficients, etc.
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    # Compute the mean reprojection error to evaluate calibration accuracy
    total_error = 0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
        total_error += error
    mean_error = total_error / len(objpoints)

    # Display calibration results
    print(f"\nReprojection Error: {mean_error:.4f}")
    print(f"\nCamera Matrix (K):\n{K}")
    print(f"Distortion Coefficients:\n{dist.ravel()}")

    return K, dist, imgpoints, objpoints


# ====================================================================================
# Stereo Rectification
# ====================================================================================
def stereo_calibrate(K1, K2, dist1, dist2, objpoints1, imgpoints1, imgpoints2):
    # Set calibration flags to control how the stereo calibration behaves
    flags = 0
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS     # Use the initial camera intrinsics provided
    flags |= cv2.CALIB_FIX_FOCAL_LENGTH        # Assume focal length is fixed
    flags |= cv2.CALIB_ZERO_TANGENT_DIST       # Tangential distortion is assumed to be zero

    # Perform stereo calibration to estimate the relative pose (R, T) between two cameras
    _, K1, dist1, K2, dist2, R, T, _, F = cv2.stereoCalibrate(
        objpoints1, imgpoints1, imgpoints2,
        K1, dist1, K2, dist2,
        (imgL.shape[1], imgL.shape[0]),  # Image size
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
        flags=flags
    )

    # Undistort the original stereo images using the calibration results
    undistorted_img1 = cv2.undistort(imgL, K1, dist1, None, K1)
    undistorted_img2 = cv2.undistort(imgR, K2, dist2, None, K2)

    # Convert undistorted images to grayscale for further processing or visualization
    grayLU = cv2.cvtColor(undistorted_img1, cv2.COLOR_BGR2GRAY)
    grayRU = cv2.cvtColor(undistorted_img2, cv2.COLOR_BGR2GRAY)

    # Display the undistorted color images side by side
    combined_image = np.hstack((undistorted_img1, undistorted_img2))
    plt.figure(figsize=(12, 6))
    plt.imshow(combined_image[:, :, ::-1])  # Convert BGR to RGB for display
    plt.title('Undistorted Left + Right Images')
    plt.axis('off')
    plt.show()

    # Show the grayscale undistorted images separately
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(grayLU, cmap='gray')
    plt.title('Undistorted Left (Gray)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(grayRU, cmap='gray')
    plt.title('Undistorted Right (Gray)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Output the extrinsic parameters: rotation and translation between the two cameras
    print('Rotation matrix (R):')
    print(R)
    print('Translation vector (T):')
    print(T)

    return grayLU, grayRU, K1, dist1, K2, dist2, R, T, F


# ====================================================================================
# Perform stereo rectification to compute the rectification transforms
# ====================================================================================
def rectify(K1, dist1, K2, dist2, R, T):
    # Display the dimensions of the stereo images (assumes imgL and imgR are accessible globally)
    print((imgL.shape[1], imgL.shape[0]))
    print((imgR.shape[1], imgR.shape[0]))

    # Stereo rectification: computes rotation (R1, R2) and projection (P1, P2) matrices for both cameras,
    # and Q matrix used for 3D reconstruction
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, dist1, K2, dist2, (imgL.shape[1], imgL.shape[0]), R, T, alpha=-1
    )

    # Print the resulting matrices for debugging and analysis
    print("R1:\n", R1)
    print("R2:\n", R2)
    print("P1:\n", P1)
    print("P2:\n", P2)
    print("Q:\n", Q)

    # Compute the rectification transformation maps for remapping each image
    map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, (imgL.shape[1], imgL.shape[0]), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, dist2, R2, P2, (imgR.shape[1], imgR.shape[0]), cv2.CV_32FC1)

    # Apply the rectification maps to both left and right images
    imgLU = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    imgRU = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

    # Convert rectified images to RGB for visualization
    imgLU_rgb = cv2.cvtColor(imgLU, cv2.COLOR_BGR2RGB)
    imgRU_rgb = cv2.cvtColor(imgRU, cv2.COLOR_BGR2RGB)

    # Display the rectified stereo pair side-by-side
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.hconcat([imgLU_rgb, imgRU_rgb]))
    plt.title('Rectified Images')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Return rectified images and Q matrix used for depth estimation
    return imgRU, imgLU, Q



# ====================================================================================
# Visualizing Epipolar Lines (for Rectified Images)
# ====================================================================================
# Choose a point in the left image
def epipolar_lines(grayLU, imgLU, grayRU, imgRU, F):
    # Select a point in the left grayscale image — using the image center as an example
    ptL = (grayLU.shape[1] // 2, grayLU.shape[0] // 2)

    # Compute the epipolar line in the LEFT image corresponding to the selected point using F
    # Note: the point is from image 1, so pass 1 to computeCorrespondEpilines
    lineL = cv2.computeCorrespondEpilines(np.array([ptL], dtype=np.float32), 1, F)
    lineL = lineL.reshape(-1, 3)  # Each line is of the form ax + by + c = 0

    # Draw the computed epipolar line on a copy of the left image
    epiImageL = imgLU.copy()
    for line in lineL:
        x0, y0 = map(int, [0, -line[2] / line[1]])  # Left edge
        x1, y1 = map(int, [grayLU.shape[1], -(line[2] + line[0] * grayLU.shape[1]) / line[1]])  # Right edge
        cv2.line(epiImageL, (x0, y0), (x1, y1), (0, 255, 0), 1)

    # Compute the epipolar line in the RIGHT image corresponding to the same point
    lineR = cv2.computeCorrespondEpilines(np.array([ptL], dtype=np.float32), 2, F)
    lineR = lineR.reshape(-1, 3)

    # Draw the epipolar line on a copy of the right image
    epiImageR = imgRU.copy()
    for line in lineR:
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [grayRU.shape[1], -(line[2] + line[0] * grayRU.shape[1]) / line[1]])
        cv2.line(epiImageR, (x0, y0), (x1, y1), (0, 255, 0), 1)

    # Convert images to RGB for display
    epiImageL = cv2.cvtColor(epiImageL, cv2.COLOR_BGR2RGB)
    epiImageR = cv2.cvtColor(epiImageR, cv2.COLOR_BGR2RGB)

    # Display both images with their epipolar lines side-by-side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(epiImageL)
    plt.title('Left Rectified Image with Epipolar Line')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(epiImageR)
    plt.title('Right Rectified Image with Epipolar Line')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



# ====================================================================================
# Feature Detection and Matching
# ====================================================================================

def detect_matches(imgLU, imgRU):
    # Step 1: Detect SIFT keypoints and descriptors in both images
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(imgLU, None)
    keypoints2, descriptors2 = sift.detectAndCompute(imgRU, None)

    # Step 2: Match descriptors using the Brute Force matcher with k=2 for ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Step 3: Apply Lowe's ratio test to filter good matches
    good_matches_ratio = []
    pts1_ratio, pts2_ratio = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches_ratio.append(m)
            pts1_ratio.append(keypoints1[m.queryIdx].pt)
            pts2_ratio.append(keypoints2[m.trainIdx].pt)

    # Convert matched points to proper format for findFundamentalMat
    pts1_ratio = np.float32(pts1_ratio).reshape(-1, 1, 2)
    pts2_ratio = np.float32(pts2_ratio).reshape(-1, 1, 2)

    # Step 4: Compute the fundamental matrix using RANSAC to eliminate outliers
    F, mask = cv2.findFundamentalMat(pts1_ratio, pts2_ratio, cv2.FM_RANSAC, 3.0, 0.99)
    mask = mask.ravel().tolist()

    # Step 5: Keep only inlier matches based on the RANSAC mask
    good_matches = [good_matches_ratio[i] for i, m in enumerate(mask) if m]
    pts1 = np.int32([pts1_ratio[i] for i, m in enumerate(mask) if m]).reshape(-1, 2)
    pts2 = np.int32([pts2_ratio[i] for i, m in enumerate(mask) if m]).reshape(-1, 2)

    print("len(good_matches)")
    print(len(good_matches))

    # Step 6: Visualize the final inlier matches
    result_img = cv2.drawMatches(
        imgLU, keypoints1, imgRU, keypoints2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Matches')
    plt.show()

    return pts1, pts2


# ====================================================================================
# Triangulation and 3D Reconstruction
# ====================================================================================
def triangulate_3d_points(K1, K2, R, T, pts1, pts2):
    # Compute projection matrices
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T.reshape(-1, 1)))

    # Ensure points are in float32 and shape (2, N)
    pts1 = np.float32(pts1).T
    pts2 = np.float32(pts2).T

    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

    # Convert from homogeneous to 3D
    points_3d = (points_4d[:3] / points_4d[3]).T

    # Visualize with Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    o3d.visualization.draw_geometries([pcd])


# ====================================================================================
# STEREO MATCHING — Computes the disparity map from stereo images using block matching and filters valid depth range.
# ====================================================================================
def stereo_matching(rectL, rectR):
    # Step 1: Create a StereoSGBM object with specified parameters
    stereo = cv2.StereoSGBM_create(
        minDisparity=20,            # Minimum possible disparity value
        numDisparities=16*12,       # Maximum disparity range (must be divisible by 16)
        blockSize=11                # Size of the matching block
    )

    # Step 2: Compute the disparity map
    # Note: SGBM multiplies disparity by 16 internally. Divide by 16 to get the true disparity values.
    disparityMap = stereo.compute(rectL, rectR).astype(np.float32) / 16

    # Step 3: Normalize disparity values to 8-bit and apply a color map for visualization
    disparityImg = cv2.normalize(
        src=disparityMap, dst=None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
    )
    disparityImg = cv2.applyColorMap(disparityImg, cv2.COLORMAP_JET)

    # Step 4: Display the color-mapped disparity image
    cv2.imshow("Disparity ColorMap", disparityImg)
    while True:
        if cv2.waitKey(0) == 27:  # Exit on pressing ESC
            break
    cv2.destroyAllWindows()
    return disparityMap



# ====================================================================================
# 3D RECONSTRUCTION — Reconstructs 3D point cloud from the disparity map and colors it using the left image.
# ====================================================================================
def dense_3d(disp8, imgLU, Q):
    # Mask valid disparity values for 3D point reconstruction
    mask = disp8 > disp8.min()

    # Reproject disparity map into 3D space based on the transformation matrix
    points_3D = cv2.reprojectImageTo3D(disp8, Q)

    # Extract valid 3D points from the reprojected map using the mask
    points = points_3D[mask]

    # Convert the left image to RGB format for Open3D compatibility
    imgLU_rgb = cv2.cvtColor(imgLU, cv2.COLOR_BGR2RGB)

    # Normalize pixel colors and match them to the corresponding valid 3D points
    colors = imgLU_rgb[mask].astype(np.float32) / 255.0

    # Create a point cloud in Open3D with the valid 3D points and their associated colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud in 3D space
    o3d.visualization.draw_geometries([pcd])
    
    return pcd

# ====================================================================================
# Point Cloud Post-Processing 
# ====================================================================================
def post_processing(pcd): 
    # Step 1: Remove statistical outliers from the point cloud
    # This filters out noise by analyzing the distribution of neighboring points
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=20,  # Number of neighbors to analyze for each point
        std_ratio=2.0     # Threshold based on standard deviation
    )
    pcd_filtered = pcd.select_by_index(ind)  # Keep only inlier points

    # Step 2: Downsample the filtered point cloud using a voxel grid
    # This reduces the number of points, making visualization and processing faster
    pcd_down = pcd_filtered.voxel_down_sample(voxel_size=0.02)

    # Step 3: Visualize the processed point cloud
    o3d.visualization.draw_geometries(
        [pcd_down], 
        window_name="Filtered Point Cloud"
    )