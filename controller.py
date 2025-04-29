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
def capture_images(path,cam,position=""):
    cap = cv2.VideoCapture(cam)
    image_count = 0
    takenPic=37
    if position != "":
        takenPic=1
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if(path!=0):
            cv2.imshow(f'Camera {path}', frame)
        else:
            cv2.imshow('Camera', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            toSavePath=f"Images/cal/{image_count}_{path}.png"
            image_count += 1
            if position != "":
                toSavePath= f"Images/{position[0].upper()}.png"
            cv2.imwrite(toSavePath, frame)
            if image_count == takenPic:
                break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ====================================================================================
# Calibrate cameras
# ====================================================================================
def calibrate_camera(image_pattern):
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    gray = None
    h, w = None, None

    images = sorted(glob.glob(image_pattern))

    for i, fname in enumerate(images):
        img = cv2.imread(fname)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            if i < 5:
                cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(6, 4))
                plt.imshow(img_rgb)
                plt.title(f" Corners (Image {i+1})")
                plt.axis("off")
                plt.show()
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    # Reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
        total_error += error
    mean_error = total_error / len(objpoints)

    print(f"\nReprojection Error: {mean_error:.4f}")
    print(f"\nCamera Matrix (K):\n{K}")
    print(f"Distortion Coefficients:\n{dist.ravel()}")

    return K, dist, imgpoints, objpoints

# ====================================================================================
# Stereo Rectification
# ====================================================================================
def stereo_calibrate(K1, K2, dist1, dist2, objpoints1, imgpoints1, imgpoints2):
    # Convert images to grayscale
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    flags = 0
    # flags |= cv2.CALIB_FIX_INTRINSIC
    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
    flags |= cv2.CALIB_ZERO_TANGENT_DIST
    # Stereo calibration
    _, K1, dist1, K2, dist2, R, T, _, F = cv2.stereoCalibrate(
        objpoints1, imgpoints1, imgpoints2,
        K1, dist1, K2, dist2, (imgL.shape[1], imgL.shape[0]),
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
        flags=flags
    )

    # Undistort images directly using original intrinsics
    undistorted_img1 = cv2.undistort(imgL, K1, dist1, None, K1)
    undistorted_img2 = cv2.undistort(imgR, K2, dist2, None, K2)

    # Convert to grayscale for display
    grayLU = cv2.cvtColor(undistorted_img1, cv2.COLOR_BGR2GRAY)
    grayRU = cv2.cvtColor(undistorted_img2, cv2.COLOR_BGR2GRAY)

    # Show side-by-side undistorted images
    combined_image = np.hstack((undistorted_img1, undistorted_img2))
    plt.figure(figsize=(12, 6))
    plt.imshow(combined_image[:, :, ::-1])  # Convert BGR to RGB for display
    plt.title('Undistorted Left + Right Images')
    plt.axis('off')
    plt.show()

    # Show grayscale versions separately
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

    # Print extrinsics
    print('Rotation matrix (R):')
    print(R)
    print('Translation vector (T):')
    print(T)

    return grayLU, grayRU, K1, dist1, K2, dist2, R, T, F

# ====================================================================================
# Perform stereo rectification to compute the rectification transforms
# ====================================================================================
def rectify(K1, dist1, K2, dist2, R, T):
    print((imgL.shape[1], imgL.shape[0]))
    print((imgR.shape[1], imgR.shape[0]))
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, dist1, K2, dist2, (imgL.shape[1], imgL.shape[0]), R, T,alpha=-1
    )

    print("R1:\n", R1)
    print("R2:\n", R2)
    print("P1:\n", P1)
    print("P2:\n", P2)
    print("Q:\n", Q)
    # Compute the rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, (imgL.shape[1], imgL.shape[0]), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, dist2, R2, P2,(imgR.shape[1], imgR.shape[0]), cv2.CV_32FC1)
    
    # Rectify the images using the calculated maps
    imgLU = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    imgRU = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

    # Visualize undistorted and rectified stereo images
    imgLU_rgb = cv2.cvtColor(imgLU, cv2.COLOR_BGR2RGB)
    imgRU_rgb = cv2.cvtColor(imgRU, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.hconcat([imgLU_rgb, imgRU_rgb]))
    plt.title('Rectified Images')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return imgRU,imgLU,Q


# ====================================================================================
# Visualizing Epipolar Lines (for Rectified Images)
# ====================================================================================
# Choose a point in the left image
def epipolar_lines(grayLU, imgLU, grayRU, imgRU, F):
    # Choose a point in the left image (we'll use the center for demonstration)
    ptL = (grayLU.shape[1] // 2, grayLU.shape[0] // 2)  # Center point (you can choose another)

    # Compute the corresponding epipolar line in the right image using the fundamental matrix F
    lineL = cv2.computeCorrespondEpilines(np.array([ptL], dtype=np.float32), 1, F)
    lineL = lineL.reshape(-1, 3)

    epiImageL = imgLU.copy()  # Left image copy for drawing lines
    # Draw the epipolar line on the left image
    for line in lineL:
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [grayLU.shape[1], -(line[2] + line[0] * grayLU.shape[1]) / line[1]])
        cv2.line(epiImageL, (x0, y0), (x1, y1), (0, 255, 0), 1)

    # Compute the epipolar line for the right image
    lineR = cv2.computeCorrespondEpilines(np.array([ptL], dtype=np.float32), 2, F)
    lineR = lineR.reshape(-1, 3)

    epiImageR = imgRU.copy()  # Right image copy for drawing lines
    # Draw the epipolar line on the right image
    for line in lineR:
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [grayRU.shape[1], -(line[2] + line[0] * grayRU.shape[1]) / line[1]])
        cv2.line(epiImageR, (x0, y0), (x1, y1), (0, 255, 0), 1)
    epiImageL = cv2.cvtColor(epiImageL, cv2.COLOR_BGR2RGB)
    epiImageR = cv2.cvtColor(epiImageR, cv2.COLOR_BGR2RGB)
    # Display the images with epipolar lines
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
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(imgLU, None)
    keypoints2, descriptors2 = sift.detectAndCompute(imgRU, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches_ratio = []
    pts1_ratio, pts2_ratio = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches_ratio.append(m)
            pts1_ratio.append(keypoints1[m.queryIdx].pt)
            pts2_ratio.append(keypoints2[m.trainIdx].pt)

    pts1_ratio = np.float32(pts1_ratio).reshape(-1, 1, 2)
    pts2_ratio = np.float32(pts2_ratio).reshape(-1, 1, 2)

    # Find the fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(pts1_ratio, pts2_ratio, cv2.FM_RANSAC, 3.0, 0.99)
    mask = mask.ravel().tolist()

    good_matches = [good_matches_ratio[i] for i, m in enumerate(mask) if m]
    pts1 = np.int32([pts1_ratio[i] for i, m in enumerate(mask) if m]).reshape(-1, 2)
    pts2 = np.int32([pts2_ratio[i] for i, m in enumerate(mask) if m]).reshape(-1, 2)
    print("len(good_matches)")
    print(len(good_matches))
    # Visualize the matched features after RANSAC
    result_img = cv2.drawMatches(imgLU, keypoints1, imgRU, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

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
    stereo = cv2.StereoSGBM_create(
            minDisparity=20,
            numDisparities=16*12,
            blockSize=11,
        )
    # In this case, disparity will be multiplied by 16 internally! Divide by 16 to get real value.
    disparityMap = stereo.compute(rectL, rectR).astype(np.float32)/16
    # Normalize and apply a color map
    disparityImg = cv2.normalize(src=disparityMap, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    disparityImg = cv2.applyColorMap(disparityImg, cv2.COLORMAP_JET)
  
    # Plot the disparity map with colormap
    cv2.imshow("Disparity ColorMap", disparityImg)
    while True:
        if cv2.waitKey(0) == 27:
            break
    cv2.destroyAllWindows()
    return disparityMap


# ====================================================================================
# 3D RECONSTRUCTION — Reconstructs 3D point cloud from the disparity map and colors it using the left image.
# ====================================================================================
def compute_Q(K1, K2, T):
    # Extract focal lengths and principal points
    fx  = K1[0, 0]
    fy  = K2[1, 1]  # Assuming fy is the same for both
    cx1 = K1[0, 2]
    cx2 = K2[0, 2]
    cy  = K1[1, 2]
    a1  = K1[0, 1]  # Skew of camera 1
    a2  = K2[0, 1]  # Skew of camera 2

    # Baseline (assumes translation in x only, typical for stereo rigs)
    b = np.linalg.norm(T)

    # Construct Q matrix
    Q = np.eye(4, dtype=np.float64)
    Q[0, 1] = -a1 / fy
    Q[0, 3] = a1 * cy / fy - cx1
    Q[1, 1] = fx / fy
    Q[1, 3] = -cy * fx / fy
    Q[2, 2] = 0
    Q[2, 3] = -fx
    Q[3, 1] = (a2 - a1) / (fy * b)
    Q[3, 2] = 1.0 / b
    Q[3, 3] = ((a1 - a2) * cy + (cx2 - cx1) * fy) / (fy * b)

    return Q
def dense_3d(disp8,imgLU,Q,K1, K2, T):
    Q = compute_Q(K1, K2, T)
    mask = disp8 > disp8.min()

    points_3D=cv2.reprojectImageTo3D(disp8, Q)
    # Get valid 3D points
    points = points_3D[mask]

    # Convert image to RGB (Open3D uses RGB)
    imgLU_rgb = cv2.cvtColor(imgLU, cv2.COLOR_BGR2RGB)

    # Normalize colors and get corresponding colors for valid points
    colors = imgLU_rgb[mask].astype(np.float32) / 255.0

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd])
    return pcd

# ====================================================================================
# Point Cloud Post-Processing 
# ====================================================================================
def post_processing(pcd): 
    # Remove outliers using the point cloud variable you defined (pcd)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_filtered = pcd.select_by_index(ind)

    # Voxel downsampling to reduce point cloud density
    pcd_down = pcd_filtered.voxel_down_sample(voxel_size=0.02)

    # Save or visualize the post-processed point cloud
    o3d.visualization.draw_geometries([pcd_down], window_name="Filtered Point Cloud")



