import tkinter as tk
from tkinter import messagebox
import numpy as np

from controller import (
    capture_images,
    calibrate_camera,
    stereo_calibrate,
    rectify,
    epipolar_lines,
    detect_matches,
    stereo_matching,
    dense_3d,
    post_processing,
    triangulate_3d_points
)

def start_processing():
    try:
        # Paths for the stereo calibration images from both cameras
        camera1Path = 'Images/cal/*_L.png'
        camera2Path = 'Images/cal/*_R.png'

        # Calibrate both cameras separately
        print("Calibrate first cam")
        K1, dist1, firstImagePoints, objpoints1 = calibrate_camera(camera1Path)
        print("Calibrate second cam")
        K2, dist2, secondImagePoints, _ = calibrate_camera(camera2Path)

        # Perform stereo calibration and compute the fundamental matrix
        grayLU, grayRU, K1, dist1, K2, dist2, R, T, F = stereo_calibrate(K1, K2, dist1, dist2, objpoints1, firstImagePoints, secondImagePoints)

        # Reshape the calibration points for later use
        firstImagePoints = np.array(firstImagePoints, dtype=np.float32).reshape(-1, 2)
        secondImagePoints = np.array(secondImagePoints, dtype=np.float32).reshape(-1, 2)

        # Rectify the stereo images for parallel viewing
        imgRU, imgLU, Q = rectify(K1, dist1, K2, dist2, R, T)

        # Draw epipolar lines on the images for validation
        epipolar_lines(grayLU, imgLU, grayRU, imgRU, F)

        # Detect keypoints and matches between the left and right images
        kpL, kpR = detect_matches(imgLU, imgRU)

        # Perform stereo matching to generate a disparity map
        disp8 = stereo_matching(imgLU, imgRU)

        # Generate the 3D point cloud from the disparity map
        pcd = dense_3d(disp8, imgLU, Q)

        # Post-process the point cloud (e.g., filtering or smoothing)
        post_processing(pcd)

        # Triangulate 3D points from the keypoints
        triangulate_3d_points(K1, K2, R, T, kpL, kpR)

        # Inform the user that the processing is complete
        messagebox.showinfo("Success", "Processing complete!")
        
    except Exception as e:
        # Handle any exceptions that occur during processing
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def capture_images_action():
    # Capture stereo images for calibration (left and right)
    capture_images("L", 0)
    capture_images("R", 1)
    capture_images(0, 0, "left")
    capture_images(0, 1, "right")

    # Inform the user that the images were captured successfully
    messagebox.showinfo("Capture", "Images captured successfully!")

# Set up the main Tkinter window for the application
root = tk.Tk()
root.title("Stereo Vision Calibration")
root.geometry("800x600")  # Window size

# Create buttons for capturing images and starting the processing
capture_button = tk.Button(root, text="Capture", width=20, height=2, command=capture_images_action)
capture_button.place(relx=0.5, rely=0.4, anchor='center')

start_button = tk.Button(root, text="Start", width=20, height=2, command=start_processing)
start_button.place(relx=0.5, rely=0.6, anchor='center')

# Start the Tkinter event loop
root.mainloop()
