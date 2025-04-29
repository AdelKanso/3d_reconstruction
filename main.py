import tkinter as tk
from tkinter import messagebox
import numpy as np
import time

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

camera1Path = 'Images/cal/*_L.png'
camera2Path = 'Images/cal/*_R.png'

def start_processing():
    try:
        camera1Path='Images/cal/*_L.png'
        camera2Path='Images/cal/*_R.png'

        print("Calibrate first cam")
        K1, dist1,firstImagePoints, objpoints1= calibrate_camera(camera1Path)
        print("Calibrate second cam")
        K2, dist2,  secondImagePoints, _ = calibrate_camera(camera2Path)
        grayLU, grayRU, K1, dist1, K2, dist2, R, T, F=stereo_calibrate(K1,K2,dist1,dist2,objpoints1,firstImagePoints,secondImagePoints)

        firstImagePoints = np.array(firstImagePoints, dtype=np.float32).reshape(-1, 2)
        secondImagePoints = np.array(secondImagePoints, dtype=np.float32).reshape(-1, 2)

        imgRU,imgLU,Q=rectify(K1, dist1, K2, dist2,R, T)

        epipolar_lines(grayLU, imgLU, grayRU, imgRU, F)

        kpL,kpR=detect_matches(imgLU,imgRU)

        disp8=stereo_matching(imgLU, imgRU)

        pcd= dense_3d(disp8,imgLU,Q,K1, K2, T)

        post_processing(pcd)
        triangulate_3d_points(K1, K2, R, T, kpL, kpR)
        messagebox.showinfo("Success", "Processing complete!")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def capture_images_action():
    capture_images("L",0)
    capture_images("R",1)
    capture_images(0,0,"left")
    capture_images(0,1,"right")
    messagebox.showinfo("Capture", "Images captured successfully!")

# Setting up the main window
root = tk.Tk()
root.title("Stereo Vision Calibration")
root.geometry("800x600")  # Full screen window

# Create and place the buttons
capture_button = tk.Button(root, text="Capture", width=20, height=2, command=capture_images_action)
capture_button.place(relx=0.5, rely=0.4, anchor='center')

start_button = tk.Button(root, text="Start", width=20, height=2, command=start_processing)
start_button.place(relx=0.5, rely=0.6, anchor='center')

# Start the Tkinter event loop
root.mainloop()
