


import numpy as np

from controller import calibrate_camera,stereo_calibrate,rectify,epipolar_lines,detect_matches,triangulate_3d_points,stereo_matching,dense_3d,post_processing

pattern_size=(9, 6)
camera1Path='Images/cal/cal*.png'
camera2Path='Images/cal/cal*.png'
print("Calibrate first cam")
K1, dist1, ret1, firstImagePoints, objpoints1, gray1, h1, w1 = calibrate_camera(camera1Path)
print("Calibrate second cam")
K2, dist2, ret2, secondImagePoints, objpoints2, gray2, h2, w2 = calibrate_camera(camera2Path)

grayLU,grayRU,F,R,T,grayL=stereo_calibrate(K1,K2,dist1,dist2,objpoints1,firstImagePoints,secondImagePoints)

firstImagePoints = np.array(firstImagePoints, dtype=np.float32).reshape(-1, 2)
secondImagePoints = np.array(secondImagePoints, dtype=np.float32).reshape(-1, 2)

imgRU,imgLU=rectify(K1, dist1, K2, dist2, grayL,R, T)

epipolar_lines(grayLU, imgLU, grayRU, imgRU, F)

kpL,kpR=detect_matches(imgLU,imgRU)

triangulate_3d_points(K1, K2, R, T, kpL, kpR)

disp8=stereo_matching(grayLU, grayRU,h1,w1,imgLU)

pcd=dense_3d(imgLU,disp8,h1,w1,K1)

post_processing(pcd)