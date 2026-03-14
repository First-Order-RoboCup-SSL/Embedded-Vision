import cv2
import numpy as np
import glob
import os

# ==========================================
# 1) CALIBRATION SETTINGS
# ==========================================
# IMPORTANT: Update this to match your specific checkerboard!
# Count the INNER intersections where the black and white squares meet.
# (e.g., a standard chessboard is 8x8 squares, but 7x7 inner corners)
CHECKERBOARD = (8, 6) 

# Termination criteria for refining the corner coordinates (sub-pixel accuracy)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points based on the checkerboard dimensions (assumes flat Z=0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

# ==========================================
# 2) LOAD IMAGES (Absolute Path Fix)
# ==========================================
# Dynamically grab the exact folder this python script lives in
current_folder = os.path.dirname(os.path.abspath(__file__))

# Point directly to the folder where image_snapshot.py saved the images
image_dir = os.path.join(current_folder, "calibration_images_219")
image_path_pattern = os.path.join(image_dir, 'calib_frame_*.jpg')

image_files = glob.glob(image_path_pattern)

print(f"Looking for images in: {image_dir}")

if not image_files:
    print("No images found! Please check the folder path and ensure you have captured images.")
    exit()

print(f"Found {len(image_files)} images. Finding checkerboard corners...")

# ==========================================
# 3) EXTRACT CORNERS
# ==========================================
gray = None
success_count = 0

for fname in image_files:
    img = cv2.imread(fname)
    if img is None:
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        success_count += 1
        
        # Refine the corner locations to sub-pixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
        # Resize for display so it doesn't overflow your screen
        display_img = cv2.resize(img, (820, 616))
        cv2.imshow('Finding Corners...', display_img)
        cv2.waitKey(100) # Show each detected frame for 100ms
    else:
        # Just extracting the filename for a cleaner print statement
        base_name = os.path.basename(fname)
        print(f"Could not find a clear {CHECKERBOARD} checkerboard in {base_name}")

cv2.destroyAllWindows()

# ==========================================
# 4) CALCULATE CALIBRATION
# ==========================================
if success_count > 0:
    print(f"\nSuccessfully found corners in {success_count} out of {len(image_files)} images.")
    print("Calculating calibration parameters... this may take a moment.")
    
    # This is the function that actually calculates the math
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("\n--- CALIBRATION SUCCESSFUL ---")
    print(f"Overall RMS re-projection error: {ret:.4f} pixels (Lower is better!)")
    print("\n1. Camera Matrix (mtx) [Intrinsics]:")
    print(repr(mtx))
    print("\n2. Distortion Coefficients (dist):")
    print(repr(dist))
    
    # Save the parameters to a file precisely in the current script's directory
    save_file = os.path.join(current_folder, 'calibration_data.npz')
    np.savez(save_file, mtx=mtx, dist=dist)
    print(f"\nSaved parameters to: {save_file}")
    
else:
    print("\nFailed to find checkerboard corners in any images.")
    print("Make sure your CHECKERBOARD dimensions are set to the INNER corners, not the squares!")