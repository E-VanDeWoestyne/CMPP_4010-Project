import cv2
import numpy as np

# AI generated ArUco marker generator for forklift detection system
# This script creates a 4x4 ArUco marker with ID 0, suitable for our pallet detection system. The marker will be saved as a PNG file.
def generate_aruco_marker():
    # 1. Select the specific dictionary we discussed
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # 2. Choose the marker ID (0 is standard for a single-marker setup)
    marker_id = 0
    
    # 3. Set the image resolution (400x400 pixels is plenty crisp for printing)
    marker_size_pixels = 400
    
    # 4. Generate the image array
    marker_image = np.zeros((marker_size_pixels, marker_size_pixels), dtype=np.uint8)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels)
    
    # 5. Save the file
    filename = f"forklift_marker_4x4_id{marker_id}.png"
    cv2.imwrite(filename, marker_image)
    
    print(f"Success! Marker saved as '{filename}'.")
    print("Please print this image and ensure the black square measures EXACTLY 15cm x 15cm (or update ARUCO_MARKER_SIZE_M).")

if __name__ == "__main__":
    generate_aruco_marker()