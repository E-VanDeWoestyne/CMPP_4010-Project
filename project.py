from pathlib import Path
import math

from ultralytics import YOLO

model = YOLO("whole_pallet_s_640.pt")

image_paths = sorted(Path(".").glob("Image4.jpg"))

# Assumed camera parameters
ASSUMED_DEPTH_M = 5.0  # pallet distance in meters
HORIZONTAL_FOV_DEG = 60.0  # horizontal field of view in degrees

# Open output file for writing (will be overwritten each run)
with open("results.txt", "w") as f:
    for image_path in image_paths:
        results = model.predict(str(image_path), conf=0.25, save=False)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Calculate pallet dimensions in pixels
                height_px = y2 - y1
                width_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # diagonal width
                side_width_px = width_px / math.sqrt(3)  # side width of flat face
                
                # Estimate focal length from horizontal FOV
                img_width = result.orig_img.shape[1]
                focal_length = (img_width / 2.0) / math.tan(math.radians(HORIZONTAL_FOV_DEG) / 2.0)
                
                # Convert pixel dimensions to meters
                side_width_m = (side_width_px * ASSUMED_DEPTH_M) / focal_length
                height_m = (height_px * ASSUMED_DEPTH_M) / focal_length
                
                output = f"{image_path.name}: Pallet detected: conf={confidence:.2f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}), diagonal={width_px:.1f}px, side_width={side_width_px:.1f}px, height={height_px:.1f}px | Real-world (at {ASSUMED_DEPTH_M}m): side_width={side_width_m:.3f}m, height={height_m:.3f}m"
                print(output)
                f.write(output + "\n")