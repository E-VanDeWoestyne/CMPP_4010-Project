import math
from pathlib import Path
from dataclasses import dataclass
from ultralytics import YOLO
import cv2

PROTOTYPE_DEPTH_M = 5.0
PROTOTYPE_FOV_DEG = 60.0
ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png"}
CONFIDENCE = 0.5
ARUCO_MARKER_SIZE_M = 0.15

@dataclass
class BoundingBox:
	"""Represents bounding box coordinates."""
	x1: float
	y1: float
	x2: float
	y2: float


class CameraParameters:
	"""Encapsulates camera properties and physical geometry calculations."""
	def __init__(self, depth_m: float, horizontal_fov_deg: float):
		self.depth_m = depth_m
		self.horizontal_fov_deg = horizontal_fov_deg

	def calculate_focal_length(self, img_width: int) -> float:
		"""Calculates focal length in pixels using the horizontal FOV."""
		try:
			return (img_width / 2.0) / math.tan(math.radians(self.horizontal_fov_deg) / 2.0)
		except ZeroDivisionError as e:
			print(f"Error calculating focal length: {e}")
			return None

	def pixel_to_meters(self, pixels: float, focal_length: float) -> float:
		"""Converts pixel measurements to real-world meters at the assumed depth."""
		try:
			return (pixels * self.depth_m) / focal_length
		except ZeroDivisionError as e:
			print(f"Error converting pixels to meters: {e}")
			return None


class PalletDetection:
	"""Represents a single detected pallet and calculates its geometry."""
	def __init__(self, bbox: BoundingBox, confidence: float, class_id: int, img_width: int, camera: CameraParameters):
		self.bbox = bbox
		self.confidence = confidence
		self.class_id = class_id
		self.camera = camera

		# Calculate pixel dimensions
		self.height_px = bbox.y2 - bbox.y1
		self.diagonal_width_px = math.hypot(bbox.x2 - bbox.x1, bbox.y2 - bbox.y1) # AI simplified version of sqrt((x2-x1)^2 + (y2-y1)^2)
		self.side_width_px = self.diagonal_width_px / math.sqrt(3)

		# Estimate focal length and real-world dimensions
		self.focal_length = camera.calculate_focal_length(img_width)
		self.side_width_m = camera.pixel_to_meters(self.side_width_px, self.focal_length)
		self.height_m = camera.pixel_to_meters(self.height_px, self.focal_length)

	def to_string(self, image_name: str) -> str:
		"""Formats the detection into a standard result string."""
		return (
			f"{image_name}: Pallet detected: conf={self.confidence:.2f}, "
			f"bbox=({self.bbox.x1:.0f},{self.bbox.y1:.0f},{self.bbox.x2:.0f},{self.bbox.y2:.0f}), "
			f"diagonal={self.diagonal_width_px:.1f}px, side_width={self.side_width_px:.1f}px, height={self.height_px:.1f}px | "
			f"Real-world (at {self.camera.depth_m}m): side_width={self.side_width_m:.3f}m, height={self.height_m:.3f}m"
		)


class PalletDetector:
    """Loads a YOLO model and returns structured PalletDetection objects."""
    def __init__(self, model_path: str = "models/whole_pallet_s_640.pt", camera: CameraParameters = None, conf: float = CONFIDENCE, imgsz: int = 640):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Error initializing YOLO model: {e}")
            raise e

        self.camera = camera or CameraParameters(depth_m=PROTOTYPE_DEPTH_M, horizontal_fov_deg=PROTOTYPE_FOV_DEG)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, image_path: Path) -> list[PalletDetection]:
        """Runs prediction on an image and returns a list of PalletDetections."""
        try:
            results = self.model.predict(str(image_path), conf=self.conf, imgsz=self.imgsz, save=False, verbose=False)
        except Exception as e:
            print(f"Error during detection on {image_path}: {e}")
            return []

        detections = []
        for result in results:
            try:
                img_width = result.orig_img.shape[1]        
                focal_length = self.camera.calculate_focal_length(img_width)   
                aruco_depth = get_aruco_depth(image_path, focal_length, ARUCO_MARKER_SIZE_M)
                
                if aruco_depth is not None:
                    self.camera.depth_m = aruco_depth
                else:
                    self.camera.depth_m = PROTOTYPE_DEPTH_M

                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    bbox = BoundingBox(x1, y1, x2, y2)
                    detections.append(
                        PalletDetection(
                            bbox=bbox,
                            confidence=confidence,
                            class_id=cls_id,
                            img_width=img_width,
                            camera=self.camera
                        )
                    )
            except (AttributeError, TypeError) as e:
                print(f"Error processing detection results for {image_path}: {e}")

        return detections


class BatchProcessor:
	"""Manages the process of running detections over multiple files and saving results."""
	def __init__(self, detector: PalletDetector, output_file: str = "results.txt"):
		self.detector = detector
		self.output_file = output_file

	def get_images(self) -> list[Path]:
		"""Returns a sorted list of image paths matching the allowed extensions (.jpg, .jpeg, .png)."""
		return sorted(
			p for p in Path("images").glob("*")
			if p.suffix.lower() in ALLOWED_IMG_EXT
		)

	def process(self):
		"""Processes images, prints results, and saves them to the output file."""
		image_paths = self.get_images()

		try:
			with open(self.output_file, "w") as f:
				for image_path in image_paths:
					detections = self.detector.detect(image_path)
					for detection in detections:
						output_line = detection.to_string(image_path.name)
						print(output_line)
						f.write(output_line + "\n")
			print(f"Results saved to {self.output_file}")
		except OSError as e:
			print(f"Error during batch processing: {e}")
			raise e

def get_aruco_depth(image_path: Path, focal_length: float, marker_size_m: float) -> float:
    """Detects a 4x4 ArUco marker and returns its depth in meters."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Modern OpenCV ArUco setup
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    
    corners, ids, _ = detector.detectMarkers(img)
    
    if ids is not None and len(ids) > 0:
        # Calculate the pixel size of the first detected marker
        perimeter = cv2.arcLength(corners[0][0], True)
        marker_size_px = perimeter / 4.0
        
        if marker_size_px > 0:
            # Basic depth math: Z = (Real_Size * Focal_Length) / Pixel_Size
            return (marker_size_m * focal_length) / marker_size_px
            
    return None


if __name__ == "__main__":
	# Configure camera parameters
	# In the future, these would be set based on actual camera specs and deployment conditions
	camera = CameraParameters(depth_m=PROTOTYPE_DEPTH_M, horizontal_fov_deg=PROTOTYPE_FOV_DEG)

	# Initialize services
	detector = PalletDetector(camera=camera)
	processor = BatchProcessor(detector=detector)
		
	# Run prediction pipeline
	processor.process()