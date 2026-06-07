import math
from pathlib import Path
from dataclasses import dataclass
from ultralytics import YOLO

@dataclass
class BoundingBox:
    """Represents bounding box coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float


class CameraParameters:
    """Encapsulates camera properties and physical geometry calculations."""
    def __init__(self, depth_m: float = 5.0, horizontal_fov_deg: float = 60.0):
        self.depth_m = depth_m
        self.horizontal_fov_deg = horizontal_fov_deg

    def calculate_focal_length(self, img_width: int) -> float:
        """Calculates focal length in pixels using the horizontal FOV."""
        return (img_width / 2.0) / math.tan(math.radians(self.horizontal_fov_deg) / 2.0)

    def pixel_to_meters(self, pixels: float, focal_length: float) -> float:
        """Converts pixel measurements to real-world meters at the assumed depth."""
        return (pixels * self.depth_m) / focal_length


class PalletDetection:
    """Represents a single detected pallet and calculates its geometry."""
    def __init__(self, bbox: BoundingBox, confidence: float, class_id: int, img_width: int, camera: CameraParameters):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.camera = camera

        # Calculate pixel dimensions
        self.height_px = bbox.y2 - bbox.y1
        self.diagonal_width_px = math.sqrt((bbox.x2 - bbox.x1)**2 + (bbox.y2 - bbox.y1)**2)
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
    def __init__(self, model_path: str = "models/whole_pallet_s_640.pt", camera: CameraParameters = None, conf: float = 0.25, imgsz: int = 640):
        self.model = YOLO(model_path)
        self.camera = camera or CameraParameters()
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, image_path: Path) -> list[PalletDetection]:
        """Runs prediction on an image and returns a list of PalletDetections."""
        results = self.model.predict(str(image_path), conf=self.conf, imgsz=self.imgsz, save=False)
        detections = []

        for result in results:
            img_width = result.orig_img.shape[1]
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
        return detections


class BatchProcessor:
    """Manages the process of running detections over multiple files and saving results."""
    def __init__(self, detector: PalletDetector, image_pattern: str = "Image*", output_file: str = "results.txt"):
        self.detector = detector
        self.image_pattern = image_pattern
        self.output_file = output_file

    def get_images(self) -> list[Path]:
        """Globs and sorts images matching the pattern."""
        return sorted(Path("./images").glob(self.image_pattern))

    def process(self):
        """Processes images, prints results, and saves them to the output file."""
        image_paths = self.get_images()

        with open(self.output_file, "w") as f:
            for image_path in image_paths:
                detections = self.detector.detect(image_path)
                for detection in detections:
                    output_line = detection.to_string(image_path.name)
                    print(output_line)
                    f.write(output_line + "\n")


if __name__ == "__main__":
    # Configure parameters
    camera = CameraParameters()
    
    # Initialize services
    detector = PalletDetector(camera=camera)
    processor = BatchProcessor(detector=detector)
    
    # Run prediction pipeline
    processor.process()