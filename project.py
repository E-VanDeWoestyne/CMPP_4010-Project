import csv
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROTOTYPE_DEPTH_M = 5.0
PROTOTYPE_FOV_DEG = 60.0
ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png"}
CONFIDENCE = 0.15
ARUCO_MARKER_SIZE_M = 0.1889125
RESULT_HEADERS = [
    "image_name",
    "processed_at",
    "confidence",
    "class_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "diagonal_width_px",
    "side_width_px",
    "height_px",
    "side_width_m",
    "height_m",
]

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)


class CameraGeometryError(Exception):
    """Base exception for camera geometry calculation failures."""


class FocalLengthError(CameraGeometryError):
    """Raised when focal length cannot be calculated from camera parameters."""


class PixelConversionError(CameraGeometryError):
    """Raised when pixel measurements cannot be converted to real-world units."""


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
        try:
            return (img_width / 2.0) / math.tan(
                math.radians(self.horizontal_fov_deg) / 2.0
            )
        except ZeroDivisionError as e:
            raise FocalLengthError(
                f"Cannot calculate focal length for horizontal_fov_deg={self.horizontal_fov_deg}: {e}"
            ) from e

    def pixel_to_meters(self, pixels: float, focal_length: float) -> float:
        try:
            return (pixels * self.depth_m) / focal_length
        except ZeroDivisionError as e:
            raise PixelConversionError(
                f"Cannot convert pixels to meters with focal_length={focal_length}: {e}"
            ) from e


class PalletDetection:
    """Represents a single detected pallet and calculates its geometry."""

    def __init__(
        self,
        bbox: BoundingBox,
        confidence: float,
        class_id: int,
        focal_length: float,
        camera: CameraParameters,
    ):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.camera = camera
        self.focal_length = focal_length

        self.height_px = bbox.y2 - bbox.y1
        self.diagonal_width_px = math.hypot(bbox.x2 - bbox.x1, bbox.y2 - bbox.y1)
        self.side_width_px = self.diagonal_width_px / math.sqrt(3)

        self.side_width_m = camera.pixel_to_meters(
            self.side_width_px, self.focal_length
        )
        self.height_m = camera.pixel_to_meters(self.height_px, self.focal_length)

    def to_string(self, image_name: str) -> str:
        """Formats the detection into a standard result string."""
        return (
            f"{image_name}: Pallet detected: conf={self.confidence:.2f}, "
            f"bbox=({self.bbox.x1:.0f},{self.bbox.y1:.0f},{self.bbox.x2:.0f},{self.bbox.y2:.0f}), "
            f"diagonal={self.diagonal_width_px:.1f}px, side_width={self.side_width_px:.1f}px, height={self.height_px:.1f}px | "
            f"Real-world (at {self.camera.depth_m}m): side_width={self.side_width_m:.3f}m, height={self.height_m:.3f}m"
        )

    def to_csv_row(
        self, image_name: str, processed_at: str
    ) -> dict[str, str | int | float]:
        """Converts the detection into a spreadsheet-friendly dictionary."""
        return {
            "image_name": image_name,
            "processed_at": processed_at,
            "confidence": f"{self.confidence:.6f}",
            "class_id": self.class_id,
            "x1": f"{self.bbox.x1:.2f}",
            "y1": f"{self.bbox.y1:.2f}",
            "x2": f"{self.bbox.x2:.2f}",
            "y2": f"{self.bbox.y2:.2f}",
            "diagonal_width_px": f"{self.diagonal_width_px:.2f}",
            "side_width_px": f"{self.side_width_px:.2f}",
            "height_px": f"{self.height_px:.2f}",
            "side_width_m": f"{self.side_width_m:.6f}",
            "height_m": f"{self.height_m:.6f}",
        }


class PalletDetector:
    """Loads a YOLO model and returns structured PalletDetection objects."""

    def __init__(
        self,
        model_path: str = "models/whole_pallet_s_640.pt",
        camera: CameraParameters = None,
        conf: float = CONFIDENCE,
        imgsz: int = 640,
    ):
        self.model_path = model_path
        self.camera = camera or CameraParameters(
            depth_m=PROTOTYPE_DEPTH_M, horizontal_fov_deg=PROTOTYPE_FOV_DEG
        )
        self.conf = conf
        self.imgsz = imgsz

        # Thread-local storage guarantees each worker thread holds an isolated YOLO instance
        self._thread_local = threading.local()

    def _get_model(self) -> YOLO:
        """Retrieves or creates a thread-local instance of the YOLO model."""
        if not hasattr(self._thread_local, "model"):
            try:
                self._thread_local.model = YOLO(self.model_path)
            except Exception as e:
                print(f"Error initializing YOLO model on thread {threading.get_ident()}: {e}")
                raise e
        return self._thread_local.model

    def detect(self, image_path: Path) -> list[PalletDetection]:
        """Runs prediction on an image and returns a list of PalletDetections."""
        try:
            model = self._get_model()
            results = model.predict(
                str(image_path),
                conf=self.conf,
                iou=0.45,
                imgsz=self.imgsz,
                rect=True,
                save=False,
                verbose=False,
            )
        except Exception as e:
            print(f"Error during detection on {image_path}: {e}")
            return []

        detections = []
        for result in results:
            img_width = result.orig_img.shape[1]

            try:
                focal_length = self.camera.calculate_focal_length(img_width)
            except FocalLengthError as e:
                print(f"Skipping {image_path}: {e}")
                continue

            aruco_depth = get_aruco_depth(
                result.orig_img, focal_length, ARUCO_MARKER_SIZE_M
            )

            current_depth = (
                aruco_depth if aruco_depth is not None else PROTOTYPE_DEPTH_M
            )

            frame_camera = CameraParameters(
                depth_m=current_depth, horizontal_fov_deg=self.camera.horizontal_fov_deg
            )

            try:
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
                            focal_length=focal_length,
                            camera=frame_camera,
                        )
                    )
            except (AttributeError, TypeError) as e:
                print(f"Error processing detection results for {image_path}: {e}")
            except PixelConversionError as e:
                print(f"Error converting geometry for {image_path}: {e}")

        return detections


class BatchProcessor:
    """Manages processing detections across multiple files and saving results."""

    def __init__(
        self,
        detector: PalletDetector,
        output_file: str = "results.csv",
        max_workers: int = None,
    ):
        self.detector = detector
        self.output_file = output_file
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1))
        self.csv_lock = threading.Lock()

    def get_images(self) -> list[Path]:
        """Returns a sorted list of image paths matching allowed extensions."""
        return sorted(
            p for p in Path("images").glob("*") if p.suffix.lower() in ALLOWED_IMG_EXT
        )

    def _process_single_image(self, image_path: Path, writer: csv.DictWriter):
        """Worker task: processes an image and writes output under a lock."""
        detections = self.detector.detect(image_path)
        processed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if detections:
            with self.csv_lock:
                for detection in detections:
                    output_line = detection.to_string(image_path.name)
                    print(output_line)
                    writer.writerow(
                        detection.to_csv_row(image_path.name, processed_at)
                    )

    def process(self):
        """Processes images, prints results, and writes them to a CSV spreadsheet."""
        image_paths = self.get_images()

        try:
            with open(self.output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=RESULT_HEADERS)
                writer.writeheader()

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(
                            self._process_single_image, image_path, writer
                        )
                        for image_path in image_paths
                    ]
                    for future in as_completed(futures):
                        future.result()

            print(f"Results saved to {self.output_file}")
        except OSError as e:
            print(f"Error during batch processing: {e}")
            raise e


def get_aruco_depth(
    img: np.ndarray, focal_length: float, marker_size_m: float
) -> float | None:
    """Detects a 4x4 ArUco marker in an image matrix and returns its depth in meters."""
    if img is None:
        return None

    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(img)

    if ids is not None and len(ids) > 0:
        pts = corners[0][0]
        marker_size_px = float(
            np.mean(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1))
        )

        if marker_size_px > 0:
            return (marker_size_m * focal_length) / marker_size_px

    return None


if __name__ == "__main__":
    camera = CameraParameters(
        depth_m=PROTOTYPE_DEPTH_M, horizontal_fov_deg=PROTOTYPE_FOV_DEG
    )
    detector = PalletDetector(camera=camera)
    processor = BatchProcessor(detector=detector)
    processor.process()