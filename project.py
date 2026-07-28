import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image, ImageOps

PROTOTYPE_DEPTH_M = 5.0
PROTOTYPE_FOV_DEG = 85.0
ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png"}
CONFIDENCE = 0.50
ARUCO_MARKER_SIZE_IN = 7 + (7 / 16)
ARUCO_MARKER_SIZE_M = ARUCO_MARKER_SIZE_IN * 0.0254

# Names match the training skeleton exactly, in the same order used for kpt_shape/flip_idx.
KEYPOINT_NAMES = (
    "front_top_left",
    "front_top_right",
    "front_bottom_right",
    "front_bottom_left",
    "back_top_right",
    "back_bottom_right",
    "back_top_left",
    "back_bottom_left",
)

# Below this, a keypoint is treated as "not really there" rather than a real detection.
# Worth tuning empirically once you have more real-world predictions to look at.
KEYPOINT_VISIBILITY_THRESHOLD = 0.3


class CameraGeometryError(Exception):
    """Base exception for camera geometry calculation failures."""


class FocalLengthError(CameraGeometryError):
    """Raised when focal length cannot be calculated from the given camera parameters."""


class PixelConversionError(CameraGeometryError):
    """Raised when a pixel measurement cannot be converted to real-world units."""


@dataclass
class BoundingBox:
    """Represents bounding box coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Keypoint:
    """A single detected keypoint in pixel space, with the model's confidence in it."""

    x: float
    y: float
    confidence: float

    @property
    def is_visible(self) -> bool:
        return self.confidence >= KEYPOINT_VISIBILITY_THRESHOLD

    def distance_to(self, other: "Keypoint") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class PalletKeypoints:
    """The 8 named corner keypoints for one detected pallet, matching the training skeleton."""

    front_top_left: Keypoint
    front_top_right: Keypoint
    front_bottom_right: Keypoint
    front_bottom_left: Keypoint
    back_top_right: Keypoint
    back_bottom_right: Keypoint
    back_top_left: Keypoint
    back_bottom_left: Keypoint

    @classmethod
    def from_yolo(cls, xy: np.ndarray, conf: np.ndarray) -> "PalletKeypoints":
        """Builds from one instance's Ultralytics keypoint arrays: xy shape (8, 2), conf shape (8,)."""
        points = [
            Keypoint(x=float(x), y=float(y), confidence=float(c))
            for (x, y), c in zip(xy, conf)
        ]
        if len(points) != len(KEYPOINT_NAMES):
            raise ValueError(
                f"Expected {len(KEYPOINT_NAMES)} keypoints, got {len(points)}"
            )
        return cls(*points)


@dataclass
class PixelGeometry:
    """Pure pixel-space measurements derived from a pallet's detected keypoints."""

    front_width_px: float
    height_px: float
    side_depth_px: Optional[float]
    visible_side: Optional[str]  # "right", "left", or None if neither side is visible


def _pair_length(first: Keypoint, second: Keypoint) -> Optional[float]:
    if first.is_visible and second.is_visible:
        return first.distance_to(second)
    return None


def _average_length(candidates: list[Optional[float]]) -> Optional[float]:
    lengths = [candidate for candidate in candidates if candidate is not None]
    if not lengths:
        return None
    return sum(lengths) / len(lengths)


def compute_pixel_geometry(kp: PalletKeypoints) -> PixelGeometry:
    """Computes a pallet's real pixel-space geometry from its 8 detected keypoints.

    front_width_px: distance across the front face's top edge.
    height_px: average of the two front vertical edges (more robust to a single noisy point).
    side_depth_px: distance from the front face back into whichever side is actually
        visible, using its back-top corner. None if neither side's keypoints are
        confidently detected.

    Pure function of `kp` alone, so it's unit-testable without a camera or a YOLO model.
    """
    front_width_px = _average_length(
        [
            _pair_length(kp.front_top_left, kp.front_top_right),
            _pair_length(kp.front_bottom_left, kp.front_bottom_right),
            _pair_length(kp.back_top_left, kp.back_top_right),
            _pair_length(kp.back_bottom_left, kp.back_bottom_right),
        ]
    )
    height_px = _average_length(
        [
            _pair_length(kp.front_top_left, kp.front_bottom_left),
            _pair_length(kp.front_top_right, kp.front_bottom_right),
            _pair_length(kp.back_top_left, kp.back_bottom_left),
            _pair_length(kp.back_top_right, kp.back_bottom_right),
        ]
    )

    right_side_depth_px = _average_length(
        [
            _pair_length(kp.front_top_right, kp.back_top_right),
            _pair_length(kp.front_bottom_right, kp.back_bottom_right),
        ]
    )
    left_side_depth_px = _average_length(
        [
            _pair_length(kp.front_top_left, kp.back_top_left),
            _pair_length(kp.front_bottom_left, kp.back_bottom_left),
        ]
    )

    side_depth_px: Optional[float] = None
    visible_side: Optional[str] = None

    if right_side_depth_px is not None or left_side_depth_px is not None:
        right_count = sum(
            candidate is not None
            for candidate in (
                _pair_length(kp.front_top_right, kp.back_top_right),
                _pair_length(kp.front_bottom_right, kp.back_bottom_right),
            )
        )
        left_count = sum(
            candidate is not None
            for candidate in (
                _pair_length(kp.front_top_left, kp.back_top_left),
                _pair_length(kp.front_bottom_left, kp.back_bottom_left),
            )
        )

        if right_count >= left_count:
            side_depth_px = right_side_depth_px
            visible_side = "right"
        else:
            side_depth_px = left_side_depth_px
            visible_side = "left"

    return PixelGeometry(
        front_width_px=front_width_px,
        height_px=height_px,
        side_depth_px=side_depth_px,
        visible_side=visible_side,
    )


class CameraParameters:
    """Encapsulates camera properties and physical geometry calculations."""

    def __init__(
        self,
        depth_m: float,
        horizontal_fov_deg: float,
        rear_plane_offset_m: float = 0.0,
    ):
        self.depth_m = depth_m
        self.horizontal_fov_deg = horizontal_fov_deg
        self.rear_plane_offset_m = rear_plane_offset_m

    def calculate_focal_length(self, img_width: int) -> float:
        try:
            return (img_width / 2.0) / math.tan(
                math.radians(self.horizontal_fov_deg) / 2.0
            )
        except ZeroDivisionError as e:
            raise FocalLengthError(
                f"Cannot calculate focal length for horizontal_fov_deg={self.horizontal_fov_deg}: {e}"
            ) from e

    def pixel_to_meters(
        self, pixels: float, focal_length: float, depth_m: Optional[float] = None
    ) -> float:
        try:
            effective_depth_m = self.depth_m if depth_m is None else depth_m
            return (pixels * effective_depth_m) / focal_length
        except ZeroDivisionError as e:
            raise PixelConversionError(
                f"Cannot convert pixels to meters with focal_length={focal_length}: {e}"
            ) from e


class PalletDetection:
    """Represents a single detected pallet and its real-world geometry, derived from keypoints."""

    def __init__(
        self,
        bbox: BoundingBox,
        keypoints: PalletKeypoints,
        confidence: float,
        class_id: int,
        focal_length: float,
        camera: CameraParameters,
    ):
        self.bbox = bbox
        self.keypoints = keypoints
        self.confidence = confidence
        self.class_id = class_id
        self.camera = camera
        self.focal_length = focal_length

        geometry = compute_pixel_geometry(keypoints)
        self.front_width_px = geometry.front_width_px
        self.height_px = geometry.height_px
        self.side_depth_px = geometry.side_depth_px
        self.visible_side = geometry.visible_side

        # Convert to real-world dimensions using the frame's precomputed focal length
        self.front_width_m = (
            camera.pixel_to_meters(self.front_width_px, self.focal_length)
            if self.front_width_px is not None
            else None
        )
        self.height_m = (
            camera.pixel_to_meters(self.height_px, self.focal_length)
            if self.height_px is not None
            else None
        )
        self.side_depth_m = (
            camera.pixel_to_meters(
                self.side_depth_px,
                self.focal_length,
                depth_m=camera.depth_m + camera.rear_plane_offset_m,
            )
            if self.side_depth_px is not None
            else None
        )
        self.rear_plane_offset_m = camera.rear_plane_offset_m

    def to_string(self, image_name: str) -> str:
        """Formats the detection into a standard result string."""
        front_width_info = (
            f"front_width={self.front_width_m:.3f}m"
            if self.front_width_m is not None
            else "front_width=unknown"
        )
        height_info = (
            f"height={self.height_m:.3f}m"
            if self.height_m is not None
            else "height=unknown"
        )
        side_info = (
            f"side_depth={self.side_depth_m:.3f}m ({self.visible_side} side visible)"
            if self.side_depth_m is not None
            else "side not visible"
        )
        return (
            f"{image_name}: Pallet detected: conf={self.confidence:.2f}, "
            f"bbox=({self.bbox.x1:.0f},{self.bbox.y1:.0f},{self.bbox.x2:.0f},{self.bbox.y2:.0f}) | "
            f"Real-world (at {self.camera.depth_m}m): "
            f"{front_width_info}, {height_info}, {side_info}"
        )


class PalletDetector:
    """Loads a YOLO pose model and returns structured PalletDetection objects."""

    def __init__(
        self,
        model_path: str = "models/pallet_pose_best.pt",
        camera: CameraParameters = None,
        conf: float = CONFIDENCE,
        imgsz: int = 640,
    ):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Error initializing YOLO model: {e}")
            raise e

        self.camera = camera or CameraParameters(
            depth_m=PROTOTYPE_DEPTH_M, horizontal_fov_deg=PROTOTYPE_FOV_DEG
        )
        self.conf = conf
        self.imgsz = imgsz

    def detect(
        self, image_path: Path, annotation_dir: Optional[Path] = None
    ) -> list[PalletDetection]:
        """Runs prediction on an image and returns a list of PalletDetections."""
        try:
            pil_img = Image.open(image_path)
            pil_img = ImageOps.exif_transpose(pil_img)

            results = self.model.predict(
                pil_img,
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

            if annotation_dir is not None:
                annotated_image = result.plot()
                annotation_path = annotation_dir / f"{image_path.stem}_annotated{image_path.suffix.lower()}"
                cv2.imwrite(str(annotation_path), annotated_image)

            try:
                focal_length = self.camera.calculate_focal_length(img_width)
            except FocalLengthError as e:
                print(f"Skipping {image_path}: {e}")
                continue

            aruco_depth = get_aruco_depth(
                result.orig_img, focal_length, ARUCO_MARKER_SIZE_M
            )

            # Assign to a local variable instead of altering the shared global reference state
            current_depth = (
                aruco_depth if aruco_depth is not None else PROTOTYPE_DEPTH_M
            )

            # Instantiate a unique camera parameters instance specifically for this detection frame
            frame_camera = CameraParameters(
                depth_m=current_depth,
                horizontal_fov_deg=self.camera.horizontal_fov_deg,
                rear_plane_offset_m=self.camera.rear_plane_offset_m,
            )

            if result.keypoints is None:
                print(f"No keypoints returned for {image_path} — is model_path a pose model?")
                continue

            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy()

            try:
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    bbox = BoundingBox(x1, y1, x2, y2)
                    keypoints = PalletKeypoints.from_yolo(
                        keypoints_xy[i], keypoints_conf[i]
                    )

                    detections.append(
                        PalletDetection(
                            bbox=bbox,
                            keypoints=keypoints,
                            confidence=confidence,
                            class_id=cls_id,
                            focal_length=focal_length,  # Reused across all boxes in this frame
                            camera=frame_camera,  # Pass the safe local frame instance
                        )
                    )
            except (AttributeError, TypeError, IndexError, ValueError) as e:
                print(f"Error processing detection results for {image_path}: {e}")
            except PixelConversionError as e:
                print(f"Error converting geometry for {image_path}: {e}")

        return detections


class BatchProcessor:
    """Manages the process of running detections over multiple files and saving results."""

    def __init__(
        self,
        detector: PalletDetector,
        output_file: str = "results.txt",
        annotation_dir: str = "annotated_images",
    ):
        self.detector = detector
        self.output_file = output_file
        self.annotation_dir = Path(annotation_dir)

    def get_images(self) -> list[Path]:
        """Returns a sorted list of image paths matching the allowed extensions (.jpg, .jpeg, .png)."""
        return sorted(
            p for p in Path("images").glob("*") if p.suffix.lower() in ALLOWED_IMG_EXT
        )

    def process(self):
        """Processes images, prints results, and saves them to the output file."""
        image_paths = self.get_images()
        self.annotation_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.output_file, "w") as f:
                for image_path in image_paths:
                    detections = self.detector.detect(
                        image_path, annotation_dir=self.annotation_dir
                    )
                    for detection in detections:
                        output_line = detection.to_string(image_path.name)
                        print(output_line)
                        f.write(output_line + "\n")
            print(f"Results saved to {self.output_file}")
        except OSError as e:
            print(f"Error during batch processing: {e}")
            raise e


def get_aruco_depth(
    img: np.ndarray, focal_length: float, marker_size_m: float
) -> float:
    """Detects a 4x4 ArUco marker in an already-decoded image and returns its depth in meters."""
    if img is None:
        return None

    # Modern OpenCV ArUco setup
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, _ = detector.detectMarkers(img)

    if ids is not None and len(ids) > 0:
        pts = corners[0][0]
        # Calculate the pixel size of the first detected marker
        marker_size_px = float(
            np.mean(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1))
        )

        if marker_size_px > 0:
            # Basic depth math: Z = (Real_Size * Focal_Length) / Pixel_Size
            return (marker_size_m * focal_length) / marker_size_px

    return None


if __name__ == "__main__":
    # Configure camera parameters
    # In the future, these would be set based on actual camera specs and deployment conditions
    camera = CameraParameters(
        depth_m=PROTOTYPE_DEPTH_M,
        horizontal_fov_deg=PROTOTYPE_FOV_DEG,
    )

    # Initialize services
    detector = PalletDetector(camera=camera)
    processor = BatchProcessor(detector=detector)

    # Run prediction pipeline
    processor.process()