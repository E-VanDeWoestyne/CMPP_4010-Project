import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import math

# Assuming your file is named project.py
from project import (
    BoundingBox,
    CameraParameters,
    PalletDetection,
    PalletDetector,
    BatchProcessor,
    CONFIDENCE,
    PROTOTYPE_DEPTH_M,
    ARUCO_MARKER_SIZE_M,
    get_aruco_depth
)

class TestBoundingBox(unittest.TestCase):
    def test_bounding_box_initialization(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=120.0)
        self.assertEqual(bbox.x1, 10.0)
        self.assertEqual(bbox.y1, 20.0)
        self.assertEqual(bbox.x2, 110.0)
        self.assertEqual(bbox.y2, 120.0)


class TestCameraParameters(unittest.TestCase):
    def setUp(self):
        self.camera = CameraParameters(depth_m=5.0, horizontal_fov_deg=60.0)

    def test_calculate_focal_length(self):
        focal_length = self.camera.calculate_focal_length(640)
        expected = 320.0 / math.tan(math.radians(30.0))
        self.assertAlmostEqual(focal_length, expected, places=5)

    def test_pixel_to_meters(self):
        meters = self.camera.pixel_to_meters(100.0, 500.0)
        self.assertEqual(meters, 1.0)


class TestPalletDetection(unittest.TestCase):
    def setUp(self):
        self.camera = CameraParameters(depth_m=5.0, horizontal_fov_deg=60.0)
        self.bbox = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=200.0)
        self.detection = PalletDetection(
            bbox=self.bbox,
            confidence=0.85,
            class_id=0,
            img_width=640,
            camera=self.camera
        )

    def test_pixel_dimensions(self):
        self.assertEqual(self.detection.height_px, 100.0)
        expected_diagonal = math.hypot(100.0, 100.0)
        self.assertAlmostEqual(self.detection.diagonal_width_px, expected_diagonal, places=5)
        self.assertAlmostEqual(self.detection.side_width_px, expected_diagonal / math.sqrt(3.0), places=5)

    def test_real_world_dimensions(self):
        focal_length = self.camera.calculate_focal_length(640)
        expected_side_m = (self.detection.side_width_px * 5.0) / focal_length
        expected_height_m = (100.0 * 5.0) / focal_length
        self.assertAlmostEqual(self.detection.side_width_m, expected_side_m, places=5)
        self.assertAlmostEqual(self.detection.height_m, expected_height_m, places=5)

    def test_to_string(self):
        output = self.detection.to_string("test_img.jpg")
        self.assertIn("test_img.jpg", output)
        self.assertIn("Pallet detected", output)
        self.assertIn("conf=0.85", output)
        self.assertIn("bbox=(100,100,200,200)", output)


class TestPalletDetector(unittest.TestCase):
    @patch("project.get_aruco_depth")
    @patch("project.YOLO")
    def test_detect_parses_results_correctly(self, mock_yolo_cls, mock_get_aruco):
        # Force ArUco to return None to test the default prototype fallback depth
        mock_get_aruco.return_value = None

        mock_yolo_instance = MagicMock()
        mock_yolo_cls.return_value = mock_yolo_instance

        mock_result = MagicMock()
        mock_result.orig_img.shape = (480, 640, 3)

        mock_box = MagicMock()
        mock_box.cls = [0.0]
        mock_box.conf = [0.92]
        mock_box.xyxy = MagicMock()
        mock_box.xyxy.__getitem__.return_value.tolist.return_value = [100.0, 150.0, 200.0, 250.0]
        mock_result.boxes = [mock_box]
        
        mock_yolo_instance.predict.return_value = [mock_result]

        detector = PalletDetector(model_path="dummy.pt")
        detections = detector.detect(Path("dummy.jpg"))

        mock_yolo_instance.predict.assert_called_once_with(
            "dummy.jpg", conf=CONFIDENCE, imgsz=640, save=False, verbose=False
        )
        self.assertEqual(len(detections), 1)
        
        det = detections[0]
        self.assertEqual(det.camera.depth_m, PROTOTYPE_DEPTH_M)
        self.assertEqual(det.confidence, 0.92)
        self.assertEqual(det.bbox.x1, 100.0)


class TestBatchProcessor(unittest.TestCase):
    @patch("project.Path.glob")
    def test_get_images(self, mock_glob):
        mock_glob.return_value = [Path("images/Image4.jpg"), Path("images/Image3.jpg")]
        
        detector = MagicMock()
        processor = BatchProcessor(detector=detector, output_file="results.txt")
        images = processor.get_images()

        self.assertEqual(images, [Path("images/Image3.jpg"), Path("images/Image4.jpg")])

    @patch("project.Path.glob")
    @patch("builtins.open", new_callable=mock_open)
    def test_process_runs_pipeline(self, mock_file, mock_glob):
        mock_glob.return_value = [Path("images/Image4.jpg")]
        
        mock_detector = MagicMock()
        mock_detection = MagicMock()
        mock_detection.to_string.return_value = "formatted_output"
        mock_detector.detect.return_value = [mock_detection]

        processor = BatchProcessor(detector=mock_detector, output_file="test_results.txt")
        processor.process()

        mock_detector.detect.assert_called_once_with(Path("images/Image4.jpg"))
        mock_file.assert_called_once_with("test_results.txt", "w")
        mock_file().write.assert_called_once_with("formatted_output\n")


class TestArUcoDepth(unittest.TestCase):
    @patch("project.cv2.imread")
    def test_get_aruco_depth_file_missing(self, mock_imread):
        mock_imread.return_value = None
        depth = get_aruco_depth(Path("invalid.jpg"), focal_length=500.0, marker_size_m=0.12)
        self.assertIsNone(depth)

    @patch("project.cv2.aruco.ArucoDetector")
    @patch("project.cv2.arcLength")
    @patch("project.cv2.imread")
    def test_get_aruco_depth_success(self, mock_imread, mock_arc_length, mock_detector_cls):
        # Set up mock image layout
        mock_imread.return_value = MagicMock()
        
        # Setup mock detector behaviors
        mock_detector_instance = MagicMock()
        mock_detector_cls.return_value = mock_detector_instance
        
        # Mock successful coordinates returned: (corners, ids, rejected)
        mock_corners = [[["dummy_coord_array"]]]
        mock_ids = [1]
        mock_detector_instance.detectMarkers.return_value = (mock_corners, mock_ids, [])
        
        # Say perimeter is 400px -> marker side is 100px
        mock_arc_length.return_value = 400.0
        
        # Expected depth calculations: (0.10m * 500.0 focal) / 100px = 0.5 meters
        depth = get_aruco_depth(Path("valid.jpg"), focal_length=500.0, marker_size_m=0.10)
        self.assertAlmostEqual(depth, 0.5, places=5)


if __name__ == "__main__":
    unittest.main()