import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import math

from project import (
    BoundingBox,
    CameraParameters,
    PalletDetection,
    PalletDetector,
    BatchProcessor,
    CONFIDENCE
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
        # focal_length = (width / 2) / tan(fov / 2)
        # For width=640, fov=60: (320) / tan(30 deg) = 320 / (1 / sqrt(3)) = 320 * sqrt(3) ~ 554.25625
        focal_length = self.camera.calculate_focal_length(640)
        expected = 320.0 / math.tan(math.radians(30.0))
        self.assertAlmostEqual(focal_length, expected, places=5)

    def test_pixel_to_meters(self):
        # meters = (pixels * depth) / focal_length
        # (100 px * 5.0 m) / 500 = 1.0 m
        meters = self.camera.pixel_to_meters(100.0, 500.0)
        self.assertEqual(meters, 1.0)


class TestPalletDetection(unittest.TestCase):
    def setUp(self):
        self.camera = CameraParameters(depth_m=5.0, horizontal_fov_deg=60.0)
        self.bbox = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=200.0)
        # width_px = sqrt(100^2 + 100^2) = sqrt(20000) ~ 141.421356
        # height_px = 200 - 100 = 100
        # side_width_px = width_px / sqrt(3) ~ 141.421356 / 1.7320508 ~ 81.649658
        self.detection = PalletDetection(
            bbox=self.bbox,
            confidence=0.85,
            class_id=0,
            img_width=640,
            camera=self.camera
        )

    def test_pixel_dimensions(self):
        self.assertEqual(self.detection.height_px, 100.0)
        expected_diagonal = math.sqrt(100**2 + 100**2)
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
    @patch("project.YOLO")
    def test_detect_parses_results_correctly(self, mock_yolo_cls):
        # Set up mock YOLO instance
        mock_yolo_instance = MagicMock()
        mock_yolo_cls.return_value = mock_yolo_instance

        # Set up mock predict return value
        mock_result = MagicMock()
        mock_result.orig_img.shape = (480, 640, 3)  # height, width, channels

        # Mock boxes
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
            "dummy.jpg", conf=CONFIDENCE, imgsz=640, save=False
        )
        self.assertEqual(len(detections), 1)
        
        det = detections[0]
        self.assertEqual(det.confidence, 0.92)
        self.assertEqual(det.class_id, 0)
        self.assertEqual(det.bbox.x1, 100.0)
        self.assertEqual(det.bbox.y1, 150.0)
        self.assertEqual(det.bbox.x2, 200.0)
        self.assertEqual(det.bbox.y2, 250.0)


class TestBatchProcessor(unittest.TestCase):
    @patch("project.Path.glob")
    def test_get_images(self, mock_glob):
        mock_glob.return_value = [Path("images/Image4.jpg"), Path("images/Image3.jpg")]
        
        detector = MagicMock()
        processor = BatchProcessor(detector=detector, output_file="results.txt")
        images = processor.get_images()

        self.assertEqual(images, [Path("images/Image3.jpg"), Path("images/Image4.jpg")]) # Should be sorted

    @patch("project.Path.glob")
    @patch("builtins.open", new_callable=mock_open)
    def test_process_runs_pipeline(self, mock_file, mock_glob):
        mock_glob.return_value = [Path("images/Image4.jpg")]
        
        # Mock detector and a detection
        mock_detector = MagicMock()
        mock_detection = MagicMock()
        mock_detection.to_string.return_value = "formatted_output"
        mock_detector.detect.return_value = [mock_detection]

        processor = BatchProcessor(detector=mock_detector, output_file="test_results.txt")
        processor.process()

        mock_detector.detect.assert_called_once_with(Path("images/Image4.jpg"))
        mock_detection.to_string.assert_called_once_with("Image4.jpg")
        
        # Verify writing to file
        mock_file.assert_called_once_with("test_results.txt", "w")
        mock_file().write.assert_called_once_with("formatted_output\n")


if __name__ == "__main__":
    unittest.main()
