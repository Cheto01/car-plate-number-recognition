"""Vehicle and license plate detection using YOLOv11."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.core.config import get_settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class VehicleDetector:
    """
    Vehicle detector using YOLOv11.
    Detects vehicles (cars, trucks, buses, motorcycles) in images/videos.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,
    ):
        """
        Initialize vehicle detector.

        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on (cuda, cpu, mps)
            classes: List of class IDs to detect (COCO: 2=car, 3=motorcycle, 5=bus, 7=truck)
        """
        self.settings = get_settings()
        self.model_config = self.settings.models.get("vehicle_detector", {})

        # Set parameters (prefer passed args over config)
        self.model_path = model_path or self.model_config.get("path", "yolov8n.pt")
        self.confidence_threshold = confidence_threshold or self.model_config.get(
            "confidence_threshold", 0.5
        )
        self.iou_threshold = iou_threshold or self.model_config.get("iou_threshold", 0.45)
        self.device = device or self.model_config.get("device", "cuda")
        self.classes = classes or self.model_config.get("classes", [2, 3, 5, 7])

        # Validate device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        # Load model
        logger.info(f"Loading vehicle detector: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        logger.info(
            f"Vehicle detector initialized (device={self.device}, "
            f"conf={self.confidence_threshold}, iou={self.iou_threshold})"
        )

    def detect(
        self, image: np.ndarray, return_confidence: bool = True
    ) -> List[Dict[str, any]]:
        """
        Detect vehicles in an image.

        Args:
            image: Input image (BGR format)
            return_confidence: Include confidence scores

        Returns:
            List of detections with format:
            [{'bbox': [x1, y1, x2, y2], 'class': class_id, 'confidence': score}, ...]
        """
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            detection = {
                "bbox": box.xyxy[0].cpu().numpy().tolist(),
                "class": int(box.cls[0]),
                "class_name": results.names[int(box.cls[0])],
            }

            if return_confidence:
                detection["confidence"] = float(box.conf[0])

            detections.append(detection)

        return detections

    def detect_batch(
        self, images: List[np.ndarray], return_confidence: bool = True
    ) -> List[List[Dict[str, any]]]:
        """
        Detect vehicles in a batch of images.

        Args:
            images: List of input images
            return_confidence: Include confidence scores

        Returns:
            List of detection lists for each image
        """
        results = self.model.predict(
            images,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        all_detections = []
        for result in results:
            detections = []
            for box in result.boxes:
                detection = {
                    "bbox": box.xyxy[0].cpu().numpy().tolist(),
                    "class": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                }

                if return_confidence:
                    detection["confidence"] = float(box.conf[0])

                detections.append(detection)

            all_detections.append(detections)

        return all_detections


class LicensePlateDetector:
    """
    License plate detector using custom-trained YOLOv11.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.6,
        iou_threshold: float = 0.4,
        device: Optional[str] = None,
    ):
        """
        Initialize license plate detector.

        Args:
            model_path: Path to custom YOLO model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on
        """
        self.settings = get_settings()
        self.model_config = self.settings.models.get("plate_detector", {})

        self.model_path = model_path or self.model_config.get(
            "path", "data/models/license_plate_detector.pt"
        )
        self.confidence_threshold = confidence_threshold or self.model_config.get(
            "confidence_threshold", 0.6
        )
        self.iou_threshold = iou_threshold or self.model_config.get("iou_threshold", 0.4)
        self.device = device or self.model_config.get("device", "cuda")

        # Validate device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        # Check if model exists
        if not Path(self.model_path).exists():
            logger.warning(
                f"License plate model not found at {self.model_path}. "
                "Please train or download a model."
            )
            self.model = None
        else:
            logger.info(f"Loading license plate detector: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)

            logger.info(
                f"License plate detector initialized (device={self.device}, "
                f"conf={self.confidence_threshold})"
            )

    def detect(
        self, image: np.ndarray, crop_region: Optional[List[int]] = None
    ) -> List[Dict[str, any]]:
        """
        Detect license plates in an image.

        Args:
            image: Input image (BGR format)
            crop_region: Optional [x1, y1, x2, y2] to crop image before detection

        Returns:
            List of detections with format:
            [{'bbox': [x1, y1, x2, y2], 'confidence': score}, ...]
        """
        if self.model is None:
            logger.warning("License plate model not loaded")
            return []

        # Optionally crop image to region of interest (e.g., vehicle bbox)
        if crop_region is not None:
            x1, y1, x2, y2 = map(int, crop_region)
            cropped = image[y1:y2, x1:x2]
        else:
            cropped = image
            x1, y1 = 0, 0

        results = self.model.predict(
            cropped,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            # Adjust bbox coordinates if image was cropped
            bbox = box.xyxy[0].cpu().numpy()
            bbox[0] += x1
            bbox[1] += y1
            bbox[2] += x1
            bbox[3] += y1

            detection = {
                "bbox": bbox.tolist(),
                "confidence": float(box.conf[0]),
            }

            detections.append(detection)

        return detections

    def detect_in_vehicles(
        self, image: np.ndarray, vehicle_bboxes: List[List[int]]
    ) -> List[List[Dict[str, any]]]:
        """
        Detect license plates within vehicle bounding boxes.

        Args:
            image: Full input image
            vehicle_bboxes: List of vehicle bboxes [[x1, y1, x2, y2], ...]

        Returns:
            List of plate detections for each vehicle
        """
        all_detections = []

        for vehicle_bbox in vehicle_bboxes:
            detections = self.detect(image, crop_region=vehicle_bbox)
            all_detections.append(detections)

        return all_detections


def get_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU score (0-1)
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def match_plates_to_vehicles(
    vehicle_detections: List[Dict],
    plate_detections: List[Dict],
    iou_threshold: float = 0.1,
) -> Dict[int, Dict]:
    """
    Match detected license plates to vehicles using spatial proximity.

    Args:
        vehicle_detections: List of vehicle detections
        plate_detections: List of plate detections
        iou_threshold: Minimum IoU to consider a match

    Returns:
        Dictionary mapping vehicle index to best plate detection
    """
    matches = {}

    for v_idx, vehicle in enumerate(vehicle_detections):
        best_plate = None
        best_iou = iou_threshold

        for plate in plate_detections:
            # Check if plate is inside or overlapping with vehicle
            iou = get_iou(vehicle["bbox"], plate["bbox"])

            if iou > best_iou:
                best_iou = iou
                best_plate = plate

        if best_plate is not None:
            matches[v_idx] = best_plate

    return matches
