"""Main ALPR pipeline for processing videos."""
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.config import get_settings
from src.core.database import (
    Detection,
    LicensePlate,
    TrafficStatistics,
    Vehicle,
    VideoSource,
    get_db_manager,
)
from src.core.logging_config import get_logger
from src.models.detector import LicensePlateDetector, VehicleDetector, match_plates_to_vehicles
from src.models.ocr import LicensePlateOCR
from src.models.tracker import TrackInterpolator, VehicleTracker

logger = get_logger(__name__)


class ALPRPipeline:
    """
    End-to-end Automatic License Plate Recognition pipeline.
    Processes videos to detect vehicles, track them, recognize license plates,
    and store results in database.
    """

    def __init__(
        self,
        vehicle_detector: Optional[VehicleDetector] = None,
        plate_detector: Optional[LicensePlateDetector] = None,
        tracker: Optional[VehicleTracker] = None,
        ocr: Optional[LicensePlateOCR] = None,
        interpolator: Optional[TrackInterpolator] = None,
    ):
        """
        Initialize ALPR pipeline.

        Args:
            vehicle_detector: Vehicle detector instance
            plate_detector: License plate detector instance
            tracker: Vehicle tracker instance
            ocr: OCR instance
            interpolator: Track interpolator instance
        """
        self.settings = get_settings()
        self.db_manager = get_db_manager()

        # Initialize components
        self.vehicle_detector = vehicle_detector or VehicleDetector()
        self.plate_detector = plate_detector or LicensePlateDetector()
        self.tracker = tracker or VehicleTracker()
        self.ocr = ocr or LicensePlateOCR()
        self.interpolator = interpolator or TrackInterpolator()

        # Pipeline configuration
        self.pipeline_config = self.settings.pipeline

        # Statistics
        self.stats = {
            "frames_processed": 0,
            "vehicles_detected": 0,
            "plates_detected": 0,
            "plates_recognized": 0,
            "processing_time": 0,
        }

        logger.info("ALPR Pipeline initialized")

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        save_to_db: bool = True,
        save_to_csv: bool = False,
        show_progress: bool = True,
    ) -> Dict:
        """
        Process a video file.

        Args:
            video_path: Path to input video
            output_path: Path to output video (with visualizations)
            save_to_db: Save results to database
            save_to_csv: Save results to CSV
            show_progress: Show progress bar

        Returns:
            Dictionary with processing results and statistics
        """
        logger.info(f"Processing video: {video_path}")
        start_time = time.time()

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return {"error": "Failed to open video"}

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        logger.info(
            f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames, {duration:.2f}s"
        )

        # Create video source record in database
        video_source_id = None
        if save_to_db:
            video_source_id = self._create_video_source_record(
                video_path, fps, frame_count, width, height, duration
            )

        # Setup output video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*self.pipeline_config.video_codec)
            writer = cv2.VideoWriter(
                output_path, fourcc, fps, (width, height)
            )

        # Process frames
        all_detections = []
        frame_detections = {}  # For interpolation
        vehicle_plates = {}  # Track plates for each vehicle

        frame_idx = 0
        progress_bar = tqdm(total=frame_count, desc="Processing") if show_progress else None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames if configured
            if self.pipeline_config.frame_skip > 0 and frame_idx % (
                self.pipeline_config.frame_skip + 1
            ) != 0:
                frame_idx += 1
                if progress_bar:
                    progress_bar.update(1)
                continue

            # Resize frame if configured
            if self.pipeline_config.resize_width and self.pipeline_config.resize_height:
                frame = cv2.resize(
                    frame,
                    (self.pipeline_config.resize_width, self.pipeline_config.resize_height),
                )

            # Process frame
            frame_results = self._process_frame(
                frame, frame_idx, fps, video_source_id, vehicle_plates
            )

            all_detections.extend(frame_results)
            if frame_results:
                frame_detections[frame_idx] = frame_results

            # Draw visualizations if output video requested
            if writer is not None:
                vis_frame = self._visualize_detections(frame.copy(), frame_results)
                writer.write(vis_frame)

            frame_idx += 1
            if progress_bar:
                progress_bar.update(1)

        if progress_bar:
            progress_bar.close()

        cap.release()
        if writer:
            writer.release()

        # Interpolate missing frames
        if self.pipeline_config.enable_interpolation and frame_detections:
            logger.info("Interpolating missing frames...")
            interpolated = self.interpolator.interpolate_all_tracks(frame_detections)

            # Add interpolated detections to database
            if save_to_db:
                self._save_interpolated_detections(interpolated, video_source_id)

        # Save to CSV if requested
        if save_to_csv:
            self._save_to_csv(all_detections, video_path)

        # Update video source record
        if save_to_db and video_source_id:
            self._update_video_source_processed(video_source_id)

        # Generate statistics
        self._generate_statistics(video_source_id, all_detections)

        processing_time = time.time() - start_time
        self.stats["processing_time"] = processing_time

        logger.info(
            f"Processing complete: {frame_idx} frames in {processing_time:.2f}s "
            f"({frame_idx / processing_time:.2f} fps)"
        )

        return {
            "video_source_id": video_source_id,
            "frames_processed": frame_idx,
            "detections": len(all_detections),
            "processing_time": processing_time,
            "stats": self.stats,
        }

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        video_source_id: Optional[int],
        vehicle_plates: Dict,
    ) -> List[Dict]:
        """Process a single frame."""
        timestamp = frame_idx / fps if fps > 0 else 0

        # Detect vehicles
        vehicle_detections = self.vehicle_detector.detect(frame)
        self.stats["vehicles_detected"] += len(vehicle_detections)

        # Track vehicles
        tracked_vehicles = self.tracker.update(vehicle_detections, frame_idx)

        # Detect license plates
        results = []
        for vehicle in tracked_vehicles:
            vehicle_bbox = vehicle["bbox"]

            # Detect plate in vehicle region
            plate_detections = self.plate_detector.detect(frame, crop_region=vehicle_bbox)

            if not plate_detections:
                continue

            # Use best plate detection
            best_plate = max(plate_detections, key=lambda x: x["confidence"])
            self.stats["plates_detected"] += 1

            # Crop plate region
            x1, y1, x2, y2 = map(int, best_plate["bbox"])
            plate_img = frame[y1:y2, x1:x2]

            # OCR
            ocr_result = self.ocr.read_plate(plate_img)

            if ocr_result["confidence"] >= self.ocr.confidence_threshold:
                self.stats["plates_recognized"] += 1

                # Store best plate text for this vehicle
                track_id = vehicle["track_id"]
                if track_id not in vehicle_plates:
                    vehicle_plates[track_id] = []

                vehicle_plates[track_id].append(
                    {
                        "text": ocr_result["text_formatted"],
                        "confidence": ocr_result["confidence"],
                        "frame": frame_idx,
                    }
                )

            # Create detection result
            result = {
                "frame_number": frame_idx,
                "timestamp": timestamp,
                "track_id": vehicle["track_id"],
                "vehicle_bbox": vehicle_bbox,
                "vehicle_class": vehicle.get("class_name", "vehicle"),
                "vehicle_confidence": vehicle.get("confidence", 0.0),
                "plate_bbox": best_plate["bbox"],
                "plate_confidence": best_plate["confidence"],
                "plate_text": ocr_result.get("text_formatted", ""),
                "plate_text_raw": ocr_result.get("text", ""),
                "text_confidence": ocr_result.get("confidence", 0.0),
                "is_valid": ocr_result.get("is_valid", False),
                "ocr_engine": ocr_result.get("engine", "unknown"),
            }

            results.append(result)

            # Save to database
            if video_source_id:
                self._save_detection(result, video_source_id)

        self.stats["frames_processed"] += 1
        return results

    def _visualize_detections(
        self, frame: np.ndarray, detections: List[Dict]
    ) -> np.ndarray:
        """Draw visualizations on frame."""
        for det in detections:
            # Draw vehicle bbox
            v_bbox = det["vehicle_bbox"]
            x1, y1, x2, y2 = map(int, v_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw track ID
            track_id = det["track_id"]
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Draw plate bbox
            p_bbox = det["plate_bbox"]
            px1, py1, px2, py2 = map(int, p_bbox)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)

            # Draw plate text
            plate_text = det.get("plate_text", "")
            if plate_text:
                cv2.putText(
                    frame,
                    plate_text,
                    (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

        return frame

    def _create_video_source_record(
        self, video_path: str, fps: float, frame_count: int, width: int, height: int, duration: float
    ) -> int:
        """Create video source record in database."""
        db = self.db_manager.get_session()
        try:
            video_source = VideoSource(
                filename=Path(video_path).name,
                filepath=str(Path(video_path).absolute()),
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
                duration=duration,
                file_size=Path(video_path).stat().st_size if Path(video_path).exists() else 0,
                upload_date=datetime.utcnow(),
            )
            db.add(video_source)
            db.commit()
            db.refresh(video_source)
            return video_source.id
        finally:
            db.close()

    def _save_detection(self, result: Dict, video_source_id: int):
        """Save single detection to database."""
        db = self.db_manager.get_session()
        try:
            detection = Detection(
                video_source_id=video_source_id,
                frame_number=result["frame_number"],
                timestamp=result["timestamp"],
                vehicle_bbox_x1=result["vehicle_bbox"][0],
                vehicle_bbox_y1=result["vehicle_bbox"][1],
                vehicle_bbox_x2=result["vehicle_bbox"][2],
                vehicle_bbox_y2=result["vehicle_bbox"][3],
                vehicle_confidence=result["vehicle_confidence"],
                plate_bbox_x1=result["plate_bbox"][0],
                plate_bbox_y1=result["plate_bbox"][1],
                plate_bbox_x2=result["plate_bbox"][2],
                plate_bbox_y2=result["plate_bbox"][3],
                plate_confidence=result["plate_confidence"],
                plate_text=result.get("plate_text", ""),
                plate_text_confidence=result.get("text_confidence", 0.0),
            )
            db.add(detection)
            db.commit()
        finally:
            db.close()

    def _save_interpolated_detections(self, interpolated: Dict, video_source_id: int):
        """Save interpolated detections to database."""
        # Implementation for saving interpolated data
        pass

    def _update_video_source_processed(self, video_source_id: int):
        """Mark video source as processed."""
        db = self.db_manager.get_session()
        try:
            video_source = db.query(VideoSource).filter_by(id=video_source_id).first()
            if video_source:
                video_source.processed = True
                video_source.processed_date = datetime.utcnow()
                db.commit()
        finally:
            db.close()

    def _generate_statistics(self, video_source_id: Optional[int], detections: List[Dict]):
        """Generate and save traffic statistics."""
        if not video_source_id or not detections:
            return

        # Group by hour
        df = pd.DataFrame(detections)
        # Add your statistics calculation logic here

        logger.info(f"Generated statistics for video {video_source_id}")

    def _save_to_csv(self, detections: List[Dict], video_path: str):
        """Save detections to CSV file."""
        if not detections:
            return

        df = pd.DataFrame(detections)
        csv_path = Path(video_path).stem + "_detections.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved detections to {csv_path}")
