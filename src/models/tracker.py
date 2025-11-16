"""Object tracking using ByteTrack and BotSORT."""
from typing import Dict, List, Optional

import numpy as np
import supervision as sv

from src.core.config import get_settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class VehicleTracker:
    """
    Multi-object tracker for vehicles using ByteTrack or BotSORT.
    Uses the supervision library which provides state-of-the-art tracking algorithms.
    """

    def __init__(
        self,
        tracker_type: str = "bytetrack",
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_box_area: int = 100,
    ):
        """
        Initialize vehicle tracker.

        Args:
            tracker_type: Type of tracker ('bytetrack' or 'botsort')
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to buffer when lost
            match_thresh: Matching threshold (0-1)
            min_box_area: Minimum bounding box area
        """
        self.settings = get_settings()
        tracking_config = self.settings.tracking

        self.tracker_type = tracker_type or tracking_config.tracker_type
        self.track_thresh = track_thresh or tracking_config.track_thresh
        self.track_buffer = track_buffer or tracking_config.track_buffer
        self.match_thresh = match_thresh or tracking_config.match_thresh
        self.min_box_area = min_box_area or tracking_config.min_box_area

        # Initialize tracker based on type
        if self.tracker_type.lower() == "bytetrack":
            self.tracker = sv.ByteTrack(
                track_activation_threshold=self.track_thresh,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
                frame_rate=30,
            )
            logger.info(f"ByteTrack initialized (thresh={self.track_thresh})")
        elif self.tracker_type.lower() == "botsort":
            self.tracker = sv.BYTETracker(
                track_activation_threshold=self.track_thresh,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
            )
            logger.info(f"BotSORT initialized (thresh={self.track_thresh})")
        else:
            raise ValueError(f"Unknown tracker type: {self.tracker_type}")

        self.active_tracks = {}  # Store active track information

    def update(
        self, detections: List[Dict], frame_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections from detector
                       Format: [{'bbox': [x1,y1,x2,y2], 'confidence': score, 'class': id}, ...]
            frame_id: Optional frame number for tracking

        Returns:
            List of tracked detections with track IDs
            Format: [{'bbox': [...], 'track_id': id, 'confidence': score, ...}, ...]
        """
        if not detections:
            # Update tracker with empty detections to maintain track buffers
            empty_sv_detections = sv.Detections.empty()
            self.tracker.update_with_detections(empty_sv_detections)
            return []

        # Convert detections to supervision format
        bboxes = np.array([d["bbox"] for d in detections])
        confidences = np.array([d.get("confidence", 1.0) for d in detections])
        class_ids = np.array([d.get("class", 0) for d in detections])

        # Filter by minimum box area
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        valid_mask = areas >= self.min_box_area
        bboxes = bboxes[valid_mask]
        confidences = confidences[valid_mask]
        class_ids = class_ids[valid_mask]

        if len(bboxes) == 0:
            empty_sv_detections = sv.Detections.empty()
            self.tracker.update_with_detections(empty_sv_detections)
            return []

        # Create supervision Detections object
        sv_detections = sv.Detections(
            xyxy=bboxes,
            confidence=confidences,
            class_id=class_ids,
        )

        # Update tracker
        tracked_detections = self.tracker.update_with_detections(sv_detections)

        # Convert back to our format with track IDs
        tracked_objects = []
        for i in range(len(tracked_detections)):
            track_id = tracked_detections.tracker_id[i]

            obj = {
                "track_id": int(track_id),
                "bbox": tracked_detections.xyxy[i].tolist(),
                "confidence": float(tracked_detections.confidence[i]),
                "class": int(tracked_detections.class_id[i]),
            }

            # Update active tracks info
            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = {
                    "first_frame": frame_id,
                    "last_frame": frame_id,
                    "detections_count": 1,
                }
            else:
                self.active_tracks[track_id]["last_frame"] = frame_id
                self.active_tracks[track_id]["detections_count"] += 1

            tracked_objects.append(obj)

        return tracked_objects

    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """
        Get information about a specific track.

        Args:
            track_id: Track ID to query

        Returns:
            Track information dictionary or None if not found
        """
        return self.active_tracks.get(track_id)

    def get_active_tracks(self) -> Dict[int, Dict]:
        """
        Get all active tracks.

        Returns:
            Dictionary of track_id -> track_info
        """
        return self.active_tracks.copy()

    def reset(self):
        """Reset tracker state."""
        if self.tracker_type.lower() == "bytetrack":
            self.tracker = sv.ByteTrack(
                track_activation_threshold=self.track_thresh,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
                frame_rate=30,
            )
        else:
            self.tracker = sv.BYTETracker(
                track_activation_threshold=self.track_thresh,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
            )

        self.active_tracks = {}
        logger.info("Tracker reset")


class TrackInterpolator:
    """
    Interpolates missing detections in tracking sequences.
    Fills gaps when detections are temporarily lost.
    """

    def __init__(self, max_frames_gap: int = 10):
        """
        Initialize interpolator.

        Args:
            max_frames_gap: Maximum number of frames to interpolate across
        """
        self.max_frames_gap = max_frames_gap

    def interpolate_track(
        self, track_sequence: List[Dict], frame_numbers: List[int]
    ) -> List[Dict]:
        """
        Interpolate missing frames in a track sequence.

        Args:
            track_sequence: List of detections for a track
            frame_numbers: Corresponding frame numbers

        Returns:
            Interpolated track sequence with filled gaps
        """
        if len(track_sequence) < 2:
            return track_sequence

        interpolated = []
        sorted_indices = sorted(range(len(frame_numbers)), key=lambda i: frame_numbers[i])

        for i in range(len(sorted_indices) - 1):
            idx = sorted_indices[i]
            next_idx = sorted_indices[i + 1]

            current_frame = frame_numbers[idx]
            next_frame = frame_numbers[next_idx]
            gap = next_frame - current_frame

            # Add current detection
            interpolated.append(track_sequence[idx])

            # Interpolate if gap is within threshold
            if 1 < gap <= self.max_frames_gap:
                current_bbox = track_sequence[idx]["bbox"]
                next_bbox = track_sequence[next_idx]["bbox"]

                # Linear interpolation
                for frame_offset in range(1, gap):
                    alpha = frame_offset / gap

                    interp_bbox = [
                        current_bbox[j] + alpha * (next_bbox[j] - current_bbox[j])
                        for j in range(4)
                    ]

                    interp_detection = {
                        "bbox": interp_bbox,
                        "track_id": track_sequence[idx]["track_id"],
                        "confidence": track_sequence[idx]["confidence"],
                        "class": track_sequence[idx]["class"],
                        "interpolated": True,
                        "frame_number": current_frame + frame_offset,
                    }

                    interpolated.append(interp_detection)

        # Add last detection
        interpolated.append(track_sequence[sorted_indices[-1]])

        return interpolated

    def interpolate_all_tracks(
        self, detections_by_frame: Dict[int, List[Dict]]
    ) -> Dict[int, List[Dict]]:
        """
        Interpolate all tracks across all frames.

        Args:
            detections_by_frame: Dictionary mapping frame_number -> list of detections

        Returns:
            Dictionary with interpolated detections
        """
        # Group detections by track_id
        tracks = {}
        for frame_num, detections in detections_by_frame.items():
            for det in detections:
                track_id = det.get("track_id")
                if track_id is None:
                    continue

                if track_id not in tracks:
                    tracks[track_id] = {"detections": [], "frames": []}

                tracks[track_id]["detections"].append(det)
                tracks[track_id]["frames"].append(frame_num)

        # Interpolate each track
        interpolated_by_frame = {}
        for track_id, track_data in tracks.items():
            interpolated_track = self.interpolate_track(
                track_data["detections"], track_data["frames"]
            )

            # Re-organize by frame
            for det in interpolated_track:
                frame_num = det.get("frame_number") or track_data["frames"][
                    track_data["detections"].index(det)
                ]

                if frame_num not in interpolated_by_frame:
                    interpolated_by_frame[frame_num] = []

                interpolated_by_frame[frame_num].append(det)

        return interpolated_by_frame
