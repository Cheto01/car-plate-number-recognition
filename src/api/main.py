"""FastAPI REST API for ALPR System."""
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.database import Detection, LicensePlate, Vehicle, VideoSource, get_db_manager
from src.core.logging_config import get_logger, setup_logging
from src.pipeline.alpr_pipeline import ALPRPipeline

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI
settings = get_settings()
app = FastAPI(
    title="ALPR System API",
    description="Automatic License Plate Recognition System with Traffic Analytics",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database
db_manager = get_db_manager()

# Pipeline instance (lazy initialization)
_pipeline = None


def get_pipeline() -> ALPRPipeline:
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ALPRPipeline()
    return _pipeline


# Pydantic models
class ProcessVideoRequest(BaseModel):
    """Request model for video processing."""

    save_output_video: bool = True
    save_to_csv: bool = False


class ProcessVideoResponse(BaseModel):
    """Response model for video processing."""

    video_source_id: int
    frames_processed: int
    detections: int
    processing_time: float
    output_video_url: Optional[str] = None


class DetectionResponse(BaseModel):
    """Response model for detection."""

    id: int
    frame_number: int
    timestamp: float
    vehicle_bbox: List[float]
    plate_bbox: Optional[List[float]] = None
    plate_text: Optional[str] = None
    confidence: float


class VehicleResponse(BaseModel):
    """Response model for vehicle."""

    id: int
    track_id: int
    vehicle_type: str
    first_seen_frame: int
    last_seen_frame: int
    total_frames_detected: int
    license_plates: List[str] = []


class VideoSourceResponse(BaseModel):
    """Response model for video source."""

    id: int
    filename: str
    upload_date: datetime
    duration: float
    fps: float
    frame_count: int
    processed: bool


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "ALPR System API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "process_video": "/api/v1/process",
            "videos": "/api/v1/videos",
            "detections": "/api/v1/detections",
            "vehicles": "/api/v1/vehicles",
            "analytics": "/api/v1/analytics",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/v1/process", response_model=ProcessVideoResponse)
async def process_video(
    file: UploadFile = File(...),
    save_output_video: bool = Query(True),
    save_to_csv: bool = Query(False),
):
    """
    Process uploaded video for license plate detection.

    Args:
        file: Video file to process
        save_output_video: Generate output video with visualizations
        save_to_csv: Save results to CSV

    Returns:
        Processing results and statistics
    """
    logger.info(f"Received video upload: {file.filename}")

    # Validate file
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(400, "Invalid video format. Supported: mp4, avi, mov, mkv")

    # Save uploaded file
    upload_dir = Path("data/raw/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    video_path = upload_dir / f"{datetime.utcnow().timestamp()}_{file.filename}"

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Video saved to {video_path}")

        # Process video
        output_path = None
        if save_output_video:
            output_path = str(video_path).replace("/raw/", "/processed/").replace(
                file.filename, f"processed_{file.filename}"
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        pipeline = get_pipeline()
        results = pipeline.process_video(
            video_path=str(video_path),
            output_path=output_path,
            save_to_db=True,
            save_to_csv=save_to_csv,
        )

        response = ProcessVideoResponse(
            video_source_id=results["video_source_id"],
            frames_processed=results["frames_processed"],
            detections=results["detections"],
            processing_time=results["processing_time"],
            output_video_url=f"/api/v1/videos/{results['video_source_id']}/download"
            if save_output_video
            else None,
        )

        logger.info(f"Video processing complete: {results}")
        return response

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(500, f"Error processing video: {str(e)}")
    finally:
        file.file.close()


@app.get("/api/v1/videos", response_model=List[VideoSourceResponse])
async def list_videos(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    processed_only: bool = Query(False),
):
    """
    List all processed videos.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        processed_only: Return only processed videos

    Returns:
        List of video sources
    """
    db = db_manager.get_session()
    try:
        query = db.query(VideoSource)

        if processed_only:
            query = query.filter(VideoSource.processed == True)

        videos = query.offset(skip).limit(limit).all()

        return [
            VideoSourceResponse(
                id=v.id,
                filename=v.filename,
                upload_date=v.upload_date,
                duration=v.duration,
                fps=v.fps,
                frame_count=v.frame_count,
                processed=v.processed,
            )
            for v in videos
        ]
    finally:
        db.close()


@app.get("/api/v1/videos/{video_id}")
async def get_video(video_id: int):
    """Get video details."""
    db = db_manager.get_session()
    try:
        video = db.query(VideoSource).filter(VideoSource.id == video_id).first()

        if not video:
            raise HTTPException(404, "Video not found")

        return {
            "id": video.id,
            "filename": video.filename,
            "filepath": video.filepath,
            "upload_date": video.upload_date,
            "duration": video.duration,
            "fps": video.fps,
            "frame_count": video.frame_count,
            "width": video.width,
            "height": video.height,
            "processed": video.processed,
            "processed_date": video.processed_date,
        }
    finally:
        db.close()


@app.get("/api/v1/videos/{video_id}/download")
async def download_video(video_id: int):
    """Download processed video."""
    db = db_manager.get_session()
    try:
        video = db.query(VideoSource).filter(VideoSource.id == video_id).first()

        if not video:
            raise HTTPException(404, "Video not found")

        if not video.processed:
            raise HTTPException(400, "Video not yet processed")

        # Find processed video file
        processed_path = Path(video.filepath).parent.parent / "processed" / f"processed_{video.filename}"

        if not processed_path.exists():
            raise HTTPException(404, "Processed video file not found")

        return FileResponse(processed_path, media_type="video/mp4", filename=f"processed_{video.filename}")
    finally:
        db.close()


@app.get("/api/v1/detections")
async def list_detections(
    video_id: Optional[int] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    """
    List detections.

    Args:
        video_id: Filter by video ID
        skip: Number of records to skip
        limit: Maximum number of records to return

    Returns:
        List of detections
    """
    db = db_manager.get_session()
    try:
        query = db.query(Detection)

        if video_id:
            query = query.filter(Detection.video_source_id == video_id)

        detections = query.offset(skip).limit(limit).all()

        return [
            {
                "id": d.id,
                "frame_number": d.frame_number,
                "timestamp": d.timestamp,
                "vehicle_bbox": [
                    d.vehicle_bbox_x1,
                    d.vehicle_bbox_y1,
                    d.vehicle_bbox_x2,
                    d.vehicle_bbox_y2,
                ],
                "plate_bbox": [
                    d.plate_bbox_x1,
                    d.plate_bbox_y1,
                    d.plate_bbox_x2,
                    d.plate_bbox_y2,
                ]
                if d.plate_bbox_x1
                else None,
                "plate_text": d.plate_text,
                "vehicle_confidence": d.vehicle_confidence,
                "plate_confidence": d.plate_confidence,
            }
            for d in detections
        ]
    finally:
        db.close()


@app.get("/api/v1/vehicles")
async def list_vehicles(
    video_id: Optional[int] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    """List tracked vehicles."""
    db = db_manager.get_session()
    try:
        query = db.query(Vehicle)

        if video_id:
            query = query.filter(Vehicle.video_source_id == video_id)

        vehicles = query.offset(skip).limit(limit).all()

        return [
            {
                "id": v.id,
                "track_id": v.track_id,
                "vehicle_type": v.vehicle_type,
                "first_seen_frame": v.first_seen_frame,
                "last_seen_frame": v.last_seen_frame,
                "total_frames_detected": v.total_frames_detected,
                "license_plates": [lp.plate_number for lp in v.license_plates],
            }
            for v in vehicles
        ]
    finally:
        db.close()


@app.get("/api/v1/plates/search")
async def search_plates(plate_number: str = Query(..., min_length=1)):
    """Search for a specific license plate."""
    db = db_manager.get_session()
    try:
        plates = (
            db.query(LicensePlate)
            .filter(LicensePlate.plate_number.like(f"%{plate_number}%"))
            .all()
        )

        return [
            {
                "id": p.id,
                "plate_number": p.plate_number,
                "vehicle_id": p.vehicle_id,
                "confidence": p.confidence,
                "first_detected_frame": p.first_detected_frame,
                "detection_count": p.detection_count,
                "is_valid_format": p.is_valid_format,
            }
            for p in plates
        ]
    finally:
        db.close()


@app.get("/api/v1/analytics/summary")
async def analytics_summary(video_id: Optional[int] = Query(None)):
    """Get analytics summary."""
    db = db_manager.get_session()
    try:
        query = db.query(Detection)
        if video_id:
            query = query.filter(Detection.video_source_id == video_id)

        total_detections = query.count()

        plates_detected = (
            query.filter(Detection.plate_text != None).filter(Detection.plate_text != "").count()
        )

        unique_plates = (
            db.query(LicensePlate.plate_number).distinct().count()
        )

        return {
            "total_detections": total_detections,
            "plates_detected": plates_detected,
            "unique_plates": unique_plates,
            "detection_rate": plates_detected / total_detections if total_detections > 0 else 0,
        }
    finally:
        db.close()


def start_server():
    """Start the API server."""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
    )


if __name__ == "__main__":
    start_server()
