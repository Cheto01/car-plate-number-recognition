"""Database models and connection management for ALPR System."""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

from src.core.config import get_settings

Base = declarative_base()


class VideoSource(Base):
    """Video source information."""

    __tablename__ = "video_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False)
    filepath = Column(Text, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    duration = Column(Float)  # Duration in seconds
    fps = Column(Float)
    frame_count = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    codec = Column(String(50))
    file_size = Column(Integer)  # Size in bytes
    processed = Column(Boolean, default=False)
    processed_date = Column(DateTime)
    metadata = Column(JSON)

    # Relationships
    detections = relationship("Detection", back_populates="video_source", cascade="all, delete-orphan")


class Vehicle(Base):
    """Vehicle tracking information."""

    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, nullable=False, index=True)
    video_source_id = Column(Integer, ForeignKey("video_sources.id"), nullable=False)
    vehicle_type = Column(String(50))  # car, truck, bus, motorcycle
    first_seen_frame = Column(Integer)
    last_seen_frame = Column(Integer)
    first_seen_time = Column(DateTime, default=datetime.utcnow)
    last_seen_time = Column(DateTime)
    total_frames_detected = Column(Integer, default=0)
    confidence_avg = Column(Float)
    direction = Column(String(20))  # north, south, east, west
    estimated_speed = Column(Float)  # km/h
    metadata = Column(JSON)

    # Relationships
    video_source = relationship("VideoSource")
    license_plates = relationship("LicensePlate", back_populates="vehicle", cascade="all, delete-orphan")
    detections = relationship("Detection", back_populates="vehicle", cascade="all, delete-orphan")


class LicensePlate(Base):
    """License plate information."""

    __tablename__ = "license_plates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False)
    plate_number = Column(String(20), nullable=False, index=True)
    plate_number_raw = Column(String(20))  # Before formatting
    country = Column(String(3))  # ISO 3166-1 alpha-3
    state = Column(String(50))
    confidence = Column(Float)
    ocr_engine = Column(String(50))  # paddleocr, easyocr, tesseract
    first_detected_frame = Column(Integer)
    last_detected_frame = Column(Integer)
    detection_count = Column(Integer, default=0)
    is_valid_format = Column(Boolean, default=False)
    detected_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="license_plates")


class Detection(Base):
    """Individual frame detection information."""

    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_source_id = Column(Integer, ForeignKey("video_sources.id"), nullable=False)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"))
    frame_number = Column(Integer, nullable=False, index=True)
    timestamp = Column(Float)  # Time in video (seconds)
    detection_time = Column(DateTime, default=datetime.utcnow)

    # Vehicle bounding box
    vehicle_bbox_x1 = Column(Float)
    vehicle_bbox_y1 = Column(Float)
    vehicle_bbox_x2 = Column(Float)
    vehicle_bbox_y2 = Column(Float)
    vehicle_confidence = Column(Float)

    # License plate bounding box
    plate_bbox_x1 = Column(Float)
    plate_bbox_y1 = Column(Float)
    plate_bbox_x2 = Column(Float)
    plate_bbox_y2 = Column(Float)
    plate_confidence = Column(Float)

    # OCR results
    plate_text = Column(String(20))
    plate_text_confidence = Column(Float)

    # Flags
    is_interpolated = Column(Boolean, default=False)
    quality_score = Column(Float)  # Overall detection quality

    # Relationships
    video_source = relationship("VideoSource", back_populates="detections")
    vehicle = relationship("Vehicle", back_populates="detections")


class TrafficEvent(Base):
    """Traffic events and analytics."""

    __tablename__ = "traffic_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_source_id = Column(Integer, ForeignKey("video_sources.id"), nullable=False)
    event_type = Column(String(50), nullable=False, index=True)  # speeding, parking_violation, etc.
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"))
    frame_number = Column(Integer)
    timestamp = Column(Float)
    severity = Column(String(20))  # low, medium, high, critical
    description = Column(Text)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)

    # Relationships
    video_source = relationship("VideoSource")
    vehicle = relationship("Vehicle")


class TrafficStatistics(Base):
    """Aggregated traffic statistics."""

    __tablename__ = "traffic_statistics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    hour = Column(Integer)  # 0-23
    video_source_id = Column(Integer, ForeignKey("video_sources.id"))

    # Counts
    total_vehicles = Column(Integer, default=0)
    car_count = Column(Integer, default=0)
    truck_count = Column(Integer, default=0)
    bus_count = Column(Integer, default=0)
    motorcycle_count = Column(Integer, default=0)

    # Rates
    plate_detection_rate = Column(Float)  # Percentage
    average_confidence = Column(Float)
    average_speed = Column(Float)

    # Peak information
    is_peak_hour = Column(Boolean, default=False)
    unique_plates_count = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)

    # Relationships
    video_source = relationship("VideoSource")


class MLExperiment(Base):
    """ML experiment tracking."""

    __tablename__ = "ml_experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_name = Column(String(200), nullable=False)
    run_id = Column(String(100), unique=True, index=True)
    model_name = Column(String(100))
    model_version = Column(String(50))
    parameters = Column(JSON)
    metrics = Column(JSON)
    artifacts_path = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    status = Column(String(20))  # running, completed, failed
    notes = Column(Text)


# Database connection management
class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self):
        """Initialize database manager."""
        self.settings = get_settings()
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize database engine based on configuration."""
        db_config = self.settings.database

        if db_config.postgres_enabled:
            # PostgreSQL connection
            connection_string = (
                f"postgresql://{db_config.postgres_username}:{db_config.postgres_password}"
                f"@{db_config.postgres_host}:{db_config.postgres_port}"
                f"/{db_config.postgres_database}"
            )
            self.engine = create_engine(
                connection_string,
                pool_size=db_config.postgres_pool_size,
                max_overflow=db_config.postgres_max_overflow,
                echo=self.settings.debug,
            )
        else:
            # SQLite connection (default)
            connection_string = f"sqlite:///{db_config.sqlite_path}"
            self.engine = create_engine(
                connection_string,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=self.settings.debug,
            )

        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """Drop all database tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()

    def reset_database(self):
        """Reset database (drop and recreate all tables)."""
        self.drop_tables()
        self.create_tables()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get database manager singleton."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.create_tables()
    return _db_manager


def get_db() -> Session:
    """Get database session (for dependency injection)."""
    db_manager = get_db_manager()
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()
