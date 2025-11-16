"""Configuration management for ALPR System."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Model configuration."""

    name: str
    path: str
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "cuda"
    classes: Optional[List[int]] = None

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device selection."""
        valid_devices = ["cuda", "cpu", "mps"]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}")
        return v


class OCRConfig(BaseSettings):
    """OCR configuration."""

    primary_engine: str = "paddleocr"
    fallback_engine: str = "easyocr"
    use_ensemble: bool = True
    languages: List[str] = ["en"]
    confidence_threshold: float = 0.7
    gpu: bool = True


class TrackingConfig(BaseSettings):
    """Tracking configuration."""

    tracker_type: str = "bytetrack"
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    min_box_area: int = 100
    mot20: bool = False


class PipelineConfig(BaseSettings):
    """Pipeline configuration."""

    source_type: str = "video"
    batch_size: int = 1
    frame_skip: int = 0
    resize_width: int = 1280
    resize_height: int = 720
    enable_interpolation: bool = True
    enable_quality_filter: bool = True
    min_detection_quality: float = 0.6
    max_frames_gap: int = 10
    save_video: bool = True
    save_images: bool = False
    save_csv: bool = True
    save_database: bool = True
    video_codec: str = "mp4v"
    video_fps: int = 30


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    # SQLite
    sqlite_enabled: bool = True
    sqlite_path: str = "data/processed/alpr.db"

    # PostgreSQL
    postgres_enabled: bool = False
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "alpr_db"
    postgres_username: str = "alpr_user"
    postgres_password: str = ""
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20

    # MongoDB
    mongodb_enabled: bool = False
    mongodb_host: str = "localhost"
    mongodb_port: int = 27017
    mongodb_database: str = "alpr_analytics"
    mongodb_username: str = ""
    mongodb_password: str = ""

    model_config = SettingsConfigDict(env_prefix="DB_")


class APIConfig(BaseSettings):
    """API configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = True
    cors_origins: List[str] = ["*"]
    max_upload_size: int = 524288000  # 500 MB
    rate_limit: str = "100/minute"

    model_config = SettingsConfigDict(env_prefix="API_")


class MLflowConfig(BaseSettings):
    """MLflow configuration."""

    enabled: bool = True
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "alpr-experiments"
    artifact_location: str = "./mlruns/artifacts"

    model_config = SettingsConfigDict(env_prefix="MLFLOW_")


class AnalyticsConfig(BaseSettings):
    """Analytics configuration."""

    dashboard_enabled: bool = True
    dashboard_port: int = 8501
    dashboard_theme: str = "dark"
    metrics: List[str] = Field(
        default=[
            "vehicle_count",
            "plate_detection_rate",
            "peak_traffic_hours",
            "vehicle_type_distribution",
            "average_speed",
            "plate_frequency",
        ]
    )
    generate_daily: bool = True
    generate_weekly: bool = True
    export_format: List[str] = ["pdf", "html", "json"]


class MonitoringConfig(BaseSettings):
    """Monitoring configuration."""

    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    sentry_enabled: bool = False
    sentry_dsn: str = ""
    traces_sample_rate: float = 0.1

    model_config = SettingsConfigDict(env_prefix="MONITORING_")


class PerformanceConfig(BaseSettings):
    """Performance configuration."""

    use_gpu: bool = True
    gpu_batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    mixed_precision: bool = True


class Settings(BaseSettings):
    """Main application settings."""

    # Application
    app_name: str = "ALPR System"
    app_version: str = "2.0.0"
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    config_dir: Path = Field(default_factory=lambda: Path("config"))
    logs_dir: Path = Field(default_factory=lambda: Path("logs"))

    # Component configs
    models: Dict[str, Any] = Field(default_factory=dict)
    tracking: Optional[TrackingConfig] = None
    pipeline: Optional[PipelineConfig] = None
    database: Optional[DatabaseConfig] = None
    api: Optional[APIConfig] = None
    mlflow: Optional[MLflowConfig] = None
    analytics: Optional[AnalyticsConfig] = None
    monitoring: Optional[MonitoringConfig] = None
    performance: Optional[PerformanceConfig] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def from_yaml(cls, config_path: str = "config/config.yaml") -> "Settings":
        """Load settings from YAML file."""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Create component configs
        settings_dict = {
            "app_name": config_data.get("app", {}).get("name", "ALPR System"),
            "app_version": config_data.get("app", {}).get("version", "2.0.0"),
            "environment": config_data.get("app", {}).get("environment", "development"),
            "debug": config_data.get("app", {}).get("debug", True),
            "log_level": config_data.get("app", {}).get("log_level", "INFO"),
            "models": config_data.get("models", {}),
        }

        # Create instance
        instance = cls(**settings_dict)

        # Initialize component configs
        if "tracking" in config_data:
            instance.tracking = TrackingConfig(**config_data["tracking"])

        if "pipeline" in config_data:
            pipeline_data = {**config_data["pipeline"].get("input", {})}
            pipeline_data.update(config_data["pipeline"].get("processing", {}))
            pipeline_data.update(config_data["pipeline"].get("output", {}))
            instance.pipeline = PipelineConfig(**pipeline_data)

        if "database" in config_data:
            db_data = {}
            for db_type in ["sqlite", "postgres", "mongodb"]:
                if db_type in config_data["database"]:
                    for key, value in config_data["database"][db_type].items():
                        db_data[f"{db_type}_{key}"] = value
            instance.database = DatabaseConfig(**db_data)

        if "api" in config_data:
            instance.api = APIConfig(**config_data["api"])

        if "mlflow" in config_data:
            instance.mlflow = MLflowConfig(**config_data["mlflow"])

        if "analytics" in config_data:
            analytics_data = {**config_data["analytics"].get("dashboard", {})}
            analytics_data.update(config_data["analytics"].get("metrics", {}))
            analytics_data.update(config_data["analytics"].get("reports", {}))
            instance.analytics = AnalyticsConfig(**analytics_data)

        if "monitoring" in config_data:
            mon_data = {}
            if "prometheus" in config_data["monitoring"]:
                for key, value in config_data["monitoring"]["prometheus"].items():
                    mon_data[f"prometheus_{key}"] = value
            if "sentry" in config_data["monitoring"]:
                for key, value in config_data["monitoring"]["sentry"].items():
                    mon_data[f"sentry_{key}"] = value
            instance.monitoring = MonitoringConfig(**mon_data)

        if "performance" in config_data:
            instance.performance = PerformanceConfig(**config_data["performance"])

        return instance

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "models",
            self.logs_dir,
            self.config_dir,
            "mlruns",
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings singleton."""
    global _settings
    if _settings is None:
        config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
        if os.path.exists(config_path):
            _settings = Settings.from_yaml(config_path)
        else:
            _settings = Settings()
        _settings.ensure_directories()
    return _settings


def reload_settings() -> Settings:
    """Reload settings (useful for testing)."""
    global _settings
    _settings = None
    return get_settings()
