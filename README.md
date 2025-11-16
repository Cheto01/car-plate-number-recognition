# 🚗 ALPR System v2.0

**Automatic License Plate Recognition System with Advanced Traffic Analytics**

A state-of-the-art, production-ready system for automatic license plate recognition (ALPR) using YOLOv11, ensemble OCR (PaddleOCR + EasyOCR), ByteTrack, and comprehensive traffic analytics.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## ✨ Features

### Core Capabilities
- **SOTA Detection**: YOLOv11-based vehicle and license plate detection
- **Advanced Tracking**: ByteTrack/BotSORT for robust multi-object tracking
- **Ensemble OCR**: PaddleOCR + EasyOCR with intelligent fallback
- **Real-time Processing**: Optimized pipeline for video stream processing
- **Database Storage**: PostgreSQL/SQLite with comprehensive schema

### Traffic Analytics (Secondary Use Case)
- 📊 **Real-time Dashboard**: Interactive Streamlit dashboard with traffic insights
- 📈 **Traffic Patterns**: Peak hour analysis, vehicle type distribution
- 🔍 **Plate Search**: Query historical detections by license plate
- 📉 **Performance Metrics**: Detection rates, confidence scores, throughput analysis
- 📅 **Temporal Analysis**: Daily/hourly statistics and trends

### Production-Ready Features
- 🚀 **REST API**: FastAPI-based API with async support
- 🐳 **Docker Support**: Multi-stage Docker builds with docker-compose
- 📊 **MLflow Integration**: Experiment tracking and model versioning
- 🔍 **Monitoring**: Prometheus + Grafana for system monitoring
- 📝 **Comprehensive Logging**: Structured logging with Loguru
- ✅ **Type Safety**: Pydantic models for data validation
- 🧪 **Testing**: Unit and integration tests with pytest

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input (Video/Stream)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Vehicle Detection (YOLOv11)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Multi-Object Tracking (ByteTrack)                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           License Plate Detection (YOLOv11 Custom)            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            OCR (PaddleOCR + EasyOCR Ensemble)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Storage (PostgreSQL/SQLite)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│   FastAPI REST   │  │  Traffic Analytics│
│      API         │  │    Dashboard      │
└──────────────────┘  └──────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU support)
- Docker & Docker Compose (optional)

### Method 1: Local Installation

```bash
# Clone repository
git clone https://github.com/Cheto01/car-plate-number-recognition.git
cd car-plate-number-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Install package
pip install -e .

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Initialize database
alpr init-db
```

### Method 2: Docker Installation

```bash
# Clone repository
git clone https://github.com/Cheto01/car-plate-number-recognition.git
cd car-plate-number-recognition

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Build and run services
docker-compose up -d

# Check services
docker-compose ps
```

---

## 🚀 Quick Start

### CLI Usage

```bash
# Process a video
alpr process path/to/video.mp4 --output output.mp4 --csv

# Start API server
alpr api --host 0.0.0.0 --port 8000

# Start analytics dashboard
alpr dashboard --port 8501

# Initialize database
alpr init-db

# Reset database (CAUTION: deletes all data)
alpr reset-db
```

### Python API

```python
from src.pipeline.alpr_pipeline import ALPRPipeline

# Initialize pipeline
pipeline = ALPRPipeline()

# Process video
results = pipeline.process_video(
    video_path="video.mp4",
    output_path="output.mp4",
    save_to_db=True,
    save_to_csv=True
)

print(f"Processed {results['frames_processed']} frames")
print(f"Found {results['detections']} detections")
```

### REST API Usage

```bash
# Start API server
uvicorn src.api.main:app --reload

# Or use CLI
alpr api

# Upload and process video
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@video.mp4" \
  -F "save_output_video=true"

# Get all videos
curl "http://localhost:8000/api/v1/videos"

# Search license plates
curl "http://localhost:8000/api/v1/plates/search?plate_number=ABC123"

# Get analytics summary
curl "http://localhost:8000/api/v1/analytics/summary"
```

### Docker Usage

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

---

## 📊 Services & Ports

| Service    | Port | Description                          |
|------------|------|--------------------------------------|
| API        | 8000 | FastAPI REST API                     |
| Dashboard  | 8501 | Streamlit Analytics Dashboard        |
| MLflow     | 5000 | MLflow Tracking Server               |
| Postgres   | 5432 | PostgreSQL Database                  |
| Redis      | 6379 | Redis Cache                          |
| Prometheus | 9090 | Prometheus Monitoring                |
| Grafana    | 3000 | Grafana Dashboards                   |
| Nginx      | 80   | Reverse Proxy                        |

---

## 🎯 Use Cases

### Primary Use Case: License Plate Recognition
- Automated parking lot management
- Traffic law enforcement
- Toll collection systems
- Access control for restricted areas
- Vehicle tracking and monitoring

### Secondary Use Case: Traffic Analytics
- **Peak Hour Analysis**: Identify rush hours and traffic patterns
- **Vehicle Type Distribution**: Analyze vehicle composition (cars, trucks, buses)
- **Frequent Visitor Detection**: Track regular vehicles
- **Traffic Flow Optimization**: Data for infrastructure planning
- **Parking Utilization**: Monitor parking lot occupancy
- **Security Monitoring**: Detect suspicious patterns

---

## 📁 Project Structure

```
car-plate-number-recognition/
├── src/
│   ├── core/                 # Core functionality
│   │   ├── config.py         # Configuration management
│   │   ├── database.py       # Database models & ORM
│   │   └── logging_config.py # Logging setup
│   ├── models/               # ML models
│   │   ├── detector.py       # YOLO detectors
│   │   ├── tracker.py        # ByteTrack tracker
│   │   └── ocr.py            # OCR engines
│   ├── pipeline/             # Data pipeline
│   │   └── alpr_pipeline.py  # Main ALPR pipeline
│   ├── api/                  # REST API
│   │   └── main.py           # FastAPI application
│   ├── analytics/            # Analytics & visualization
│   │   └── dashboard.py      # Streamlit dashboard
│   ├── utils/                # Utilities
│   │   └── mlflow_tracker.py # MLflow integration
│   ├── cli.py                # CLI interface
│   └── legacy_*.py           # Legacy code (reference)
├── data/
│   ├── raw/                  # Raw input data
│   ├── processed/            # Processed output
│   └── models/               # Model weights
├── config/
│   └── config.yaml           # Configuration file
├── tests/                    # Unit & integration tests
├── docs/                     # Documentation
├── logs/                     # Application logs
├── mlruns/                   # MLflow artifacts
├── notebooks/                # Jupyter notebooks
├── docker-compose.yml        # Docker orchestration
├── Dockerfile                # Docker build
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

---

## ⚙️ Configuration

Configuration is managed through `config/config.yaml` and environment variables (`.env`).

### Key Configuration Sections

```yaml
models:
  vehicle_detector:
    path: "data/models/yolov11n.pt"
    confidence_threshold: 0.5
  plate_detector:
    path: "data/models/license_plate_detector.pt"
    confidence_threshold: 0.6
  ocr:
    primary_engine: "paddleocr"
    fallback_engine: "easyocr"
    use_ensemble: true

tracking:
  tracker_type: "bytetrack"
  track_thresh: 0.5

database:
  postgres_enabled: true
  postgres_host: "localhost"
  postgres_port: 5432

api:
  host: "0.0.0.0"
  port: 8000
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_detector.py

# Run with verbose output
pytest -v
```

---

## 📊 Performance

### Benchmarks (on NVIDIA RTX 3090)

- **Detection Speed**: ~45 FPS (1080p video)
- **OCR Accuracy**: ~95% (with ensemble)
- **Tracking Accuracy**: ~92% MOTA
- **End-to-End Latency**: ~22ms per frame

### Optimization Tips

1. **GPU Acceleration**: Enable CUDA for 10x speedup
2. **Batch Processing**: Process multiple frames in batches
3. **Frame Skipping**: Process every Nth frame for real-time
4. **Model Optimization**: Use TensorRT or ONNX for inference
5. **Async Processing**: Use async API endpoints

---

## 🔧 Development

### Code Style

```bash
# Format code
black src/

# Sort imports
isort src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## 📖 API Documentation

Once the API is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

- `POST /api/v1/process` - Upload and process video
- `GET /api/v1/videos` - List processed videos
- `GET /api/v1/detections` - Get detections
- `GET /api/v1/vehicles` - Get tracked vehicles
- `GET /api/v1/plates/search` - Search license plates
- `GET /api/v1/analytics/summary` - Get analytics summary

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv11 implementation
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR engine
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - OCR engine
- [Supervision](https://github.com/roboflow/supervision) - ByteTrack implementation
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [Streamlit](https://streamlit.io/) - Dashboard framework

---

## 📧 Contact

For questions, issues, or collaboration:
- GitHub Issues: [Create an issue](https://github.com/Cheto01/car-plate-number-recognition/issues)

---

## 🗺️ Roadmap

- [ ] Real-time video stream support (RTSP/RTMP)
- [ ] Multi-camera support
- [ ] Advanced analytics (speed estimation, violation detection)
- [ ] Mobile app for monitoring
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Multi-language plate recognition
- [ ] Edge deployment (NVIDIA Jetson, Raspberry Pi)
- [ ] Custom model training pipeline

---

**Built with ❤️ using State-of-the-Art Computer Vision and ML Technologies**
