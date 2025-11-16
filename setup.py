"""Setup configuration for ALPR System."""
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="alpr-system",
    version="2.0.0",
    author="ALPR Team",
    description="Automatic License Plate Recognition System with Traffic Analytics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cheto01/car-plate-number-recognition",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "ultralytics>=8.1.0",
        "torch>=2.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "easyocr>=1.7.0",
        "paddleocr>=2.7.0",
        "supervision>=0.18.0",
        "pandas>=2.0.0",
        "sqlalchemy>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "pydantic-settings>=2.0.0",
        "mlflow>=2.8.0",
        "streamlit>=1.28.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "loguru>=0.7.0",
        "click>=8.1.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "alpr=src.cli:main",
            "alpr-api=src.api.main:start_server",
            "alpr-dashboard=src.analytics.dashboard:main",
        ],
    },
)
