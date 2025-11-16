"""OCR engines for license plate text recognition."""
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.config import get_settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class OCREngine:
    """Base class for OCR engines."""

    def __init__(self, languages: List[str] = None, gpu: bool = True):
        """Initialize OCR engine."""
        self.languages = languages or ["en"]
        self.gpu = gpu

    def read_text(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Read text from image.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Tuple of (text, confidence)
        """
        raise NotImplementedError


class EasyOCREngine(OCREngine):
    """EasyOCR engine for text recognition."""

    def __init__(self, languages: List[str] = None, gpu: bool = True):
        """Initialize EasyOCR."""
        super().__init__(languages, gpu)

        try:
            import easyocr

            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info(f"EasyOCR initialized (languages={self.languages}, gpu={self.gpu})")
        except ImportError:
            logger.error("EasyOCR not installed. Install with: pip install easyocr")
            self.reader = None

    def read_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Read text using EasyOCR."""
        if self.reader is None:
            return "", 0.0

        try:
            results = self.reader.readtext(image)

            if not results:
                return "", 0.0

            # Get best result (highest confidence)
            best_result = max(results, key=lambda x: x[2])
            text = best_result[1].strip()
            confidence = float(best_result[2])

            return text, confidence

        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return "", 0.0


class PaddleOCREngine(OCREngine):
    """PaddleOCR engine for text recognition."""

    def __init__(self, languages: List[str] = None, gpu: bool = True):
        """Initialize PaddleOCR."""
        super().__init__(languages, gpu)

        try:
            from paddleocr import PaddleOCR

            # Map language codes (PaddleOCR uses different codes)
            lang_map = {"en": "en", "ch": "ch", "es": "es", "fr": "fr"}
            paddle_lang = lang_map.get(self.languages[0], "en")

            self.reader = PaddleOCR(
                use_angle_cls=True,
                lang=paddle_lang,
                use_gpu=self.gpu,
                show_log=False,
            )
            logger.info(f"PaddleOCR initialized (lang={paddle_lang}, gpu={self.gpu})")
        except ImportError:
            logger.error("PaddleOCR not installed. Install with: pip install paddleocr")
            self.reader = None

    def read_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Read text using PaddleOCR."""
        if self.reader is None:
            return "", 0.0

        try:
            result = self.reader.ocr(image, cls=True)

            if not result or not result[0]:
                return "", 0.0

            # Extract text and confidence
            texts = []
            confidences = []

            for line in result[0]:
                text = line[1][0]
                conf = line[1][1]
                texts.append(text)
                confidences.append(conf)

            # Combine results
            full_text = " ".join(texts).strip()
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return full_text, avg_confidence

        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return "", 0.0


class TesseractOCREngine(OCREngine):
    """Tesseract OCR engine for text recognition."""

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        """Initialize Tesseract."""
        super().__init__(languages, gpu)

        try:
            import pytesseract

            self.tesseract = pytesseract
            logger.info(f"Tesseract initialized (languages={self.languages})")
        except ImportError:
            logger.error("Tesseract not installed. Install with: pip install pytesseract")
            self.tesseract = None

    def read_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Read text using Tesseract."""
        if self.tesseract is None:
            return "", 0.0

        try:
            # Configure Tesseract for license plates
            config = "--psm 7 --oem 3"  # PSM 7: single text line

            data = self.tesseract.image_to_data(
                image, lang="+".join(self.languages), config=config, output_type="dict"
            )

            # Filter valid text
            texts = []
            confidences = []

            for i, conf in enumerate(data["conf"]):
                if int(conf) > 0:
                    text = data["text"][i]
                    if text.strip():
                        texts.append(text)
                        confidences.append(int(conf) / 100.0)

            if not texts:
                return "", 0.0

            full_text = " ".join(texts).strip()
            avg_confidence = sum(confidences) / len(confidences)

            return full_text, avg_confidence

        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return "", 0.0


class LicensePlateOCR:
    """
    Advanced OCR system for license plate recognition.
    Supports multiple OCR engines with ensemble voting.
    """

    def __init__(
        self,
        primary_engine: str = "paddleocr",
        fallback_engine: str = "easyocr",
        use_ensemble: bool = True,
        languages: List[str] = None,
        confidence_threshold: float = 0.7,
        gpu: bool = True,
    ):
        """
        Initialize license plate OCR.

        Args:
            primary_engine: Primary OCR engine ('paddleocr', 'easyocr', 'tesseract')
            fallback_engine: Fallback engine if primary fails
            use_ensemble: Use ensemble of multiple engines for better accuracy
            languages: List of language codes
            confidence_threshold: Minimum confidence to accept
            gpu: Use GPU acceleration
        """
        self.settings = get_settings()
        ocr_config = self.settings.models.get("ocr", {})

        self.primary_engine_name = primary_engine or ocr_config.get("primary_engine", "paddleocr")
        self.fallback_engine_name = fallback_engine or ocr_config.get(
            "fallback_engine", "easyocr"
        )
        self.use_ensemble = use_ensemble or ocr_config.get("use_ensemble", True)
        self.languages = languages or ocr_config.get("languages", ["en"])
        self.confidence_threshold = confidence_threshold or ocr_config.get(
            "confidence_threshold", 0.7
        )
        self.gpu = gpu or ocr_config.get("gpu", True)

        # Initialize engines
        self.engines = {}
        self._init_engines()

        # Character mappings for common OCR errors
        self.char_map = {
            "O": "0",
            "I": "1",
            "Z": "2",
            "S": "5",
            "B": "8",
            "G": "6",
            "o": "0",
            "i": "1",
            "l": "1",
            "z": "2",
            "s": "5",
            "b": "8",
            "g": "6",
        }

    def _init_engines(self):
        """Initialize OCR engines."""
        engine_classes = {
            "paddleocr": PaddleOCREngine,
            "easyocr": EasyOCREngine,
            "tesseract": TesseractOCREngine,
        }

        # Initialize primary engine
        if self.primary_engine_name in engine_classes:
            try:
                self.engines[self.primary_engine_name] = engine_classes[
                    self.primary_engine_name
                ](self.languages, self.gpu)
                logger.info(f"Primary OCR engine: {self.primary_engine_name}")
            except Exception as e:
                logger.error(f"Failed to initialize {self.primary_engine_name}: {e}")

        # Initialize fallback engine
        if self.fallback_engine_name in engine_classes:
            try:
                self.engines[self.fallback_engine_name] = engine_classes[
                    self.fallback_engine_name
                ](self.languages, self.gpu)
                logger.info(f"Fallback OCR engine: {self.fallback_engine_name}")
            except Exception as e:
                logger.error(f"Failed to initialize {self.fallback_engine_name}: {e}")

        # Initialize ensemble (all engines)
        if self.use_ensemble:
            for engine_name in ["tesseract"]:
                if engine_name not in self.engines and engine_name in engine_classes:
                    try:
                        self.engines[engine_name] = engine_classes[engine_name](
                            self.languages, self.gpu
                        )
                    except Exception as e:
                        logger.warning(f"Could not initialize {engine_name}: {e}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.

        Args:
            image: Input license plate image

        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize if too small
        h, w = gray.shape
        if h < 50 or w < 100:
            scale = max(50 / h, 100 / w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

        return denoised

    def format_license_plate(self, text: str) -> str:
        """
        Format and correct common OCR errors in license plate text.

        Args:
            text: Raw OCR text

        Returns:
            Formatted text
        """
        # Remove spaces and special characters
        text = re.sub(r"[^A-Z0-9]", "", text.upper())

        # Apply character corrections (context-aware)
        # For US plates: typically 2-3 letters, then digits, then letters
        # This is a simple heuristic - can be improved with ML
        formatted = ""
        for i, char in enumerate(text):
            if char in self.char_map:
                # Decide whether to convert based on position/context
                # Simple rule: if surrounded by letters, keep as letter
                prev_is_letter = i > 0 and text[i - 1].isalpha()
                next_is_letter = i < len(text) - 1 and text[i + 1].isalpha()

                if prev_is_letter or next_is_letter:
                    formatted += char  # Keep as letter
                else:
                    formatted += self.char_map[char]  # Convert to digit
            else:
                formatted += char

        return formatted

    def validate_format(self, text: str, pattern: str = r"^[A-Z]{2,3}[0-9]{2,4}[A-Z]{0,3}$") -> bool:
        """
        Validate license plate format.

        Args:
            text: License plate text
            pattern: Regex pattern to match

        Returns:
            True if valid format
        """
        return bool(re.match(pattern, text))

    def read_plate(
        self, image: np.ndarray, use_preprocessing: bool = True
    ) -> Dict[str, any]:
        """
        Read license plate text from image.

        Args:
            image: License plate image (cropped)
            use_preprocessing: Apply preprocessing

        Returns:
            Dictionary with OCR results:
            {
                'text': str,
                'text_formatted': str,
                'confidence': float,
                'engine': str,
                'is_valid': bool,
                'all_results': List[Dict]  # if ensemble
            }
        """
        if use_preprocessing:
            processed = self.preprocess_image(image)
        else:
            processed = image

        results = []

        # Try all engines
        for engine_name, engine in self.engines.items():
            text, confidence = engine.read_text(processed)

            if text:
                formatted_text = self.format_license_plate(text)
                is_valid = self.validate_format(formatted_text)

                results.append(
                    {
                        "text": text,
                        "text_formatted": formatted_text,
                        "confidence": confidence,
                        "engine": engine_name,
                        "is_valid": is_valid,
                    }
                )

        if not results:
            return {
                "text": "",
                "text_formatted": "",
                "confidence": 0.0,
                "engine": "none",
                "is_valid": False,
                "all_results": [],
            }

        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)

        if self.use_ensemble and len(results) > 1:
            # Ensemble voting: use most common formatted text among valid results
            valid_results = [r for r in results if r["is_valid"]]

            if valid_results:
                # Count occurrences
                text_counts = {}
                for r in valid_results:
                    text = r["text_formatted"]
                    if text not in text_counts:
                        text_counts[text] = {"count": 0, "confidence": 0.0}
                    text_counts[text]["count"] += 1
                    text_counts[text]["confidence"] += r["confidence"]

                # Get most common
                best_text = max(text_counts.items(), key=lambda x: (x[1]["count"], x[1]["confidence"]))

                return {
                    "text": best_text[0],
                    "text_formatted": best_text[0],
                    "confidence": best_text[1]["confidence"] / best_text[1]["count"],
                    "engine": "ensemble",
                    "is_valid": True,
                    "all_results": results,
                }

        # Return best result
        best_result = results[0].copy()
        best_result["all_results"] = results
        return best_result
