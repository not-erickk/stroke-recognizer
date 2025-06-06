import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from scipy import stats
from sklearn.linear_model import LinearRegression


@dataclass
class Detection:
    """Represents a single detection from the YOLO model."""
    class_name: str
    bbox: Tuple[float, float, float, float]  # (x_center, y_center, width, height)
    confidence: float

    @property
    def x_min(self) -> float:
        return self.bbox[0] - self.bbox[2] / 2

    @property
    def x_max(self) -> float:
        return self.bbox[0] + self.bbox[2] / 2

    @property
    def y_min(self) -> float:
        return self.bbox[1] - self.bbox[3] / 2

    @property
    def y_max(self) -> float:
        return self.bbox[1] + self.bbox[3] / 2

    @property
    def area(self) -> float:
        return self.bbox[2] * self.bbox[3]


class HandwritingMetricsAnalyzer:
    """
    Analyzes handwriting quality based on YOLO detections.

    Calculates four main metrics:
    1. Legibilidad (Legibility): How understandable the word is
    2. Grafía (Graphology): How well-formed the letters are
    3. Alineación (Alignment): Respects established alignment
    4. Tamaño (Size): Maintains consistent size with scale and grid
    """

    # Define error classes
    ERROR_CLASSES = {
        'separation_from_line', 'sl',  # sl = separation_from_line
        'incomplete_loop', 'il',        # il = incomplete_loop
        'out_of_bounds', 'ob',          # ob = out_of_bounds
        'small_char', 'sc'              # sc = small_char
    }

    # Expected letters for Spanish words (can be configured)
    SPANISH_ALPHABET = set('abcdefghijklmnñopqrstuvwxyz')

    def __init__(self,
                 expected_word: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 grid_size: Optional[Tuple[int, int]] = None):
        """
        Initialize the analyzer.

        Args:
            expected_word: The expected word (if known) for better legibility analysis
            confidence_threshold: Minimum confidence to consider a detection valid
            grid_size: Expected grid cell size (width, height) in pixels
        """
        self.expected_word = expected_word.lower() if expected_word else None
        self.confidence_threshold = confidence_threshold
        self.grid_size = grid_size or (50, 50)  # Default grid size

    def analyze(self, detections: List[Dict]) -> Dict[str, float]:
        """
        Main analysis function that calculates all metrics.

        Args:
            detections: List of detection dictionaries from YOLO
                       Each dict contains: 'class', 'bbox', 'confidence'
                       OR a dict with 'predictions' key containing the list

        Returns:
            Dictionary with metrics: legibilidad, grafia, alineacion, tamaño
        """
        # Handle both formats: direct list or dict with 'predictions' key
        if isinstance(detections, dict) and 'predictions' in detections:
            detections = detections['predictions']
        
        # Convert to Detection objects
        det_objects = self._parse_detections(detections)

        # Separate letters and errors
        letters = [d for d in det_objects if self._is_letter(d.class_name)]
        errors = [d for d in det_objects if d.class_name in self.ERROR_CLASSES]

        # Calculate individual metrics
        legibilidad = self._calculate_legibilidad(letters, errors)
        grafia = self._calculate_grafia(letters, errors)
        alineacion = self._calculate_alineacion(letters, errors)
        tamano = self._calculate_tamano(letters, errors)

        return {
            'legibilidad': round(legibilidad, 2),
            'grafia': round(grafia, 2),
            'alineacion': round(alineacion, 2),
            'tamano': round(tamano, 2),
            'overall': round(np.mean([legibilidad, grafia, alineacion, tamano]), 2)
        }

    def _parse_detections(self, detections: List[Dict]) -> List[Detection]:
        """Parse raw detections into Detection objects."""
        parsed = []
        for det in detections:
            if det['confidence'] >= self.confidence_threshold:
                # Handle different bbox formats
                if 'bbox' in det:
                    # Format 1: bbox as array/tuple [x_center, y_center, width, height]
                    bbox = tuple(det['bbox'])
                else:
                    # Format 2: separate x, y, width, height fields
                    bbox = (det['x'], det['y'], det['width'], det['height'])
                
                parsed.append(Detection(
                    class_name=det['class'],
                    bbox=bbox,
                    confidence=det['confidence']
                ))
        return parsed

    def _is_letter(self, class_name: str) -> bool:
        """Check if a class name represents a letter."""
        return (class_name.lower() in self.SPANISH_ALPHABET or
                class_name.upper() in [c.upper() for c in self.SPANISH_ALPHABET])

    def _calculate_legibilidad(self, letters: List[Detection], errors: List[Detection]) -> float:
        """
        Calculate legibility score (0-100).

        Factors:
        - Letter detection confidence
        - Completeness (all expected letters detected)
        - Order correctness
        - Impact of errors on readability
        """
        if not letters:
            return 0.0

        # Base score from average confidence
        confidence_score = np.mean([l.confidence for l in letters]) * 100

        # Completeness score
        completeness_score = 100.0
        if self.expected_word:
            expected_letters = list(self.expected_word)
            detected_letters = self._extract_word_from_detections(letters)
            completeness_score = (len(detected_letters) / len(expected_letters)) * 100

        # Order score (check if letters are in correct left-to-right order)
        order_score = self._calculate_order_score(letters)

        # Error impact (reduce score based on errors)
        error_penalty = 0
        for error in errors:
            if error.class_name in ['incomplete_loop', 'il']:
                error_penalty += 5  # High impact on legibility
            elif error.class_name in ['separation_from_line', 'sl']:
                error_penalty += 3
            elif error.class_name in ['out_of_bounds', 'ob']:
                error_penalty += 2
            elif error.class_name in ['small_char', 'sc']:
                error_penalty += 2

        # Weighted combination
        legibilidad = (
                              0.3 * confidence_score +
                              0.3 * completeness_score +
                              0.2 * order_score +
                              0.2 * 100
                      ) - error_penalty

        return max(0, min(100, legibilidad))

    def _calculate_grafia(self, letters: List[Detection], errors: List[Detection]) -> float:
        """
        Calculate graphology score (0-100).

        Factors:
        - Presence of incomplete loops
        - Letter formation quality (based on confidence)
        - Consistency in letter shapes
        """
        if not letters:
            return 0.0

        # Base score from letter confidence (higher confidence = better formed)
        base_score = np.mean([l.confidence for l in letters]) * 100

        # Penalty for incomplete loops (major graphology issue)
        incomplete_loops = sum(1 for e in errors if e.class_name in ['incomplete_loop', 'il'])
        loop_penalty = incomplete_loops * 10

        # Consistency in letter sizes (standard deviation)
        if len(letters) > 1:
            areas = [l.area for l in letters]
            cv = stats.variation(areas) if np.mean(areas) > 0 else 0
            consistency_score = max(0, 100 - cv * 100)
        else:
            consistency_score = 100

        # Shape quality estimation (using bbox aspect ratios)
        shape_scores = []
        for letter in letters:
            aspect_ratio = letter.bbox[2] / letter.bbox[3] if letter.bbox[3] > 0 else 1
            # Expected aspect ratios for common letters
            expected_ratios = {
                'i': 0.3, 'l': 0.3, 't': 0.6,
                'm': 1.5, 'w': 1.5,
                'o': 1.0, 'a': 0.8, 'e': 0.8
            }

            expected = expected_ratios.get(letter.class_name.lower(), 0.7)
            deviation = abs(aspect_ratio - expected) / expected
            shape_scores.append(max(0, 100 - deviation * 50))

        shape_quality = np.mean(shape_scores) if shape_scores else 100

        # Weighted combination
        grafia = (
                         0.4 * base_score +
                         0.3 * consistency_score +
                         0.3 * shape_quality
                 ) - loop_penalty

        return max(0, min(100, grafia))

    def _calculate_alineacion(self, letters: List[Detection], errors: List[Detection]) -> float:
        """
        Calculate alignment score (0-100).

        Factors:
        - Baseline alignment (using linear regression)
        - Separation from line errors
        - Out of bounds errors
        """
        if not letters:
            return 0.0

        # Calculate baseline alignment using bottom coordinates
        x_coords = [(l.x_min + l.x_max) / 2 for l in letters]
        y_bottoms = [l.y_max for l in letters]

        # Fit a line to the bottom points
        if len(letters) >= 2:
            X = np.array(x_coords).reshape(-1, 1)
            y = np.array(y_bottoms)

            reg = LinearRegression().fit(X, y)
            predictions = reg.predict(X)

            # Calculate R² score (how well letters align to a straight line)
            r2_score = reg.score(X, y)
            alignment_score = r2_score * 100
            
            # R² can be negative if the fit is worse than a horizontal line
            # In handwriting, we expect some variation, so let's use a base score
            if alignment_score < 0:
                alignment_score = 50  # Base score for imperfect alignment

            # Calculate mean absolute deviation from the line
            deviations = np.abs(y - predictions)
            mean_deviation = np.mean(deviations)
            deviation_penalty = min(50, mean_deviation * 2)
            alignment_score -= deviation_penalty
        else:
            alignment_score = 80  # Single letter, assume decent alignment

        # Penalties for specific errors
        separation_errors = sum(1 for e in errors if e.class_name in ['separation_from_line', 'sl'])
        out_of_bounds_errors = sum(1 for e in errors if e.class_name in ['out_of_bounds', 'ob'])

        error_penalty = separation_errors * 15 + out_of_bounds_errors * 10

        alineacion = alignment_score - error_penalty
        
        # Ensure minimum score if letters are detected
        if letters and alineacion < 10:
            alineacion = 10  # Minimum score for having letters

        return max(0, min(100, alineacion))

    def _calculate_tamano(self, letters: List[Detection], errors: List[Detection]) -> float:
        """
        Calculate size consistency score (0-100).

        Factors:
        - Size consistency among letters
        - Presence of small_char errors
        - Presence of out_of_bounds errors
        - Comparison with expected grid size
        """
        if not letters:
            return 0.0

        # Calculate size consistency
        heights = [l.bbox[3] for l in letters]
        widths = [l.bbox[2] for l in letters]

        # Coefficient of variation for heights and widths
        height_cv = stats.variation(heights) if len(heights) > 1 and np.mean(heights) > 0 else 0
        width_cv = stats.variation(widths) if len(widths) > 1 and np.mean(widths) > 0 else 0

        # Lower CV means more consistent sizes
        consistency_score = max(0, 100 - (height_cv + width_cv) * 50)

        # Compare with expected grid size
        mean_height = np.mean(heights)
        mean_width = np.mean(widths)

        expected_height = self.grid_size[1] * 0.7  # Letters should be ~70% of grid height
        expected_width = self.grid_size[0] * 0.5  # Letters should be ~50% of grid width

        height_ratio = mean_height / expected_height if expected_height > 0 else 1
        width_ratio = mean_width / expected_width if expected_width > 0 else 1

        # Penalize both too small and too large
        size_score = 100
        if height_ratio < 0.6:  # Too small
            size_score -= (0.6 - height_ratio) * 100
        elif height_ratio > 1.2:  # Too large
            size_score -= (height_ratio - 1.2) * 100
            
        # Ensure size_score doesn't go below 0
        size_score = max(0, size_score)

        # Error penalties
        small_char_errors = sum(1 for e in errors if e.class_name in ['small_char', 'sc'])
        out_of_bounds_errors = sum(1 for e in errors if e.class_name in ['out_of_bounds', 'ob'])

        error_penalty = small_char_errors * 15 + out_of_bounds_errors * 10

        # Weighted combination
        tamano = (
                         0.6 * consistency_score +
                         0.4 * size_score
                 ) - error_penalty

        return max(0, min(100, tamano))

    def _extract_word_from_detections(self, letters: List[Detection]) -> str:
        """Extract the word by ordering letters left to right."""
        sorted_letters = sorted(letters, key=lambda l: l.x_min)
        return ''.join([l.class_name.lower() for l in sorted_letters])

    def _calculate_order_score(self, letters: List[Detection]) -> float:
        """Calculate how well letters are ordered left to right."""
        if len(letters) <= 1:
            return 100.0

        x_positions = [(l.x_min + l.x_max) / 2 for l in letters]

        # Check if positions are monotonically increasing
        inversions = 0
        for i in range(len(x_positions) - 1):
            if x_positions[i] > x_positions[i + 1]:
                inversions += 1

        # Score based on number of inversions
        order_score = 100 * (1 - inversions / (len(letters) - 1))

        return order_score

    def get_detailed_report(self, detections: List[Dict]) -> Dict:
        """
        Generate a detailed report with metrics and additional insights.
        """
        # Handle both formats: direct list or dict with 'predictions' key
        if isinstance(detections, dict) and 'predictions' in detections:
            detections = detections['predictions']
            
        metrics = self.analyze(detections)
        det_objects = self._parse_detections(detections)

        letters = [d for d in det_objects if self._is_letter(d.class_name)]
        errors = [d for d in det_objects if d.class_name in self.ERROR_CLASSES]

        detected_word = self._extract_word_from_detections(letters)

        report = {
            'metrics': metrics,
            'detected_word': detected_word,
            'expected_word': self.expected_word,
            'letter_count': len(letters),
            'error_count': len(errors),
            'errors_by_type': defaultdict(int),
            'confidence_stats': {
                'mean': np.mean([l.confidence for l in letters]) if letters else 0,
                'std': np.std([l.confidence for l in letters]) if letters else 0,
                'min': min([l.confidence for l in letters]) if letters else 0,
                'max': max([l.confidence for l in letters]) if letters else 0
            }
        }

        # Count errors by type
        for error in errors:
            report['errors_by_type'][error.class_name] += 1

        return report