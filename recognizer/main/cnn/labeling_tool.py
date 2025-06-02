#!/usr/bin/env python3
"""
Manual labeling tool for handwriting error dataset creation.
Displays character images and allows scoring errors from 1-10.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Error categories for handwriting analysis
ERROR_CATEGORIES = [
    "incomplete_strokes",  # Missing parts of the letter
    "incorrect_letter",  # Wrong letter shape
    "dysgraphia",  # Motor control issues
    "size_inconsistency",  # Letter size problems
    "spacing_issues",  # Poor character spacing
    "line_alignment",  # Writing above/below line
    "slant_variation",  # Inconsistent letter angles
    "pressure_irregularity",  # Uneven stroke pressure
    "proportion_errors",  # Wrong part proportions
    "closure_problems"  # Unclosed shapes
]


class LabelingSession:
    """Manages the labeling workflow and data persistence"""

    def __init__(self, images_dir: str, output_csv: str = "labels.csv"):
        self.images_dir = Path(images_dir)
        self.output_csv = Path(output_csv)
        self.current_scores = [5] * len(ERROR_CATEGORIES)  # Default middle scores
        self.current_char = ""

        # Load existing labels or create new dataframe
        if self.output_csv.exists():
            self.labels_df = pd.read_csv(self.output_csv)
            self.labeled_files = set(self.labels_df['filename'].values)
        else:
            columns = ['filename', 'character'] + ERROR_CATEGORIES
            self.labels_df = pd.DataFrame(columns=columns)
            self.labeled_files = set()

        # Get all image files
        self.image_files = sorted([
            f for f in self.images_dir.glob("*.png")
            if f.name not in self.labeled_files
        ])

        self.current_index = 0
        self.session_labels = []  # Buffer for current session

    def get_current_image(self) -> Tuple[np.ndarray, str]:
        """Load and preprocess current image"""
        if self.current_index >= len(self.image_files):
            return None, None

        img_path = self.image_files[self.current_index]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        # Resize for consistent display (maintain aspect ratio)
        height, width = img.shape
        target_size = 200
        if height > width:
            new_height = target_size
            new_width = int(width * (target_size / height))
        else:
            new_width = target_size
            new_height = int(height * (target_size / width))

        img_resized = cv2.resize(img, (new_width, new_height))

        # Add padding to center in square canvas
        canvas = np.ones((target_size, target_size), dtype=np.uint8) * 255
        y_offset = (target_size - new_height) // 2
        x_offset = (target_size - new_width) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img_resized

        return canvas, img_path.name

    def save_current_label(self):
        """Save current image labels to buffer"""
        if self.current_index >= len(self.image_files):
            return

        filename = self.image_files[self.current_index].name
        label_data = {
            'filename': filename,
            'character': self.current_char,
            **{cat: score for cat, score in zip(ERROR_CATEGORIES, self.current_scores)}
        }
        self.session_labels.append(label_data)

    def save_session(self):
        """Persist all session labels to CSV"""
        if not self.session_labels:
            return

        new_df = pd.DataFrame(self.session_labels)
        self.labels_df = pd.concat([self.labels_df, new_df], ignore_index=True)
        self.labels_df.to_csv(self.output_csv, index=False)

        # Create backup
        backup_name = f"labels_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.labels_df.to_csv(self.output_csv.parent / backup_name, index=False)

        print(f"Saved {len(self.session_labels)} labels. Total: {len(self.labels_df)}")
        self.session_labels.clear()


def create_display_image(img: np.ndarray, scores: List[int], char: str, filename: str) -> np.ndarray:
    """Create annotated display with image and scoring interface"""
    # Create larger canvas for UI
    display = np.ones((600, 800, 3), dtype=np.uint8) * 240

    # Convert grayscale image to BGR and place it
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    display[50:250, 50:250] = img_bgr

    # Title and filename
    cv2.putText(display, f"Character: '{char}' | File: {filename}",
                (300, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Instructions
    instructions = [
        "Keys: 0-9 to set score | Enter to confirm | ESC to exit",
        "Space: next category | Tab: previous | C: set character"
    ]
    for i, text in enumerate(instructions):
        cv2.putText(display, text, (300, 140 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # Error categories with scores
    y_start = 300
    for i, (category, score) in enumerate(zip(ERROR_CATEGORIES, scores)):
        y_pos = y_start + i * 25

        # Category name
        cv2.putText(display, f"{i + 1}. {category}:",
                    (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Score bar
        bar_x = 350
        bar_width = 300
        bar_height = 15

        # Background bar
        cv2.rectangle(display, (bar_x, y_pos - 12),
                      (bar_x + bar_width, y_pos - 12 + bar_height),
                      (200, 200, 200), -1)

        # Score fill (color gradient from green to red)
        fill_width = int((score / 10) * bar_width)
        color = get_score_color(score)
        cv2.rectangle(display, (bar_x, y_pos - 12),
                      (bar_x + fill_width, y_pos - 12 + bar_height),
                      color, -1)

        # Score text
        cv2.putText(display, str(score),
                    (bar_x + bar_width + 20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Progress indicator
    cv2.putText(display, "Press ENTER to save and continue",
                (250, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)

    return display


def get_score_color(score: int) -> Tuple[int, int, int]:
    """Return BGR color based on score (1=good/green, 10=bad/red)"""
    # Interpolate between green and red
    ratio = (score - 1) / 9
    green = int(255 * (1 - ratio))
    red = int(255 * ratio)
    return (0, green, red)


def main():
    """Run the labeling tool"""
    import argparse
    parser = argparse.ArgumentParser(description="Manual dataset labeling tool")
    parser.add_argument("images_dir", help="Directory containing character images")
    parser.add_argument("-o", "--output", default="labels.csv", help="Output CSV file")
    args = parser.parse_args()

    # Initialize session
    session = LabelingSession(args.images_dir, args.output)

    if not session.image_files:
        print("No unlabeled images found!")
        return

    print(f"Found {len(session.image_files)} images to label")
    print(f"Previously labeled: {len(session.labeled_files)}")

    # UI state
    current_category = 0

    while session.current_index < len(session.image_files):
        # Load current image
        img, filename = session.get_current_image()
        if img is None:
            break

        # Main labeling loop for current image
        confirmed = False
        while not confirmed:
            # Create display
            display = create_display_image(img, session.current_scores,
                                           session.current_char, filename)

            # Highlight current category
            y_pos = 300 + current_category * 25
            cv2.rectangle(display, (40, y_pos - 15), (340, y_pos + 5),
                          (255, 200, 0), 2)

            cv2.imshow("Labeling Tool", display)

            # Handle keyboard input
            key = cv2.waitKey(0) & 0xFF

            if key == 27:  # ESC - exit
                session.save_session()
                break
            elif key == 13:  # Enter - confirm and next
                if session.current_char:  # Must have character set
                    session.save_current_label()
                    session.current_index += 1
                    session.current_scores = [5] * len(ERROR_CATEGORIES)  # Reset
                    session.current_char = ""
                    current_category = 0
                    confirmed = True

                    # Auto-save every 10 images
                    if len(session.session_labels) % 10 == 0:
                        session.save_session()
            elif key == 32:  # Space - next category
                current_category = (current_category + 1) % len(ERROR_CATEGORIES)
            elif key == 9:  # Tab - previous category
                current_category = (current_category - 1) % len(ERROR_CATEGORIES)
            elif ord('0') <= key <= ord('9'):  # Number keys for scores
                score = 10 if key == ord('0') else key - ord('0')
                session.current_scores[current_category] = score
            elif key == ord('c') or key == ord('C'):  # Set character
                cv2.destroyAllWindows()
                char = input("Enter the character for this image: ").strip()
                if char:
                    session.current_char = char[0]  # Take first character only

        if key == 27:  # ESC was pressed
            break

    # Final save
    session.save_session()
    cv2.destroyAllWindows()

    print("\nLabeling complete!")
    print(f"Total images labeled: {len(session.labels_df)}")
    print(f"Output saved to: {session.output_csv}")


if __name__ == "__main__":
    main()