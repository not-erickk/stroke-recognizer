# @title Utils
import tensorflow as tf
import tensorflow_text
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from xml.dom import minidom
import gdown
import os
import matplotlib.animation as animation
import copy
from PIL import ImageEnhance, Image, ImageDraw
import colorsys
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patheffects import withStroke
import random
import warnings
import re
import time
import io
import pytesseract
from tqdm import tqdm
from copy import deepcopy
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from utils import Stroke, Ink


def get_box(data_idx, data):
    min_x = left = data['left'][data_idx]
    min_y = top = data['top'][data_idx]
    width = data['width'][data_idx]
    angle = 0
    height = data['height'][data_idx]
    angle = angle / 180.0 * np.pi
    s_x = left + np.cos(angle) * width
    s_y = top - np.sin(angle) * width
    f_x = (
        left + np.sin(angle) * height
    )
    f_y = top + np.cos(angle) * height
    max_x = (
        left
        + np.cos(angle) * width
        + np.sin(angle) * height
    )
    max_y = (
        top
        - np.sin(angle) * width
        + np.cos(angle) * height
    )
    return min_x, min_y, s_x, s_y, f_x, f_y, max_x, max_y

def rotate_crop_scale_and_pad(original, data_idx, data, pad_black=True):
    angle = 0
    height = data['height'][data_idx]
    width = data['width'][data_idx]
    min_x, min_y, s_x, s_y, f_x, f_y, _, _ = get_box(data_idx, data)
    max_x = min_x + width
    max_y = min_y + height

    output = original.rotate(angle, center=(min_x, min_y))
    crop = output.crop((min_x, min_y, max_x, max_y))

    ratio = min(224 / crop.width, 224 / crop.height)
    new_crop = crop.resize((int(crop.width * ratio), int(crop.height * ratio)))
    new_crop_np = np.array(new_crop)

    pixel_1 = new_crop_np[1, 1]
    pixel_2 = new_crop_np[1, new_crop_np.shape[-1] - 1]
    pixel_3 = new_crop_np[new_crop_np.shape[0] - 1, 1]
    pixel_4 = new_crop_np[new_crop_np.shape[0] - 1, new_crop_np.shape[-1] - 1]
    avg = np.rint(np.mean([pixel_1, pixel_2, pixel_3, pixel_4], axis=0)).astype(
        np.uint8
    )

    color = tuple(avg) if not pad_black else (0, 0, 0)
    new_image = Image.new(new_crop.mode, (224, 224), color)
    dx = (224 - new_crop.width) // 2
    dy = (224 - new_crop.height) // 2
    new_image.paste(new_crop, (dx, dy))
    return new_image, ratio, dx, dy, min_x, min_y, angle, crop


def extract_fullpage(img_path, option="tesseract"):
    ret_imgs = []
    img_info = []
    img_bbox = []
    input_image = Image.open(io.BytesIO(img_path))
    if option == "tesseract":
        data = pytesseract.image_to_data(input_image, output_type=pytesseract.Output.DICT)
        for i in tqdm(range(len(data['text']))):
            if data['text'][i].strip() != '':  # Filters out empty text results
                new_image, ratio, dx, dy, min_x, min_y, angle, _ = (
                    rotate_crop_scale_and_pad(input_image, i, data, pad_black=True)
                )
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                ret_imgs.append(new_image)
                img_info.append((ratio, dx, dy, min_x, min_y, angle))
                img_bbox.append((x, y, w, h))
    elif option == "doctr":
        doc = DocumentFile.from_images(img_path)
        predictor = ocr_predictor(pretrained=True)
        print("doctr predictor loaded.")
        result = predictor(doc)

        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        if word.value.strip() != '':
                            coords = word.geometry
                            x0, y0 = int(coords[0][0] * input_image.width), int(coords[0][1] * input_image.height)
                            x1, y1 = int(coords[1][0] * input_image.width), int(coords[1][1] * input_image.height)
                            w, h = x1 - x0, y1 - y0

                            w_expand = w * 0.1
                            h_expand = h * 0.1

                            x0 = max(0, x0 - w_expand)
                            y0 = max(0, y0 - h_expand)
                            x1 = min(input_image.width, x1 + w_expand)
                            y1 = min(input_image.height, y1 + h_expand)

                            w = x1 - x0
                            h = y1 - y0

                            x0, y0, w, h = map(int, [x0, y0, w, h])

                            # Create a mock data dictionary similar to tesseract's output
                            mock_data = {
                                'left': [x0],
                                'top': [y0],
                                'width': [w],
                                'height': [h],
                                'conf': [1.0],  # doctr doesn't provide confidence scores in the same way
                                'text': [word.value]
                            }

                            # Use the same processing function as tesseract
                            new_image, ratio, dx, dy, min_x, min_y, angle, crop = (
                                rotate_crop_scale_and_pad(input_image, 0, mock_data, pad_black=True)
                            )

                            ret_imgs.append(new_image)
                            img_info.append((ratio, dx, dy, min_x, min_y, angle))
                            img_bbox.append((x0, y0, w, h))

    print('\nFinal length: ', len(ret_imgs))

    print('\nFinal length: ', len(ret_imgs))

    # Draw the bboxes
    image = deepcopy(input_image)
    draw = ImageDraw.Draw(image)
    for bx in img_bbox:
        x, y, w, h = bx
        draw.rectangle([x, y, x + w, y + h], outline='red', width=2)

    return ret_imgs, img_info, image


warnings.filterwarnings("ignore")

def text_to_tokens(text) -> list[int]:
    pattern = r"<ink_token_(\d+)>"
    matches = re.findall(pattern, text)
    return [int(tok) for tok in matches]

def detokenize(tokens: list[int]) -> list[list[tuple[float, float]]]:
    coordinate_length = 224
    num_token_per_dimension = coordinate_length + 1
    vocabulary_size = num_token_per_dimension * 2 + 1
    start_token = num_token_per_dimension * 2

    if any([t < 0 or t >= vocabulary_size for t in tokens]):
        raise ValueError(
            f"Ink token indices should be between 0 and {vocabulary_size}"
        )
    idx = 0
    res = []
    current_stroke_tokens = []

    while idx < len(tokens):
        token = tokens[idx]
        if token == start_token:
            if current_stroke_tokens:
                res.append(current_stroke_tokens)
            current_stroke_tokens = []
            idx += 1
        else:
            if idx + 1 < len(tokens) and (tokens[idx + 1] != start_token):
                # Read in x and y coordinates.
                x = tokens[idx]
                y = tokens[idx + 1] - num_token_per_dimension
                # If the coordinates are valid, add them to detokenization ink.
                if (0 <= x <= coordinate_length) and (0 <= y <= coordinate_length):
                    current_stroke_tokens.append([x, y])
                idx += 2
            # If y doesn't exist or y is start_token, then skip this x.
            else:
                idx += 1
    if current_stroke_tokens:
        res.append(current_stroke_tokens)

    strokes = []
    for stroke in res:
        stroke_points = []
        for point in stroke:
            x, y = point
            stroke_points.append((x, y))
        strokes.append(Stroke(stroke_points))
    return Ink(strokes)

def load_and_pad_img(image):
    width, height = image.size
    ratio = min(224 / width, 224 / height)
    image = image.resize((int(width * ratio), int(height * ratio)))
    width, height = image.size
    if height < 224:
        # If width is shorter than height pad top and bottom.
        top_padding = (224 - height) // 2
        bottom_padding = 224 - height - top_padding
        padded_image = Image.new('RGB', (width, 224), (255, 255, 255))
        padded_image.paste(image, (0, top_padding))
    else:
        # Otherwise pad left and right.
        left_padding = (224 - width) // 2
        right_padding = 224 - width - left_padding
        padded_image = Image.new('RGB', (224, height), (255, 255, 255))
        padded_image.paste(image, (left_padding, 0))
    return padded_image

def scale_and_pad(original, pad_black=True):
    ratio = min(224 / original.width, 224 / original.height)
    original_np = np.array(original)
    new_crop = original.resize((int(original.width * ratio), int(original.height * ratio)))
    pixel_1 = original_np[1, 1]
    pixel_2 = original_np[1, original_np.shape[-1]-1]
    pixel_3 = original_np[original_np.shape[0]-1, 1]
    pixel_4 = original_np[original_np.shape[0]-1, original_np.shape[-1]-1]
    avg = np.rint(np.mean([pixel_1, pixel_2, pixel_3, pixel_4], axis=0)).astype(np.uint8)

    color = tuple(avg) if not pad_black else (0, 0, 0)
    new_image = Image.new(new_crop.mode, (224, 224), color)
    dx = (224 - new_crop.width) // 2
    dy = (224 - new_crop.height) // 2
    new_image.paste(new_crop, (dx, dy))
    return new_image, ratio, dx, dy, new_crop

def encode_images_in_batches(images, batch_size=32):
    def encode_image(image):
        image_np = np.array(image)[:, :, :3]
        encoded_jpeg = tf.io.encode_jpeg(image_np)
        return tf.reshape(encoded_jpeg, (1,)), image_np

    encoded_batches = []
    original_batches = []

    num_batches = len(images) // batch_size + (1 if len(images) % batch_size != 0 else 0)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(images))
        current_batch = images[start_idx:end_idx]

        encoded_batch = []
        original_batch = []
        for image in current_batch:
            encoded, original = encode_image(image)
            encoded_batch.append(encoded)
            original_batch.append(original)

        encoded_batches.append(tf.stack(encoded_batch))
        original_batches.append(np.stack(original_batch))

    return encoded_batches, original_batches

def unpad_unscale_unrotate_uncrop(ink, ratio, dx, dy, min_x, min_y, angle):
    transformed_strokes = []

    for stroke in ink:
        transformed_points = []
        for point in stroke:
            x_transformed = (point[0] - dx) / ratio
            y_transformed = (point[1] - dy) / ratio

            x_final = x_transformed + min_x
            y_final = y_transformed + min_y

            transformed_points.append((x_final, y_final))

        transformed_strokes.append(Stroke(transformed_points))

    transformed_ink = Ink(transformed_strokes)
    return transformed_ink