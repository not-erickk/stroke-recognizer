import io
import logging

import keras.saving
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from huggingface_hub import from_pretrained_keras
from PIL import Image

from utils import plot_ink
from utils.utils import detokenize, text_to_tokens, load_and_pad_img, scale_and_pad

model = tf.saved_model.load('../../models/small-p-cpu')
#model = from_pretrained_keras("Derendering/InkSight-Small-p")
#model = keras.saving.load_model('Derendering/InkSight-Small-p')
cf = model.signatures['serving_default']

prompt = "Derender the ink: q" # "Recognize and derender." or "Derender the ink: <text>"
#prompt = "Recognize and derender every stroke: q"



try:
    from google.colab import files
    in_colab = True
except ImportError:
    in_colab = False

if in_colab:
    uploaded = files.upload()
    input_image = Image.open(io.BytesIO(uploaded[list(uploaded.keys())[0]]))
else:
    file_path = '../../tests/inputs/m (1).jpg'
    input_image = Image.open(file_path)

image, _, _, _, _ = scale_and_pad(input_image)



input_text = tf.constant([prompt], dtype=tf.string)
image_encoded = tf.reshape(tf.io.encode_jpeg(np.array(image)[:, :, :3]), (1, 1))
output = cf(**{'input_text': input_text, 'image/encoded': image_encoded})

output_ink = detokenize(text_to_tokens(output['output_0'].numpy()[0][0].decode()))
fig, ax = plt.subplots()
plot_ink(output_ink, ax, input_image=load_and_pad_img(input_image))
plt.show()


