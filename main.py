import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import bit_stream_decoder
import byte_stream_generator
import channel_restorer
import histogram_generator
import huffman_code_decode_generator
import image_compressor

file_name = 'images/img.png'
original_image = cv2.imread(file_name)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Plot original image
plt.figure(figsize=(20, 10))
plt.imshow(original_image)

# Extracting the red screen
red_scale_image = original_image[:, :, 0]

# Plot original image
plt.figure(figsize=(15, 10))
plt.imshow(red_scale_image, cmap='gray', vmin=0, vmax=255)

# Extracting the green screen
green_scale_image = original_image[:, :, 1]

# Plot original image
plt.figure(figsize=(15, 10))
plt.imshow(green_scale_image, cmap='gray', vmin=0, vmax=255)

# Extracting the blue screen
blue_scale_image = original_image[:, :, 2]

# Plot original image
plt.figure(figsize=(15, 10))
plt.imshow(blue_scale_image, cmap='gray', vmin=0, vmax=255)

red_channel_histogram_array = histogram_generator.histogram_array_generator(red_scale_image)
green_channel_histogram_array = histogram_generator.histogram_array_generator(green_scale_image)
blue_channel_histogram_array = histogram_generator.histogram_array_generator(blue_scale_image)

red_channel_probability_distribution = histogram_generator.probability_distribution_generator(
    red_channel_histogram_array, 800 * 1200)
green_channel_probability_distribution = histogram_generator.probability_distribution_generator(
    green_channel_histogram_array, 800 * 1200)
blue_channel_probability_distribution = histogram_generator.probability_distribution_generator(
    blue_channel_histogram_array, 800 * 1200)



red_channel_probability_distribution['separator'] = 0
red_huffman_coding = huffman_code_decode_generator.Huffman_Coding(red_channel_probability_distribution)
red_coded_pixels, red_reverse_coded_pixels = red_huffman_coding.compress()

green_channel_probability_distribution['separator'] = 0
green_huffman_coding = huffman_code_decode_generator.Huffman_Coding(green_channel_probability_distribution)
green_coded_pixels, green_reverse_coded_pixels = green_huffman_coding.compress()

blue_huffman_coding = huffman_code_decode_generator.Huffman_Coding(blue_channel_probability_distribution)
blue_coded_pixels, blue_reverse_coded_pixels = blue_huffman_coding.compress()
with open('codes/red_channel_codes.json', 'w') as fp:
    json.dump(red_coded_pixels, fp)
with open('decodes/red_channel_decodes.json', 'w') as fp:
    json.dump(red_reverse_coded_pixels, fp)
with open('codes/green_channel_codes.json', 'w') as fp:
    json.dump(green_coded_pixels, fp)
with open('decodes/green_channel_decodes.json', 'w') as fp:
    json.dump(green_reverse_coded_pixels, fp)
with open('codes/blue_channel_codes.json', 'w') as fp:
    json.dump(blue_coded_pixels, fp)
with open('decodes/blue_channel_decodes.json', 'w') as fp:
    json.dump(blue_reverse_coded_pixels, fp)
red_channel_compressed_image = image_compressor.compressor(red_scale_image, red_coded_pixels)
green_channel_compressed_image = image_compressor.compressor(green_scale_image, green_coded_pixels)
blue_channel_compressed_image = image_compressor.compressor(blue_scale_image, blue_coded_pixels)
bit_stream = byte_stream_generator.byte_stream(red_channel_compressed_image, green_channel_compressed_image,
                                               blue_channel_compressed_image, red_coded_pixels['separator'],
                                               green_coded_pixels['separator'])
print('Compression ratio:', (len(bit_stream) / (red_scale_image.shape[0] * red_scale_image.shape[1] * 3 * 8)))

with open('bit_stream.txt', 'w') as fp:
    fp.write(bit_stream)

# image decompression

red_channel_decoder = json.load(open('./decodes/red_channel_decodes.json', 'r'))
green_channel_decoder = json.load(open('./decodes/green_channel_decodes.json', 'r'))
blue_channel_decoder = json.load(open('./decodes/blue_channel_decodes.json', 'r'))

with open('bit_stream.txt', 'r') as fr:
    bit_stream = fr.read()

pixel_stream = bit_stream_decoder.decoder(bit_stream, red_channel_decoder, green_channel_decoder, blue_channel_decoder,
                                          file_name)

with open('image_pixel_stream.txt', 'w') as fr:
    fr.write(str(pixel_stream))

# image restoring
with open('image_pixel_stream.txt', 'r') as fr:
    pixel_stream = fr.read()

pixel_stream = pixel_stream.replace('[', '')
pixel_stream = pixel_stream.replace(']', '')
pixel_stream = pixel_stream.split(', ')
pixel_stream = [int(pixel) for pixel in pixel_stream]

red_channel_pixel_stream = pixel_stream[:int(len(pixel_stream) / 3)]
green_channel_pixel_stream = pixel_stream[int(len(pixel_stream) / 3):int((2 * len(pixel_stream)) / 3)]
blue_channel_pixel_stream = pixel_stream[int((2 * len(pixel_stream)) / 3):int(len(pixel_stream))]

original_image = cv2.imread(file_name)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

red_channel_image = np.reshape(red_channel_pixel_stream, (original_image.shape[0], original_image.shape[1]))

plt.figure(figsize=(20, 10))
plt.imshow(red_channel_image, cmap='gray', vmin=0, vmax=255)

green_channel_image = np.reshape(green_channel_pixel_stream, (original_image.shape[0], original_image.shape[1]))

plt.figure(figsize=(20, 10))
plt.imshow(green_channel_image, cmap='gray', vmin=0, vmax=255)

blue_channel_image = np.reshape(blue_channel_pixel_stream, (original_image.shape[0], original_image.shape[1]))

plt.figure(figsize=(20, 10))
plt.imshow(blue_channel_image, cmap='gray', vmin=0, vmax=255)

red_channel_loss = original_image[:, :, 0] - red_channel_image
green_channel_loss = original_image[:, :, 1] - green_channel_image
blue_channel_loss = original_image[:, :, 2] - blue_channel_image

total_loss = np.sum(red_channel_loss) + np.sum(green_channel_loss) + np.sum(blue_channel_loss)
print('Total loss (accross all red, green and blue channels):', total_loss)

restored_image = channel_restorer.image_restorer(red_channel_image, green_channel_image, blue_channel_image)

print('Original image dimensions:', np.array(original_image).shape)
print('Restored image dimensions:', np.array(restored_image).shape)

fig = plt.figure(frameon=False, figsize=(20, 10))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(restored_image, aspect='auto')


fig.savefig('out.png', dpi=150)
