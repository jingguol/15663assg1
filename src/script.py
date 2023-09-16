import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy import interpolate

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 



## Read .tiff image and convert to double precision
filename = 'campus.tiff'
image = skimage.io.imread(filename)
print('Read image ' + filename)
print('Image is of shape', image.shape, ', with each pixel stored as', image.dtype)
# plt.imshow(image)
# plt.show()
image = image.astype(np.double)



## Linearization
## darkness 150, saturation 4095
darkness = 150.0
saturation = 4095.0
image = np.clip((image - darkness) / (saturation - darkness), 0.0, 1.0)
# plt.imshow(np.clip(image * 5.0, 0.0, 1.0), cmap='gray')
# plt.show()



## Identify Bayer pattern
## Uncomment to produce the image in the report
# image_1 = image[0::2, 0::2]     # topleft
# image_2 = image[0::2, 1::2]     # topright
# image_3 = image[1::2, 0::2]     # bottomleft
# image_4 = image[1::2, 1::2]     # bottomright
# fig = plt.figure()
# fig.add_subplot(2, 2, 1).set_title('grbg')
# image_r, image_g, image_b = image_2, image_1, image_3
# image_rgb = np.dstack((image_r * 2.39, image_g * 1.00, image_b * 1.60))
# plt.imshow(np.clip(image_rgb * 5.0, 0.0, 1.0))
# fig.add_subplot(2, 2, 2).set_title('rggb')
# image_r, image_g, image_b = image_1, image_2, image_4
# image_rgb = np.dstack((image_r * 2.39, image_g * 1.00, image_b * 1.60))
# plt.imshow(np.clip(image_rgb * 5.0, 0.0, 1.0))
# fig.add_subplot(2, 2, 3).set_title('bggr')
# image_r, image_g, image_b = image_4, image_2, image_1
# image_rgb = np.dstack((image_r * 2.39, image_g * 1.00, image_b * 1.60))
# plt.imshow(np.clip(image_rgb * 5.0, 0.0, 1.0))
# fig.add_subplot(2, 2, 4).set_title('gbrg')
# image_r, image_g, image_b = image_3, image_1, image_2
# image_rgb = np.dstack((image_r * 2.39, image_g * 1.00, image_b * 1.60))
# plt.imshow(np.clip(image_rgb * 5.0, 0.0, 1.0))
# plt.show()



## White balancing
## 'rggb' Bayer pattern
## White world balancing
r_max = np.max(image[0::2, 0::2])
g_max = np.max([image[0::2, 1::2], image[1::2, 0::2]])
b_max = np.max(image[1::2, 1::2])
# print(r_max, g_max, b_max)
image_whiteWorld = np.copy(image)
image_whiteWorld[0::2, 0::2] = image_whiteWorld[0::2, 0::2] * g_max / r_max
image_whiteWorld[1::2, 1::2] = image_whiteWorld[1::2, 1::2] * g_max / b_max

## Gray world balancing
r_avg = np.mean(image[0::2, 0::2])
g_avg = np.mean([image[0::2, 1::2], image[1::2, 0::2]])
b_avg = np.mean(image[1::2, 1::2])
image_grayWorld = np.copy(image)
image_grayWorld[0::2, 0::2] = image_grayWorld[0::2, 0::2] * g_avg / r_avg
image_grayWorld[1::2, 1::2] = image_grayWorld[1::2, 1::2] * g_avg / b_avg

## Camera preset balancing
r_multiplier = 2.394531
g_multiplier = 1.000000
b_multiplier = 1.597656
image_preset = np.copy(image)
image_preset[0::2, 0::2] = image_preset[0::2, 0::2] * r_multiplier
image_preset[0::2, 1::2] = image_preset[0::2, 1::2] * g_multiplier
image_preset[1::2, 0::2] = image_preset[1::2, 0::2] * g_multiplier
image_preset[1::2, 1::2] = image_preset[1::2, 1::2] * b_multiplier

## Manual white balancing
# image_rgb = np.dstack((image[0::2, 0::2] * 2.39, image[0::2, 1::2] * 1.00, image[1::2, 1::2] * 1.60))
# plt.imshow(np.clip(image_rgb * 5.0, 0.0, 1.0))
# patch = plt.ginput(4, timeout=0)
# patch = np.asarray(patch).astype(np.uint64)
# patch[:, [0, 1]] = patch[:, [1, 0]]
# patch_r = image[0::2, 0::2][patch[:, 0], patch[:, 1]]
# patch_g = image[1::2, 0::2][patch[:, 0], patch[:, 1]]
# patch_b = image[1::2, 1::2][patch[:, 0], patch[:, 1]]
# r_multiplier = 1.0 / np.mean(patch_r)
# g_multiplier = 1.0 / np.mean(patch_g)
# b_multiplier = 1.0 / np.mean(patch_b)
# image_manual = np.copy(image)
# image_manual[0::2, 0::2] = image_manual[0::2, 0::2] * r_multiplier
# image_manual[0::2, 1::2] = image_manual[0::2, 1::2] * g_multiplier
# image_manual[1::2, 0::2] = image_manual[1::2, 0::2] * g_multiplier
# image_manual[1::2, 1::2] = image_manual[1::2, 1::2] * b_multiplier

## Choose one of the three white balanced images
image = image_preset
# image_rgb = np.dstack((image[0::2, 0::2], image[0::2, 1::2], image[1::2, 1::2]))
# plt.imshow(np.clip(image_rgb * 5.0, 0.0, 1.0))
# plt.show()



## Demosaicing
## R channel
x = np.arange(0, image.shape[0], 2)
y = np.arange(0, image.shape[1], 2)
xx, yy = np.meshgrid(x, y)
z = image[xx, yy]
interp_r = interpolate.interp2d(x, y, z)
x = np.arange(0, image.shape[0], 1)
y = np.arange(0, image.shape[1], 1)
image_r = np.transpose(interp_r(x, y))

## G channel, topright
x = np.arange(0, image.shape[0], 2)
y = np.arange(1, image.shape[1], 2)
xx, yy = np.meshgrid(x, y)
z = image[xx, yy]
interp_g1 = interpolate.interp2d(x, y, z)
x = np.arange(0, image.shape[0], 1)
y = np.arange(0, image.shape[1], 1)
image_g1 = np.transpose(interp_g1(x, y))

## G channel, bottomleft
x = np.arange(1, image.shape[0], 2)
y = np.arange(0, image.shape[1], 2)
xx, yy = np.meshgrid(x, y)
z = image[xx, yy]
interp_g2 = interpolate.interp2d(x, y, z)
x = np.arange(0, image.shape[0], 1)
y = np.arange(0, image.shape[1], 1)
image_g2 = np.transpose(interp_g2(x, y))

## Actual G channel is average of two
image_g = (image_g1 + image_g2) * 0.5

## B channel
x = np.arange(1, image.shape[0], 2)
y = np.arange(1, image.shape[1], 2)
xx, yy = np.meshgrid(x, y)
z = image[xx, yy]
interp_b = interpolate.interp2d(x, y, z)
x = np.arange(0, image.shape[0], 1)
y = np.arange(0, image.shape[1], 1)
image_b = np.transpose(interp_b(x, y))

# image_rgb = np.dstack((image_r, image_g, image_b))
# plt.imshow(np.clip(image_rgb * 5.0, 0.0, 1.0))
# plt.show()



## Color space correction
M_sRGB_to_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], 
                          [0.2126729, 0.7151522, 0.0721750], 
                          [0.0193339, 0.1191920, 0.9503041]])
M_XYZ_to_cam = np.array([[0.6988, -0.1384, -0.0714], 
                         [-0.5631, 1.3410, 0.2447], 
                         [-0.1485, 0.2204, 0.7318]])
M_sRGB_to_cam = np.matmul(M_XYZ_to_cam, M_sRGB_to_XYZ)
M_sRGB_to_cam[0, :] = M_sRGB_to_cam[0, :] / np.sum(M_sRGB_to_cam[0, :])
M_sRGB_to_cam[1, :] = M_sRGB_to_cam[1, :] / np.sum(M_sRGB_to_cam[1, :])
M_sRGB_to_cam[2, :] = M_sRGB_to_cam[2, :] / np.sum(M_sRGB_to_cam[2, :])
M_cam_to_sRGB = np.linalg.inv(M_sRGB_to_cam)
image_rgb = np.zeros((image.shape[0], image.shape[1], 3))
image_rgb[:, :, 0] = M_cam_to_sRGB[0, 0] * image_r + M_cam_to_sRGB[0, 1] * image_g + M_cam_to_sRGB[0, 2] * image_b
image_rgb[:, :, 1] = M_cam_to_sRGB[1, 0] * image_r + M_cam_to_sRGB[1, 1] * image_g + M_cam_to_sRGB[1, 2] * image_b
image_rgb[:, :, 2] = M_cam_to_sRGB[2, 0] * image_r + M_cam_to_sRGB[2, 1] * image_g + M_cam_to_sRGB[2, 2] * image_b
# plt.imshow(np.clip(image_rgb * 5.0, 0.0, 1.0))
# plt.show()



## Brightness and gamma encoding
image_gray = skimage.color.rgb2gray(image_rgb)
gray_mean = np.mean(image_gray)
scale = 0.4 / gray_mean
print("Brightness scale: ", scale)
image_rgb = np.clip(image_rgb * scale, 0.0, 1.0)
# plt.imshow(image_rgb)
# plt.show()

b = image_rgb <= 0.0031308
image_rgb[b] = image_rgb[b] * 12.92
image_rgb[np.logical_not(b)] = (1 + 0.055) * np.power(image_rgb[np.logical_not(b)], 1/2.4) - 0.055
# plt.imshow(image_rgb)
# plt.show()



## Compression
filename1 = 'campus.png'
filename2 = 'campus.jpeg'
skimage.io.imsave(filename1, image_rgb)
skimage.io.imsave(filename2, image_rgb, quality=95)