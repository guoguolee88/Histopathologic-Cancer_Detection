import tensorflow as tf

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def aug(images_or_image):
    # seq = iaa.Sequential([
    #     iaa.Affine(rotate=(-25, 25)),
    #     iaa.AdditiveGaussianNoise(scale=(10, 60)),
    #     iaa.Crop(percent=(0, 0.2))
    # ])

    # seq = iaa.Sequential([
    #     iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),  # crop and pad images
    #     iaa.AddToHueAndSaturation((-60, 60)),  # change their color
    #     iaa.ElasticTransformation(alpha=90, sigma=9),  # water-like effect
    #     iaa.CoarseDropout((0.01, 0.1), size_percent=0.01)  # set large image areas to zero
    # ], random_order=True)

    # seq = iaa.Sequential([
    #     iaa.Affine(
    #         scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
    #         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #         rotate=(-25, 25),
    #         shear=(-8, 8)
    #     ),
    #     # Strengthen or weaken the contrast in each image.
    #     iaa.ContrastNormalization((0.75, 1.5)),
    #     # iaa.AdditiveGaussianNoise(scale=(30, 90)),
    #     iaa.Fliplr(0.5),  # horizontally flip 50% of all images
    #     iaa.Flipud(0.2),  # vertically flip 20% of all images
    #     iaa.Crop(px=(40, 56)),  # crop images from each side by 0 to 16px (randomly chosen)
    #     # convert images into their superpixel representation
    #     iaa.OneOf([
    #         iaa.GaussianBlur((0, 0.5)),  # blur images with a sigma between 0 and 3.0
    #         iaa.AverageBlur(k=(3, 5)),
    #         # blur image using local means with kernel sizes between 2 and 7
    #         iaa.MedianBlur(k=(3, 5)),
    #         # blur image using local medians with kernel sizes between 2 and 7
    #     ]),
    #     iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),  # sharpen images
    # ], random_order=True)

    seq = iaa.Sequential([
        # # Applies either Fliplr or Flipud to images.
        # iaa.SomeOf(1, [
        #     iaa.Fliplr(1.0),
        #     iaa.Flipud(1.0)
        # ]),
        iaa.Fliplr(0.5), # horizontally flip 50% of the images
        iaa.Flipud(0.2),  # vertically flip 20% of all images
        # Rotates all images by 90 or 270 degrees.
        # Resizes all images afterwards to keep the size that they had before augmentation.
        # This may cause the images to look distorted.
        iaa.Sometimes(0.5, iaa.Rot90((1, 3))),
        # crop some of the images by 30-50% of their height/width
        iaa.Crop(percent=(0.3, 0.5)),
        iaa.Affine(
            scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            # rotate=(-45, 45),
            shear=(-8, 8)
        ),
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.85, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND channel.
        # This can change the color (not only brightness) of the pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),

        # iaa.Sometimes(0.5,
        #     iaa.Affine(
        #         scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
        #         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #         # rotate=(-45, 45),
        #         shear=(-8, 8)
        #     )),
        # Execute 0 to 3 of the following (less important) augmenters per
        # image. Don't execute all of them, as that would often be way too
        # strong.
        # iaa.SomeOf((0, 4),
        #    [
        #        # Small gaussian blur with random sigma between 0 and 0.5.
        #        # But we only blur about 50% of all images.
        #        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.3))),
        #        # # Sharpen each image, overlay the result with the original
        #        # # image using an alpha between 0 (no sharpening) and 1
        #        # # (full sharpening effect).
        #        # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
        #
        #        # # Same as sharpen, but for an embossing effect.
        #        # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
        #        # Strengthen or weaken the contrast in each image.
        #        iaa.ContrastNormalization((0.85, 1.5)),
        #        # Add gaussian noise.
        #        # For 50% of all images, we sample the noise once per pixel.
        #        # For the other 50% of all images, we sample the noise per pixel AND
        #        # channel. This can change the color (not only brightness) of the
        #        # pixels.
        #        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        #        # Make some images brighter and some darker.
        #        # In 20% of all cases, we sample the multiplier once per channel,
        #        # which can end up changing the color of the images.
        #        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        #        # Apply affine transformations to each image.
        #        # Scale/zoom them, translate/move them, rotate them and shear them.
        #     ], random_order=True)
    ], random_order=True)  # apply augmenters in random order

    # images_aug = [seq.augment_image(images_or_image) for _ in range(8)]

    images_aug = seq.augment_images(images_or_image)
    # images_aug = seq.augment_image(images_or_image)

    # print("Augmented batch:")
    # print("Augmented:")
    # ia.imshow(np.hstack(images_aug))

    return images_aug


# def get_aug_set():
#     sometimes = lambda aug: iaa.Sometimes(0.5, aug)
#     seq = iaa.Sequential(
#         [
#             # apply the following augmenters to most images
#             iaa.Fliplr(0.5),  # horizontally flip 50% of all images
#             iaa.Flipud(0.2),  # vertically flip 20% of all images
#             sometimes(iaa.Affine(
#                 scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
#                 # scale images to 80-120% of their size, individually per axis
#                 translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # translate by -20 to +20 percent (per axis)
#                 rotate=(-10, 10),  # rotate by -45 to +45 degrees
#                 shear=(-5, 5),  # shear by -16 to +16 degrees
#                 order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
#                 cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
#                 mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
#             )),
#             # execute 0 to 5 of the following (less important) augmenters per image
#             # don't execute all of them, as that would often be way too strong
#             iaa.SomeOf((0, 5),
#                        [
#                            sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
#                            # convert images into their superpixel representation
#                            iaa.OneOf([
#                                iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 3.0
#                                iaa.AverageBlur(k=(3, 5)),
#                                # blur image using local means with kernel sizes between 2 and 7
#                                iaa.MedianBlur(k=(3, 5)),
#                                # blur image using local medians with kernel sizes between 2 and 7
#                            ]),
#                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),  # sharpen images
#                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
#                            # search either for all edges or for directed edges,
#                            # blend the result with the original image using a blobby mask
#                            iaa.SimplexNoiseAlpha(iaa.OneOf([
#                                iaa.EdgeDetect(alpha=(0.5, 1.0)),
#                                iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
#                            ])),
#                            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255), per_channel=0.5),
#                            # add gaussian noise to images
#                            iaa.OneOf([
#                                iaa.Dropout((0.01, 0.05), per_channel=0.5),  # randomly remove up to 10% of the pixels
#                                iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
#                            ]),
#                            iaa.Invert(0.01, per_channel=True),  # invert color channels
#                            iaa.Add((-2, 2), per_channel=0.5),
#                            # change brightness of images (by -10 to 10 of original value)
#                            iaa.AddToHueAndSaturation((-1, 1)),  # change hue and saturation
#                            # either change the brightness of the whole image (sometimes
#                            # per channel) or change the brightness of subareas
#                            iaa.OneOf([
#                                iaa.Multiply((0.9, 1.1), per_channel=0.5),
#                                iaa.FrequencyNoiseAlpha(
#                                    exponent=(-1, 0),
#                                    first=iaa.Multiply((0.9, 1.1), per_channel=True),
#                                    second=iaa.ContrastNormalization((0.9, 1.1))
#                                )
#                            ]),
#                            sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
#                            # move pixels locally around (with random strengths)
#                            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
#                            # sometimes move parts of the image around
#                            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
#                        ],
#                        random_order=True
#                        )
#         ],
#         random_order=True
#     )
#
#     return seq