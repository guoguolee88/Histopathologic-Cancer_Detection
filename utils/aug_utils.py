import imgaug as ia
from imgaug import augmenters as iaa


def aug(images):

    seq = iaa.Sequential([
        # # Applies either Fliplr or Flipud to images.
        # iaa.SomeOf(1, [
        #     iaa.Fliplr(0.5),
        #     iaa.Flipud(0.5)
        # ]),
        iaa.Fliplr(0.5), # horizontally flip 50% of the images
        iaa.Flipud(0.2),  # vertically flip 20% of all images

        # crop some of the images by 0-50% of their height/width
        iaa.Crop(percent=(0, 0.5)),

        # Rotates all images by 90 or 270 degrees.
        # Resizes all images afterwards to keep the size that they had before augmentation.
        # This may cause the images to look distorted.
        iaa.Sometimes(0.5, iaa.Rot90((1, 3))),

        # Small gaussian blur with random sigma between 0 and 0.3.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.3))),

        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.85, 1.5)),

        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND channel.
        # This can change the color (not only brightness) of the pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            # rotate=(-25, 25),
            shear=(-8, 8)
        ),
        # iaa.Sometimes(0.5,
        #     iaa.Affine(
        #         scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
        #         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #         # rotate=(-45, 45),
        #         shear=(-8, 8)
        #     )),

        iaa.Multiply(mul=(1. / 255), per_channel=True)

    ], random_order=False)  # apply augmenters in random order

    images_aug = seq.augment_images(images)

    # print("Augmented batch:")
    # print("Augmented:")
    # ia.imshow(np.hstack(images_aug))

    return images_aug
