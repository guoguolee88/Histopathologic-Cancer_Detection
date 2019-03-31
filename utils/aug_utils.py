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

        # only (on average) 70 percent of all images will be cropped.
        iaa.Sometimes(0.7, iaa.Crop(percent=(0, 0.5))),

        # Rotates only (on average) 70 percent of all images by 90, 180 or 270 degrees.
        # Resizes all images afterwards to keep the size that they had before augmentation.
        # This may cause the images to look distorted.
        iaa.Sometimes(0.7, iaa.Rot90((1, 3))),

        # Apply affine transformations to each (on average) 70 percent of all images.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        # iaa.Affine(
        #     scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
        #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #     # rotate=(-25, 25),
        #     shear=(-8, 8)
        # ),
        iaa.Sometimes(0.7,
            iaa.Affine(
                scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                # rotate=(-45, 45),
                shear=(-8, 8)
            )),

        # Execute 0 to 3 of the following (less important) augmenters per image.
        # Don't execute all of them, as that would often be way too strong.
        iaa.SomeOf((0, 3),
            [
                # Small gaussian blur with random sigma between 0 and 0.5.
                # But we only blur about 50% of all images.
                iaa.Sometimes(0.5,
                              iaa.GaussianBlur(sigma=(0, 0.5))),

                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                # Same as sharpen, but for an embossing effect.
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                # Search in some images either for all edges or for
                # directed edges. These edges are then marked in a black
                # and white image and overlayed with the original image
                # using an alpha of 0 to 0.7.
                iaa.Sometimes(0.5,
                              (iaa.OneOf(
                                  [
                                      iaa.EdgeDetect(alpha=(0, 0.7)),
                                      iaa.DirectedEdgeDetect(
                                          alpha=(0, 0.7), direction=(0.0, 1.0)
                                      ),
                                  ]
                              ))),

                # increases/decreases hue and saturation by random values.
                iaa.AddToHueAndSaturation((-20, 20), per_channel=True),  # change their color

                # Strengthen or weaken the contrast in each image.
                iaa.ContrastNormalization((0.75, 1.5)),

                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
                iaa.Grayscale(alpha=(0.0, 1.0)),

                # In some images move pixels locally around (with random strengths).
                iaa.Sometimes(0.5,
                              iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),

                # Add gaussian noise.
                # For 50% of all images, we sample the noise once per pixel.
                # For the other 50% of all images, we sample the noise per pixel AND channel.
                # This can change the color (not only brightness) of the pixels.
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            ], random_order=True),

        # Make some images brighter and some darker.
        # normalize
        iaa.Multiply(mul=(1. / 255), per_channel=True)

    ], random_order=False)  # apply augmenters in random order

    images_aug = seq.augment_images(images)

    # print("Augmented batch:")
    # print("Augmented:")
    # ia.imshow(np.hstack(images_aug))

    return images_aug