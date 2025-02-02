from imgaug import augmenters as iaa
import imgaug as ia

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

augmenter = iaa.Sequential(
    [
        sometimes(iaa.Crop(percent=(0, 0.1))),
        iaa.Fliplr(0.5),
        # Blur each image with varying strength using gaussian blur (sigma between 0 and 3.0),
        sometimes(iaa.GaussianBlur(sigma=(0, 3.0))),
        # change size
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        ))
    ],
    random_order=True,
)