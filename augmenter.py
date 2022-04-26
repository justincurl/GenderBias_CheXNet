from imgaug import augmenters as iaa
import imgaug as ia

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

augmenter = iaa.Sequential(
    [
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )),
        sometimes(iaa.Crop(percent=(0, 0.1))),
        iaa.Fliplr(0.5),
        sometimes(iaa.GaussianBlur(sigma=(0, 3.0)))

    ],
    random_order=True,
)
