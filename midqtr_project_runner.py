"""
DSC 20 Mid-Quarter Project Runner (cv2)
"""
# pylint: disable = E1101

import cv2
import numpy as np
from midqtr_project import (
    RGBImage,
    ImageProcessing as IP,
    ImageKNNClassifier,
)

def img_read(path):
    """
    Read the image with given `path` to a RGBImage instance.
    """
    mat = cv2.imread(path).transpose(2, 0, 1).tolist()
    mat.reverse()  # convert BGR (cv2 default behavior) to RGB
    return RGBImage(mat)


def img_save(path, image):
    """
    Save a RGBImage instance (`image`) as a image file with given `path`.
    """
    mat = np.stack(list(reversed(image.get_pixels()))).transpose(1, 2, 0)
    cv2.imwrite(path, mat)


def create_random_pixels(low, high, nrows, ncols):
    """
    Create a random pixels matrix with dimensions of
    3 (channels) x `nrows` x `ncols`, and fill in integer
    values between `low` and `high` (both exclusive).
    """
    return np.random.randint(low, high + 1, (3, nrows, ncols)).tolist()


def pixels_example():
    """
    An example of the 3-dimensional pixels matrix (3 x 5 x 10).
    """
    return [
        [
            # channel 0: red (5 rows x 10 columns)
            [206, 138, 253, 211, 102, 194, 188, 188, 120, 231],
            [204, 208, 220, 214, 203, 165, 249, 225, 198, 185],
            [113, 196, 133, 235, 173, 179, 252, 105, 214, 238],
            [152, 156, 143, 114, 166, 132, 106, 115, 116, 177],
            [231, 193, 123, 154, 184, 242, 226, 155, 222, 223],
        ],
        [
            # channel 1: green (5 rows x 10 columns)
            [214, 190, 173, 141, 248, 189, 105, 193, 125, 122],
            [209, 136, 131, 187, 177, 186, 239, 222, 175, 152],
            [239, 236, 177, 243, 183, 192, 114, 211, 147, 192],
            [168, 119, 120, 182, 190, 108, 181, 219, 198, 127],
            [251, 222, 205, 102, 104, 217, 234, 196, 131, 127],
        ],
        [
            # channel 2: blue (5 rows x 10 columns)
            [233, 188, 214, 175, 152, 174, 235, 174, 234, 149],
            [163, 169, 131, 209, 232, 180, 238, 224, 152, 214],
            [137, 135, 181, 146, 243, 210, 236, 107, 193, 200],
            [230, 233, 206, 227, 150, 131, 177, 187, 143, 150],
            [117, 188, 127, 166, 134, 219, 241, 108, 217, 202],
        ],
    ]


def image_processing_test_examples():
    """
    Examples of image processing methods tests using real images.
    """
    # read image
    dsc20_img = img_read("img/dsc20.png")

    # negate and save
    negative_dsc20_img = IP.negate(dsc20_img)
    img_save("img/out/dsc20_negate.png", negative_dsc20_img)

    # chroma key with a background image and save
    bg_img = img_read("img/blue_gradient.png")
    chroma_white_dsc20_img = IP.chroma_key(dsc20_img, bg_img, (255, 255, 255))
    img_save("img/out/dsc20_chroma_white.png", chroma_white_dsc20_img)


def knn_test_examples():
    """
    Examples of KNN classifier tests.
    """
    # make random training data (type: List[Tuple[RGBImage, str]])
    train = []
    # create training images with low intensity values
    train.extend(
        (RGBImage(create_random_pixels(0, 75, 300, 300)), "low")
        for _ in range(20)
    )
    # create training images with high intensity values
    train.extend(
        (RGBImage(create_random_pixels(180, 255, 300, 300)), "high")
        for _ in range(20)
    )

    # initialize and fit the classifier
    knn = ImageKNNClassifier(5)
    knn.fit(train)

    # should be "low"
    print(knn.predict(RGBImage(create_random_pixels(0, 75, 300, 300))))
    # can be either "low" or "high"
    print(knn.predict(RGBImage(create_random_pixels(75, 180, 300, 300))))
    # should be "high"
    print(knn.predict(RGBImage(create_random_pixels(180, 255, 300, 300))))


def part_1_tests():
    """
    Tests for RGBImage.
    
    >>> erwin = img_read("img/strangethingtemplate.png")
    >>> erwin.size()
    (1215, 720)
    >>> RGBImage(erwin.get_pixels()).size()
    (1215, 720)
    >>> img_save('img/out/strangecopy.png', RGBImage(erwin.get_pixels()))
    >>> px = erwin.get_pixels()
    >>> px[0][0][0:719] = [0 for x in range(719)]
    >>> img_save('img/out/strangedeeptest1.png', erwin)
    >>> img_save('img/out/strangedeeptest2.png', RGBImage(px))
    >>> eren = erwin.copy()
    >>> eren.pixels[0][0][0:719] = [0 for x in range(719)]
    >>> img_save('img/out/strangedeeptest3.png', eren)
    >>> img_save('img/out/strangedeeptest4.png', erwin)
    
    #TODO: Test other color channels?
    
    >>> erwin.get_pixel('-1', -1)
    Traceback (most recent call last):
    ...
    TypeError
    >>> erwin.get_pixel(-1, 0)
    Traceback (most recent call last):
    ...
    ValueError
    >>> erwin.get_pixel(2000, 0)
    Traceback (most recent call last):
    ...
    ValueError
    >>> erwin.get_pixel(0, 0)
    (91, 146, 166)
    >>> erwin.get_pixel(140, 147)
    (24, 34, 25)
    >>> erwin.set_pixel(0, -1, (3, 3, 3))
    Traceback (most recent call last):
    ...
    ValueError
    >>> print(['E', [eren.set_pixel(340,i,(255,0,0)) for i in range(719)]][0])
    E
    >>> img_save('img/out/strangecolortest1.png', eren)
    >>> print(['E', [eren.set_pixel(340,i,(-1,255,0)) for i in range(719)]][0])
    E
    >>> img_save('img/out/strangecolortest2.png', eren)
    >>> print(['E', [eren.set_pixel(340,i,(-1,-1,-1)) for i in range(719)]][0])
    E
    >>> img_save('img/out/strangecolortest3.png', eren)
    
    """
    
def part_2_tests():
    """
    Tests for ImageProcessing.
    
    >>> erwin = img_read("img/strangethingtemplate.png")
    >>> img_save('img/out/inverwin1.png', IP.negate(erwin))
    >>> img_save('img/out/inverwin2.png', erwin)
    >>> img_save('img/out/tinterwin1.png', IP.tint(erwin, (0,0,0)))
    >>> img_save('img/out/tinterwin2.png', erwin)
    >>> img_save('img/out/tinterwin3.png', IP.tint(erwin, (255,0,0)))
    >>> img_save('img/out/ccerwin1.png', IP.clear_channel(erwin, 0))
    >>> img_save('img/out/ccerwin2.png', erwin)
    >>> img_save('img/out/ccerwin3.png', IP.clear_channel(erwin, 1))
    >>> img_save('img/out/ccerwin4.png', IP.clear_channel(erwin, 2))
    >>> updated_erwin = IP.clear_channel(IP.clear_channel(
    ... IP.clear_channel(erwin, 0), 1), 2)
    >>> img_save('img/out/ccerwin5.png', updated_erwin)
    >>> img_save('img/out/croperwin1.png', IP.crop(erwin, 0, 0, (50, 50)))
    >>> img_save('img/out/croperwin2.png', IP.crop(erwin, 300, 400, (50, 50)))
    >>> img_save('img/out/croperwin3.png', IP.crop(erwin, 0, 0, (5000, 5000)))
    >>> img_save('img/out/croperwin4.png', IP.crop(erwin, 400, 240, (400, 240)))
    >>> img_save('img/out/croperwin5.png', IP.crop(erwin, 1214, 719, (1, 1)))
    >>> dsc = img_read('img/dsc20.png')
    >>> grad = img_read('img/blue_gradient.png')
    >>> img_save('img/out/dschroma1.png', IP.chroma_key(dsc, grad, (255, 205, 210)))
    >>> img_save('img/out/dschroma2.png', IP.chroma_key(dsc, grad, (255, 255, 255)))
    >>> img_save('img/out/dschroma3.png', IP.chroma_key(dsc, grad, (0, 255, 0)))
    >>> img_save('img/out/dschroma4.png', dsc)
    >>> IP.chroma_key('test1', 'test2', (255, 255, 255))
    Traceback (most recent call last):
    ...
    TypeError
    >>> IP.chroma_key(erwin, 'test2', (255, 255, 255))
    Traceback (most recent call last):
    ...
    TypeError
    >>> IP.chroma_key('test1', erwin, (255, 255, 255))
    Traceback (most recent call last):
    ...
    TypeError
    >>> IP.chroma_key(erwin, dsc, (255, 255, 255))
    Traceback (most recent call last):
    ...
    ValueError
    >>> img_save('img/out/fliperwin1.png', IP.rotate_180(erwin))
    >>> img_save('img/out/fliperwin2.png', erwin)
    """

def part_3_tests():
    """
    
    """
