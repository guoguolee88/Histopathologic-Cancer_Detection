'''
히스토그램 균일화의 한계를 극복하는 Adaptive Histogram Equalization

CLAHE 는 이미지를 일정한 크기를 가진 작은 블록으로 구분하고, 블록별로 히스토그램 균일화를 시행하여
이미지 전체에 대해 균일화를 달성하는 메커니즘을 가지고 있다.

이미지에 노이즈가 있는 경우, 타일 단위의 히스토그램 균일화를 적용하면 노이즈가 커질 수 있는데,
CLAHE 알고리즘은 이러한 노이즈를 감쇠시키는 Contrast Limiting 이라는 기법을 사용한다.

각 타일별로 히스토그램 균일화가 모두 마무리 되면, 타일간 경계 부분은 bilinear interpolation 을 적용해
매끈하게 만들어 준다.

OpenCV 버전에 따라 cv2.createCLAHE() 함수가 지원되지 않는 경우가 있다.
'''

import os
import cv2
import numpy as np

from matplotlib import pyplot as plt

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('dataset_dir',
                    '/home/ace19/dl_data/KGC_roi',
                    'Where the dataset reside.')
flags.DEFINE_string('target_dir',
                    '/home/ace19/dl_data/KGC',
                    'Where the target is.')


def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    data = os.listdir(FLAGS.dataset_dir)
    for label in data:
        label_path = os.path.join(FLAGS.dataset_dir, label)
        imgs = os.listdir(label_path)
        # 정렬된 결과 리턴, 리스트 원본을 정렬하기 위해서는 list.sort(reverse=True)
        # 참고로 list.reverse() 는 정렬이 아니고 리스트를 거꾸로 뒤짚는 것
        imgs = sorted(imgs)
        total = len(imgs)
        for idx, img_name in enumerate(imgs):
            if idx % 100 == 0:
                tf.logging.info('%d/%d completed' % (idx, total))

            filename = os.path.join(label_path, img_name)
            img_grayscale = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            # _hist_with_opencv(img_grayscale, img_name)
            # histogram_equalization_with_opencv(img_grayscale, img_name)
            contrast_limited_adaptive_histogram_equalization(img_grayscale, label, img_name)


# grayscale 이미지만 적용 가능
def histogram_equalization_with_opencv(image, name):
    equ = cv2.equalizeHist(image)
    # image 와 equ 를 수평으로 붙인다.
    res = np.hstack((image, equ))
    cv2.imshow(name, res)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

    _hist_with_opencv(res, name)


# grayscale 이미지만 적용 가능
def contrast_limited_adaptive_histogram_equalization(image, label, name):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(image)
    # image 와 equ 를 수평으로 붙인다.
    # res = np.hstack((image, img))
    # cv2.imshow(name, res)
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()
    target = os.path.join(FLAGS.target_dir, label, name)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    cv2.imwrite(target, img)

    # _hist_with_opencv(res, name)


def _hist_with_opencv(image, name):
    # 히스토그램을 구하기 위한 가장 성능이 좋은 함수는 cv2.calcHist()
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    plt.plot(hist)
    plt.title(name)
    plt.show()



if __name__ == '__main__':
    tf.app.run()