import pandas as pd
import numpy as np
import os
import shutil

import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('label_path',
                    '/home/ace19/dl_data/histopathologic_cancer_detection_ori/train_labels.csv',
                    'Path to label')

flags.DEFINE_string('whole_slide_image_path',
                    '/home/ace19/dl_data/histopathologic_cancer_detection_ori/patch_id_wsi.csv',
                    'Path to label')

flags.DEFINE_string('target_path',
                    '/home/ace19/dl_data/histopathologic_cancer_detection',
                    'Path to label')

flags.DEFINE_string('original_dataset',
                    '/home/ace19/dl_data/histopathologic_cancer_detection_ori/original_train',
                    'Path to original dataset')

DATASET_TRAIN = 'train'
DATASET_VALIDATE = 'validate'


def transfer(files, dataset_type):
    total = len(files)

    for idx, images in enumerate(files):
        if idx % 100 == 0:
            tf.logging.info('On image %d of %d', idx, total)
        src = os.path.join(FLAGS.original_dataset, images)
        dst = os.path.join(FLAGS.target_path, dataset_type, images)
        shutil.copyfile(src, dst)


def return_tumor_or_not(dic, one_id):
    return dic[one_id]


def create_dict():
    df = pd.read_csv(FLAGS.label_path)
    result_dict = {}
    for index in range(df.shape[0]):
        one_id = df.iloc[index,0]
        tumor_or_not = df.iloc[index,1]
        result_dict[one_id] = int(tumor_or_not)
    return result_dict


def find_missing(train_ids, cv_ids):
    all_ids = set(pd.read_csv(FLAGS.label_path)['id'].values)
    wsi_ids = set(train_ids + cv_ids)

    missing_ids = list(all_ids-wsi_ids)
    return missing_ids


def generate_split():
    ids = pd.read_csv(FLAGS.whole_slide_image_path)
    wsi_dict = {}
    for i in range(ids.shape[0]):
        wsi = ids.iloc[i,1]
        train_id = ids.iloc[i,0]
        wsi_array = wsi.split('_')
        number = int(wsi_array[3])
        if wsi_dict.get(number) is None:
            wsi_dict[number] = [train_id]
        else:
            wsi_dict[number].append(train_id)

    wsi_keys = list(wsi_dict.keys())
    np.random.seed()
    np.random.shuffle(wsi_keys)
    amount_of_keys = len(wsi_keys)

    keys_for_train = wsi_keys[0:int(amount_of_keys*0.8)]
    keys_for_cv = wsi_keys[int(amount_of_keys*0.8):]
    train_ids = []
    cv_ids = []

    for key in keys_for_train:
        train_ids += wsi_dict[key]

    for key in keys_for_cv:
        cv_ids += wsi_dict[key]

    dic = create_dict()

    missing_ids = find_missing(train_ids, cv_ids)
    missing_ids_total = len(missing_ids)
    np.random.seed()
    np.random.shuffle(missing_ids)

    train_missing_ids = missing_ids[0:int(missing_ids_total*0.8)]
    cv_missing_ids = missing_ids[int(missing_ids_total*0.8):]

    train_ids += train_missing_ids
    cv_ids += cv_missing_ids

    train_labels = []
    cv_labels = []

    train_tumor = 0
    for one_id in train_ids:
        temp = return_tumor_or_not(dic, one_id)
        train_tumor += temp
        train_labels.append(temp)

    cv_tumor = 0
    for one_id in cv_ids:
        temp = return_tumor_or_not(dic, one_id)
        cv_tumor += temp
        cv_labels.append(temp)
    total = len(train_ids) + len(cv_ids)

    print("Amount of train labels: {}, {}/{}".format(len(train_ids), train_tumor, len(train_ids)-train_tumor))
    print("Amount of cv labels: {}, {}/{}".format(len(cv_ids), cv_tumor, len(cv_ids) - cv_tumor))
    print("Percentage of cv labels: {}".format(len(cv_ids)/total))

    return train_ids, cv_ids, train_labels, cv_labels


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    train_ids, cv_ids, train_labels, cv_labels = generate_split()

    train_path = os.path.join(FLAGS.target_path, 'train')
    validate_path = os.path.join(FLAGS.target_path, 'validate')

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(validate_path):
        os.makedirs(validate_path)

    train = [name + '.png' for name in train_ids]
    validate = [name + '.png' for name in cv_ids]

    # Move images to directory structures
    transfer(train, DATASET_TRAIN)
    transfer(validate, DATASET_VALIDATE)


if __name__ == '__main__':
    tf.app.run()