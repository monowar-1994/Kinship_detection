import pandas as pd
import os
import numpy as np
import random
import itertools as itools
import cv2 as cv
import traceback


IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
BATCH_SIZE = 32


class DataDefinition:
    def __init__(self, _image1, _image2, _label):
        self.image1 = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        self.image1 = _image1
        self.image2 = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        self.image2 = _image2
        self.label = _label


def create_positive_data_file(training_file_name, picture_directory):
    """
    This function curates the data and stores the filepath and labels in a csv file.
    :param training_file_name: contains the family picture mapping but folder wise
    :param picture_directory: Directory where the folderwise family structure is shown
    :return: Returns number of entries it created after creating a csv file which contains all the necessary information
    """

    train_dataframe = pd.DataFrame(pd.read_csv(training_file_name))
    person1_list = []
    person2_list = []
    file_not_found_case = 0
    for index, row in train_dataframe.iterrows():
        first_person = row['p1']
        second_person = row['p2']

        try:
            list_of_pictures_person1 = os.listdir(os.path.join(picture_directory, first_person))
            list_of_pictures_person2 = os.listdir(os.path.join(picture_directory, second_person))

            for p1 in list_of_pictures_person1:
                for p2 in list_of_pictures_person2:
                    person1_list.append(os.path.join(picture_directory, first_person, p1))
                    person2_list.append(os.path.join(picture_directory, second_person, p2))
        except Exception:
            file_not_found_case += 1
            continue

        assert len(person1_list) == len(person2_list)
    labels = np.ones((len(person1_list)))
    mapping_dictionary = {'p1': person1_list, 'p2': person2_list, 'labels': labels}
    mapping_dataframe = pd.DataFrame(data=mapping_dictionary)
    # print(file_not_found_case)
    return mapping_dataframe


def create_negative_data_file(train_data_directory, negative_sample_number):
    list_of_family_folder = os.listdir(train_data_directory)
    total_sample_number = 0
    filename_list1 = []
    filename_list2 = []

    while total_sample_number < negative_sample_number:
        try:
            # Choose two random families
            families = random.sample(list_of_family_folder, 2)
            fam_1 = families[0]
            fam_2 = families[1]

            # Choose random person in family 1 and 2

            rand_fam1_person = random.sample(os.listdir(os.path.join(train_data_directory, fam_1)), 1)
            rand_fam2_person = random.sample(os.listdir(os.path.join(train_data_directory, fam_2)), 1)

            p1 = rand_fam1_person[0]
            p2 = rand_fam2_person[0]

            # Selecting all the pictures of those two specific person
            p1_pictures = os.listdir(os.path.join(train_data_directory, fam_1, p1))
            p2_pictures = os.listdir(os.path.join(train_data_directory, fam_2, p2))

            combination = list(itools.product(p1_pictures, p2_pictures))
            total_sample_number += len(combination)

            for element in combination:
                x, y = element
                filename_list1.append(os.path.join(train_data_directory, fam_1, p1, x))
                filename_list2.append(os.path.join(train_data_directory, fam_2, p2, y))
        except Exception:
            continue

    assert len(filename_list1) == len(filename_list2)
    labels = np.zeros(len(filename_list1))
    mapping_dictionary = {'p1':filename_list1, 'p2':filename_list2, 'labels':labels}
    mapping_dataframe = pd.DataFrame(data=mapping_dictionary)
    return mapping_dataframe


def create_list(pos_df, neg_df, pos_sample_num= -1, neg_sample_num= -1):
    print("Creating Data definition list")
    data_def_list = []
    failure_to_add = 0

    # get the positive data
    count = 0
    for index, row in pos_df.iterrows():
        filename1 = row['p1']
        filename2 = row['p2']

        try:
            data_def_list.append(DataDefinition(cv.imread(filename1), cv.imread(filename2), 1))
            count += 1
            if pos_sample_num != -1 and count == pos_sample_num:
                break
        except Exception:
            failure_to_add += 1
            continue

    # get the negative data
    count = 0
    for index, row in neg_df.iterrows():
        filename1 = row['p1']
        filename2 = row['p2']
        try:
            data_def_list.append(DataDefinition(cv.imread(filename1), cv.imread(filename2), 0))
            count += 1
            if neg_sample_num != -1 and count == neg_sample_num:
                break
        except Exception:
            failure_to_add += 1
            continue
    print("Data Definition list create ends")
    return data_def_list, failure_to_add


def join_and_shuffle_data(pos_df, neg_df):
    merged_data_frame = pd.concat([pos_df,neg_df],ignore_index=True)
    merged_data_frame = merged_data_frame.sample(frac=1).reset_index(drop=True)
    return merged_data_frame


def get_data(merged_data_frame, training_data_size, validation_data_size):
    print("Generating Data Array")
    x_train1 = np.zeros((training_data_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype='float32')
    x_train2 = np.zeros((training_data_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype='float32')

    x_valid1 = np.zeros((validation_data_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype='float32')
    x_valid2 = np.zeros((validation_data_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype='float32')

    y_train = np.zeros(training_data_size)
    y_valid = np.zeros(validation_data_size)

    train_index = 0
    while train_index < training_data_size:
        try:
            x_train1[train_index] = cv.imread(merged_data_frame['p1'][train_index])/255.0
            x_train2[train_index] = cv.imread(merged_data_frame['p2'][train_index])/255.0

            y_train[train_index] = merged_data_frame['labels'][train_index]
            train_index += 1
        except IOError:
            continue
        except Exception:
            traceback.print_exc()
            continue
        if train_index%1000 == 0:
            print(train_index)

    valid_index = 0

    while valid_index < validation_data_size:
        try:
            x_valid1[valid_index] = cv.imread(merged_data_frame['p1'][valid_index + training_data_size])/255.0
            x_valid2[valid_index] = cv.imread(merged_data_frame['p2'][valid_index + training_data_size])/255.0

            y_valid[valid_index] = merged_data_frame['labels'][valid_index + training_data_size]
            valid_index += 1
        except IOError:
            continue
        except Exception:
            traceback.print_exc()
            continue

    print("Data Array Generation ends")
    return x_train1, x_train2, y_train, x_valid1, x_valid2, y_valid


def preprocess(train_file_name, train_file_dir, negative_sample_number=None):
    """
    This function will create the list of pairings of images with labels. Basically two numpy array of dimensions
    (224,224,3) and their corresponding labels are stored as an object and this list contains those objects
    :param train_file_name: The csv file name that contains the positive data mappings
    :param train_file_dir: The directory where the images are stored
    :param negative_sample_number: The number of negative samples to generate . Ideally should be equal to positive sample.
    :param pos_sample: Number of positive_sample to put in the data definition list
    :param neg_sample: Number of negative sample to put in the data definition list
    :return: Retuns the data definition list
    """
    pos_df = create_positive_data_file(train_file_name, train_file_dir)
    if negative_sample_number is None:
        num = len(pos_df)
    else:
        num = negative_sample_number
    neg_df = create_negative_data_file(train_file_dir, num)

    merged_df = join_and_shuffle_data(pos_df,neg_df)
    return merged_df
