#!/usr/bin/python3

import os
import pandas as pd
import numpy as np
import random
from sklearn import svm
import pickle as pk
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from functions.utilities import read_openpose_from_json, compute_head_face_features
from functions.utilities import NUM_JOINTS, CLASS_DICT


def read_annotations_file(dataset_folder, source_data):
    if source_data == 'no_eyecontact':
        annotations_path = dataset_folder + "/leftright_annotations.txt"
    elif source_data == 'board':
        annotations_path = dataset_folder + "/leftrightboard_annotations.txt"
    else:
        print("Error with the source data!!!")
        return None

    dataframe = pd.DataFrame(columns=['participant', 'face_points', 'class', 'notes'])

    annotations_file = open(annotations_path, "r")
    annotations_contents = annotations_file.readlines()
    annotations_contents = [x.strip() for x in annotations_contents]
    annotations = [x.split(" ") for x in annotations_contents]

    # for each line in the annotation file
    for it in range(0, len(annotations)):
        annotation = annotations[it]
        annotation_split = annotation[0].split("/")
        participant = annotation_split[1]
        sample = (annotation_split[-1]).replace('.jpg', '')

        if source_data == 'no_eyecontact':
            openpose_data_dir = os.path.join(dataset_folder, participant, 'leftright_data_openpose')
        elif source_data == 'board':
            openpose_data_dir = os.path.join(dataset_folder, participant, 'leftrightboard_data_openpose')
        openpose_file = openpose_data_dir + '/' + sample + '_keypoints.json'

        if os.path.exists(openpose_file):
            poses, conf_poses, faces, conf_faces = read_openpose_from_json(openpose_file)
            if poses is not None and poses != []:
                features, _ = compute_head_face_features(poses[0], conf_poses[0], faces[0], conf_faces[0])
                if features is not None:
                    label = annotation[1]
                    dataframe = dataframe.append({
                        'participant': participant,
                        'face_points': features,
                        'class': label,
                        'notes': source_data
                    }, ignore_index=True)
                else:
                    print("None features: " + openpose_file)
            else:
                print("None poses: " + openpose_file)
        else:
            print("Error: No JSON: " + openpose_file)
            exit()

    return dataframe


def create_dataset():
    dataset_folder = os.path.join(current_path, "../../Datasets/leftRightGaze_dataset", camera)
    df = pd.DataFrame(columns=['participant', 'face_points', 'class', 'notes'])

    df_noeyecontact = read_annotations_file(dataset_folder, 'no_eyecontact')
    df_board = read_annotations_file(dataset_folder, 'board')

    if df_noeyecontact is not None:
        df = df.append(df_noeyecontact)
    if board_data:
        if df_board is not None:
            df = df.append(df_board)

    return df


def split_train_and_test(df, participants_train, participants_test):
    # build the training set
    df_train = df.loc[df['participant'].isin(participants_train)]
    y_train = df_train.loc[:, 'class'].values
    y_train = np.array([CLASS_DICT[y_train[i]] for i in range(len(y_train))])

    data_train = np.array([np.array(row) for row in df_train.loc[:, 'face_points'].values])
    x_train = data_train[:, 0:NUM_JOINTS * 2]
    c_train = data_train[:, NUM_JOINTS * 2:data_train.shape[1]]

    # weight the coordinates for its condifence value
    wx_train = np.concatenate((np.multiply(x_train[:, ::2], c_train), np.multiply(x_train[:, 1::2], c_train)), axis=1)

    # build the test set
    df_test = df.loc[df['participant'].isin(participants_test)]
    y_test = df_test.loc[:, 'class'].values
    y_test = np.array([CLASS_DICT[y_test[i]] for i in range(len(y_test))])

    data_test = np.array([np.array(row) for row in df_test.loc[:, 'face_points'].values])
    if data_test.size != 0:
        x_test = data_test[:, 0:NUM_JOINTS * 2]
        c_test = data_test[:, NUM_JOINTS * 2:data_test.shape[1]]

        # weight the coordinates for its condifence value
        wx_test = np.concatenate((np.multiply(x_test[:, ::2], c_test), np.multiply(x_test[:, 1::2], c_test)), axis=1)
    else:
        wx_test = np.array([])

    return wx_train, y_train, wx_test, y_test


# --------------------------------------
camera = 'realsense'
algorithm = 'calibrated_svm'
board_data = True
final_train = True
k = 5
# --------------------------------------

# create the dataset or load if exists
current_path = os.getcwd()
dataset_file = os.path.join(current_path, "../dataset", camera, "feats_dataset_leftright.pkl")

if not os.path.isfile(dataset_file):
    dataset = create_dataset()
    dataset.to_pickle(dataset_file)
else:
    dataset = pd.read_pickle(dataset_file)

if final_train:
    y = dataset.loc[:, 'class'].values
    y = np.array([CLASS_DICT[y[i]] for i in range(len(y))])

    data = np.array([np.array(row) for row in dataset.loc[:, 'face_points'].values])
    x = data[:, 0:NUM_JOINTS * 2]
    c = data[:, NUM_JOINTS * 2:data.shape[1]]
    wx = np.concatenate((np.multiply(x[:, ::2], c), np.multiply(x[:, 1::2], c)), axis=1)

    if algorithm == 'svm':
        print("Training svm model...")
        params = {'C': np.linspace(0.01, 10, 30), 'gamma': np.linspace(0.0001, 1, 30)}
        model = svm.SVC(decision_function_shape="ovr", kernel='rbf', class_weight='balanced')
        clf = GridSearchCV(model, param_grid=params)
        clf.fit(wx, y)
        print("The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))
        print("SVM model built.")
    elif algorithm == 'calibrated_svm':
        print("Training calibrated svm model...")
        params = {'C': np.linspace(0.01, 10, 30), 'gamma': np.linspace(0.0001, 1, 30)}
        model = svm.SVC(decision_function_shape="ovr", kernel='rbf', class_weight='balanced')
        base_clf = GridSearchCV(model, param_grid=params)
        base_clf.fit(wx, y)
        print("The best parameters are %s with a score of %0.2f" % (base_clf.best_params_, base_clf.best_score_))
        print("SVM model built.")

        clf = CalibratedClassifierCV(base_estimator=base_clf, cv="prefit")
        clf.fit(wx, y)
        print("SVM classifier has been calibrated.")
    else:
        print('No algorithm selected. Choose: svm or calibrated_svm.')
        exit

    out_model_file = current_path + '/functions/leftright_model_%s.pkl' % algorithm
    pk.dump(clf, open(out_model_file, 'wb'))
    print('Model saved.')
else:
    # train model
    metrics_accuracy = []
    metrics_precision = []
    metrics_recall = []
    metrics_f1score = []

    setsFile_participants = current_path + "/../dataset/" + camera + "/setsFile_participants.npz"
    if not os.path.isfile(setsFile_participants):
        pxx_train = []
        pxx_test = []
        print('Creating sets participant file...')
        for i in range(0, k):
            print('\nk: ', i)
            participants = set(dataset.loc[:, 'participant'].values)
            participants_test = random.sample(participants, 5)
            participants_train = list(filter(lambda p: p not in participants_test, participants))
            print('Participants for the train: ', participants_train)
            print('Participants for the test: ', participants_test)
            pxx_train.append(participants_train)
            pxx_test.append(participants_test)
        np.savez(setsFile_participants, pxx_train=pxx_train, pxx_test=pxx_test)
    else:
        participants = np.load(setsFile_participants)
        pxx_train = participants['pxx_test']
        pxx_test = participants['pxx_train']

    for i in range(0, k):
        print('\nk: ', i)
        setsFile = current_path + "/../dataset/" + camera + "/setsFile_split" + str(i) + ".npz"
        if not os.path.isfile(setsFile):
            print('Creating dataset file split number %d' % i)
            wx_train, y_train, wx_test, y_test = split_train_and_test(dataset, pxx_train[i], pxx_test[i])
            datafile = current_path + "/../dataset/" + camera + '/setsFile_split' + str(i) + '.npz'
            np.savez(datafile, wx_train=wx_train, wx_test=wx_test, y_train=y_train, y_test=y_test)

        data_arr = np.load(setsFile)
        wx_train = data_arr['wx_train']
        wx_test = data_arr['wx_test']
        y_train = data_arr['y_train']
        y_test = data_arr['y_test']

        if algorithm == 'svm':
            params = {'C': np.linspace(0.01, 10, 30), 'gamma': np.linspace(0.0001, 1, 30)}
            clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid=params)
            clf.fit(wx_train, y_train)
            # clf = CalibratedClassifierCV(base_estimator=base_clf, cv="prefit")
            # clf.fit(wx_train, y_train)
            print("The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))
        elif algorithm == 'calibrated_svm':
            params = {'C': np.linspace(0.01, 10, 30), 'gamma': np.linspace(0.0001, 1, 30)}
            model = svm.SVC(decision_function_shape="ovr", kernel='rbf', class_weight='balanced')
            base_clf = GridSearchCV(model, param_grid=params)
            base_clf.fit(wx_train, y_train)
            print("The best parameters are %s with a score of %0.2f" % (base_clf.best_params_, base_clf.best_score_))
            print("SVM model built.")

            clf = CalibratedClassifierCV(base_estimator=base_clf, cv="prefit")
            clf.fit(wx_train, y_train)
            print("SVM classifier has been calibrated.")

        y_pred = clf.predict(wx_test)
        # Model Accuracy: how often is the classifier correct?
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        metrics_accuracy.append(accuracy)
        # Model Precision: what percentage of positive tuples are labeled as such in the total predictive positive?
        precision = precision_score(y_test, y_pred, labels=list(CLASS_DICT.values()))
        print("Precision:", precision)
        metrics_precision.append(precision)
        # Model Recall: what percentage of positive tuples are labelled as such in the total actual positive?
        recall = recall_score(y_test, y_pred, labels=list(CLASS_DICT.values()))
        print("Recall:", recall)
        metrics_recall.append(recall)
        # Weighted average of the precision and recall
        f1score = f1_score(y_test, y_pred, labels=list(CLASS_DICT.values()))
        print("F1 score:", f1score)
        metrics_f1score.append(f1score)

    print('\n')
    print("%0.2f accuracy with a standard deviation of %0.2f" % (np.array(metrics_accuracy).mean(), np.array(metrics_accuracy).std()))
    print("%0.2f precision with a standard deviation of %0.2f" % (
    np.array(metrics_precision).mean(), np.array(metrics_precision).std()))
    print("%0.2f recall with a standard deviation of %0.2f" % (np.array(metrics_recall).mean(), np.array(metrics_recall).std()))
    print("%0.2f f1 score with a standard deviation of %0.2f" % (np.array(metrics_f1score).mean(), np.array(metrics_f1score).std()))