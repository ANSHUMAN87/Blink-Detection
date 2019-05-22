import numpy as np
import os
import cv2


class DataLoader(object):
    """Class for loading CEW dataset."""
    def __init__(self, dataset_name='cew',
                 dataset_path=None, image_size=(64, 64)):

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size
        if self.dataset_path is not None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'cew':
            self.dataset_path = '../dataset/'
        else:
            raise Exception(
                    'Incorrect dataset name, please input cew')

    def get_data(self):
        if self.dataset_name == 'cew':
            ground_truth_data = self._load_cew()
        return ground_truth_data

    def _load_cew(self):
        openfaces, openeyestate = self._load_subdata("OpenFace", 0)
        openfaceCount = openfaces.shape[0]
        closedfaces, closedeyestate = self._load_subdata("ClosedFace", 1)
        closedfaceCount = closedfaces.shape[0]
        count_0 = 0
        count_1 = 0
        step_0 = 8
        step_1 = 8
        faces = []
        eyestates = []
        for i in range(0, 200):
            face_tmp = openfaces[count_0:count_0 + step_0, ...]
            for j in range(0, face_tmp.shape[0]):
                faces.append(face_tmp[j])
            for j in range(0, face_tmp.shape[0]):
                eyestates.append(openeyestate[count_0:count_0 + step_0, ...][j])

            count_0 = (count_0 + step_0) if openfaceCount > count_0 + step_0 else 0

            face_tmp = closedfaces[count_1:count_1 + step_1, ...]
            for j in range(0, face_tmp.shape[0]):
                faces.append(face_tmp[j])
            for j in range(0, face_tmp.shape[0]):
                eyestates.append(closedeyestate[count_1:count_1 + step_1, ...][j])

            count_1 = (count_1 + step_1) if closedfaceCount > count_1 + step_1 else 0

        faces = np.asarray(faces)
        eyestates = np.asarray(eyestates)
        return faces, eyestates

    def _load_subdata(self, name, eyestate):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        file_paths = []
        for folder, subfolders, filenames in os.walk(os.path.join(self.dataset_path, name)):
            for filename in filenames:
                if filename.lower().endswith(('.jpg')):
                    file_paths.append(os.path.join(folder, filename))

        num_faces = len(file_paths)
        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size, 3))
        eyestates = np.zeros(shape=(num_faces, num_classes))
        for file_arg, file_path in enumerate(file_paths):
            image_array = cv2.imread(file_path)
            image_array = cv2.resize(image_array, (y_size, x_size))
            faces[file_arg] = image_array
            eyestates[file_arg, eyestate] = 1
        #faces = np.expand_dims(faces, -1)
        return faces, eyestates


def get_labels(dataset_name):
    if dataset_name == 'cew':
        return {0: 'open', 1: 'close'}
    else:
        raise Exception('Invalid dataset name')


def get_class_to_arg(dataset_name='cew'):
    if dataset_name == 'cew':
        return {'open': 0, 'close': 1}
    else:
        raise Exception('Invalid dataset name')


def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data
