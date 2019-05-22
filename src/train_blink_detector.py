#######################################################################
# Author: Anshuman Tripathy                                           #
# Email: a.tripathy87@gmail.com                                       #
# Github: https://github.com/ANSHUMAN87/Blink-Detection               #
# Description: Train blink detection model based on CEW dataset       #
#######################################################################

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from cnn import blink_detector
from datasets import DataLoader
from datasets import split_data
from utils import preprocess_input

# parameters
batch_size = 32
num_epochs = 10000
input_shape = (64, 64, 3)
validation_split = .2
verbose = 1
num_classes = 2
patience = 50
base_path = '../trained_models/'

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# model parameters/compilation
model = blink_detector(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# callbacks
log_file_path = base_path + 'cew_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
#early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                          patience=int(patience/4), verbose=1)
trained_models_path = base_path + 'cew_blink_detect'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                            save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# loading dataset
data_loader = DataLoader('cew')
faces, eyestates = data_loader.get_data()
print("out shapes: ", faces.shape, eyestates.shape)
faces = preprocess_input(faces)
num_samples, num_classes = eyestates.shape
train_data, val_data = split_data(faces, eyestates, validation_split)
train_faces, train_eyestates = train_data
model.fit_generator(data_generator.flow(train_faces, train_eyestates,
                                    batch_size),
                steps_per_epoch=len(train_faces) / batch_size,
                epochs=num_epochs, verbose=1, callbacks=callbacks,
                validation_data=val_data)
