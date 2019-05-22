# Blink-Detection
Train a model to detect if someone has blinked in an image.

# Dataset
The model is trained with [CEW dataset](http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html)
As i have trained with Facial images in size of 64x64, this model needs a face detection model.
So i have also enclosed opencv face detection model for ease of use.
If anyone wants to use another face detection model, then may be code has to change little bit accordingly.

# Usage
python blink_detector_demo.py ../test-image/IMG_20190427_223302_3.jpg ../haarcascade_frontalface_default.xml ../trained_models/cew_blink_detect.hdf5

# NOTE
I have created this project just for fun. In case any issue, feel free to post.

