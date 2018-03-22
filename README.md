## Environment configuration

First run commands below and this will install CUDA 9.0, cudnn 7.0  and tensorflow-gpu 1.5. These are required for TensorFlow Object Detection API of the version included in this repository. The details about TensorFlow Object Detection API is [here](https://github.com/tensorflow/models/tree/master/research/object_detection)
```
cd envsetup
chmod +x ./*.sh
./prepare.sh
```
After running the setup script above, virtual enviroment, *tensorflow-py3ve* is created. From next time, you need to run below to activate the virtual environment first.
```
source ~/tensorflow-py3ve/bin/activate
```

## Run training
```
python -m object_detection.train --logtostderr --pipeline_config_path=YOUR_CONFIG_FILE --train_dir=TRAIN_RESULT_OUTPUT_DIRECTORY
```

You need to configure your pipeline file for training.

[Configuring an object detection pipeline](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md)

You can download pre-trained object detection models from here.

[Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

## Run evaluation
```
python -m object_detection.eval  --logtostderr  --pipeline_config_path=YOUR_CONFIG_FILE --checkpoint_dir=TRAIN_RESULT_OUTPUT_DIRECTORY --eval_dir=EVAL_RESULT_OUTPUT_DIRECTORY
```
You have to run this evaluation concurrently with training script. This script periodically checks the update in checkpoint_dir and runs evaluation on them.


## TensorBoard setting
```
tensorboard --logdir YOUR_RESULT_OUTPUT_DIRECTORY
ssh -L 16006:127.0.0.1:6006 YOUR_DSVM_IP_ADDRESS
```
Browse to 127.0.0.1:16006 and you can see the training/eval results