# transfer-learning-with-tensorflow
[retrain_v5.py](https://github.com/pk00095/transfer-learning-with-tensorflow/blob/master/retrain_v5.py) is a customised version of [retrain.py](https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py) for training on image data

# Customizations Done
* Added 2 __dense(Fully connected)__ layers followed by a SoftMax layer.
* Added __dropout__ to both the dense layers.
* Added *batch_normalization* for dense layers
* Replaced __Gradient-descent__ with __Gradient-descent with Momentum__
* Added live streamed confusion Matrix

