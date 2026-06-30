# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='./model/keypoint_classifier/keypoint_classifier.keras',
    ):
        self.model = tf.keras.models.load_model(model_path)

    def __call__(
        self,
        landmark_list,
    ):
        # Input must be 2D: [batch, features]
        input_data = np.array([landmark_list], dtype=np.float32)

        # Predict
        result = self.model.predict(input_data, verbose=0)

        # Get index of the highest probability
        result_index = np.argmax(np.squeeze(result))

        return result_index
