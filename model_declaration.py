from keras import layers
from keras import models
from keras.applications import vgg19
import numpy as np
import keras.backend as K


class SimiliarityLoss(layers.Layer):
    def __init__(self, **kwargs):
        super(SimiliarityLoss, self).__init__(**kwargs)

    def call(self, inputs):
        feature_map1, feature_map2 = inputs
        similarity_loss = K.tf.losses.mean_squared_error(feature_map1, feature_map2)
        self.add_loss(similarity_loss)
        return [feature_map1, feature_map2]


class ContrastiveLoss(layers.Layer):
    def __init__(self, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)

    def call(self, inputs):
        feature_map1, feature_map2, labels, predictions = inputs

        similarity_diff = K.tf.losses.mean_squared_error(feature_map1, feature_map2)

        contrastive_loss = K.sum((labels * similarity_diff) - ((1 - labels) * similarity_diff))
        crossentropy_loss = K.tf.losses.sigmoid_cross_entropy(labels, predictions)

        total_loss = contrastive_loss + crossentropy_loss
        self.add_loss(total_loss)

        return [contrastive_loss, crossentropy_loss, total_loss]


def contrastive_loss(fmap1, fmap2):
    def loss(y_true, y_pred):
        similarity_diff = K.tf.losses.mean_squared_error(fmap1, fmap2)
        contrastive_loss = K.sum((y_true * similarity_diff) - ((1 - y_true) * similarity_diff))
        crossentropy_loss = K.tf.losses.sigmoid_cross_entropy(y_true, y_pred)
        total_loss = contrastive_loss + crossentropy_loss
        return total_loss

    return loss


def ms_ssim_loss(fmap1, fmap2):
    def loss(y_true, y_pred):
        max_tensor = K.max(K.maximum(fmap1, fmap2))
        max_val = np.max(K.eval(max_tensor))
        similarity_diff = K.tf.image.ssim_multiscale(fmap1,fmap2,max_val)
        contrastive_loss = K.sum((y_true * similarity_diff) - ((1 - y_true) * similarity_diff))
        crossentropy_loss = K.tf.losses.sigmoid_cross_entropy(y_true, y_pred)
        total_loss = contrastive_loss + crossentropy_loss
        return total_loss
    return loss


def get_model():
    input_branch_a = layers.Input(shape=(224, 224, 3))
    input_branch_b = layers.Input(shape=(224, 224, 3))

    base_a = vgg19.VGG19(include_top=False, weights='imagenet')
    base_a.name = 'model_a'

    for layer in base_a.layers:
        if layer.name.startswith('block5'):
            layer.trainable = True
        else:
            layer.trainable = False

    base_b = vgg19.VGG19(include_top=False, weights='imagenet')
    base_b.name = 'model_b'

    for layer in base_b.layers:
        if layer.name.startswith('block5'):
            layer.trainable = True
        else:
            layer.trainable = False

    branch_a = base_a(input_branch_a)
    branch_b = base_b(input_branch_b)

    assert branch_a is not None
    assert branch_b is not None

    print('assertion_complete')

    vgg_feature_a = branch_a
    vgg_feature_b = branch_b

    flattened_fmap1 = layers.Flatten()(branch_a)
    flattened_fmap2 = layers.Flatten()(branch_b)

    print('Flattening complete')

    merged_layer = layers.merge.concatenate([flattened_fmap1, flattened_fmap2])
    normalization_layer = layers.BatchNormalization(axis=-1)(merged_layer)

    dense_layer1 = layers.Dense(256, activation='relu')(normalization_layer)
    dropout_layer1 = layers.Dropout(0.5)(dense_layer1)
    dense_layer2 = layers.Dense(128, activation='relu')(dropout_layer1)
    dropout_layer2 = layers.Dropout(0.5)(dense_layer2)

    prediction_layer = layers.Dense(1, activation='tanh')(dropout_layer2)

    print('predictions complete')

    branched_model = models.Model(inputs=[input_branch_a, input_branch_b], outputs=[prediction_layer])

    return branched_model, vgg_feature_a, vgg_feature_b


def get_simple_conv():
    input_branch_a = layers.Input(shape=(224, 224, 3))
    input_branch_b = layers.Input(shape=(224, 224, 3))

    # Branch A
    conv_a1 = layers.Conv2D(32,(5,5),strides=(2,2),activation='relu')(input_branch_a)
    batch_norm_a1 = layers.BatchNormalization(axis = -1)(conv_a1)

    conv_a2 = layers.Conv2D(64, (5, 5), strides=(2, 2), activation='relu')(batch_norm_a1)
    batch_norm_a2 = layers.BatchNormalization(axis=-1)(conv_a2)

    conv_a3 = layers.Conv2D(128, (5, 5), strides=(2, 2), activation='relu')(batch_norm_a2)
    batch_norm_a3 = layers.BatchNormalization(axis=-1)(conv_a3)

    conv_a4 = layers.Conv2D(256, (5, 5),strides= (2, 2), activation='relu')(batch_norm_a3)
    batch_norm_a4 = layers.BatchNormalization(axis=-1)(conv_a4)

    flatten_1 = layers.Flatten()(batch_norm_a4)

    # Branch B

    conv_b1 = layers.Conv2D(32, (5, 5),strides= (2, 2), activation='relu')(input_branch_b)
    batch_norm_b1 = layers.BatchNormalization(axis=-1)(conv_b1)

    conv_b2 = layers.Conv2D(64, (5, 5),strides= (2, 2), activation='relu')(batch_norm_b1)
    batch_norm_b2 = layers.BatchNormalization(axis=-1)(conv_b2)

    conv_b3 = layers.Conv2D(128, (5, 5),strides= (2, 2), activation='relu')(batch_norm_b2)
    batch_norm_b3 = layers.BatchNormalization(axis=-1)(conv_b3)

    conv_b4 = layers.Conv2D(256, (5, 5),strides= (2, 2), activation='relu')(batch_norm_b3)
    batch_norm_b4 = layers.BatchNormalization(axis=-1)(conv_b4)

    flatten_2 = layers.Flatten()(batch_norm_b4)

    # Merge both branches

    merged_layer = layers.merge.concatenate([flatten_1, flatten_2])

    dense_layer1 = layers.Dense(256, activation='relu')(merged_layer)
    dropout_layer1 = layers.Dropout(0.5)(dense_layer1)
    prediction_layer = layers.Dense(1, activation='sigmoid')(dropout_layer1)

    branched_model = models.Model(inputs=[input_branch_a, input_branch_b], outputs=[prediction_layer])

    return branched_model, conv_a4, conv_b4



