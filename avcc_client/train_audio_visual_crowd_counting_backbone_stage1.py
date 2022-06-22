#!/usr/bin/env python
# coding: utf-8

# Download pre-trained VGGish models from [here](https://github.com/DTaoo/VGGish).
#
# Download the DISCO dataset from [here](https://dtaoo.github.io/dataset.html) or [here](https://zenodo.org/record/3828468).
#
# Copy the `vggish_params.py`, `mel_features.py` and `preprocess_sound.py` files from [this repository](https://github.com/DTaoo/VGGish) next to this notebook.
#
# Modify `vggish_params.py` to resolve [this issue](https://github.com/DTaoo/VGGish/issues/11).

# In[ ]:


SELECTED_GPUS = [0, 1]  # which GPUs to use

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    [str(gpu_number) for gpu_number in SELECTED_GPUS]
)

import tensorflow as tf

tf.get_logger().setLevel("INFO")

assert len(tf.config.list_physical_devices("GPU")) > 0

GPUS = tf.config.experimental.list_physical_devices("GPU")
for gpu in GPUS:
    tf.config.experimental.set_memory_growth(gpu, True)

DISTRIBUTED_STRATEGY = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.NcclAllReduce(),
    devices=["/gpu:%d" % index for index in range(len(SELECTED_GPUS))],
)

NUM_GPUS = DISTRIBUTED_STRATEGY.num_replicas_in_sync

print("Number of devices: {}".format(NUM_GPUS))

import librosa
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.io
import scipy.stats as st
import sys
import tensorflow_addons as tfa
from scipy import signal
from skimage.transform import resize
from preprocess_sound import preprocess_sound

DISCO_PATH = "disco"
WAVEFORMS_PATH = os.path.join(DISCO_PATH, "auds")
IMAGES_PATH = os.path.join(DISCO_PATH, "imgs")
TRAIN_DENSITY_MAPS_PATH = os.path.join(DISCO_PATH, "train")
VAL_DENSITY_MAPS_PATH = os.path.join(DISCO_PATH, "val")
TEST_DENSITY_MAPS_PATH = os.path.join(DISCO_PATH, "test")
DISCO_SAMPLING_RATE = 48000
CACHE_DIR = os.path.join(DISCO_PATH, "cache")
VIDEO_SIZE = (576, 1024)
AUDIO_SIZE = (96, 64, 1)


# In[2]:


def VGGish(
    load_weights=True,
    weights="audioset",
    input_tensor=None,
    input_shape=AUDIO_SIZE,
    out_dim=128,
    include_top=True,
    pooling="avg",
):
    if weights not in {"audioset", None}:
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization) or `audioset` "
            "(pre-training on audioset)."
        )
    if input_tensor is None:
        aud_input = tf.keras.layers.Input(shape=input_shape, name="input_1")
    else:
        aud_input = input_tensor

    x = tf.keras.layers.Conv2D(
        64, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv1"
    )(aud_input)
    x = tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding="same", name="pool1"
    )(x)

    x = tf.keras.layers.Conv2D(
        128, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv2"
    )(x)
    x = tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding="same", name="pool2"
    )(x)

    x = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        activation="relu",
        padding="same",
        name="conv3/conv3_1",
    )(x)
    x = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        activation="relu",
        padding="same",
        name="conv3/conv3_2",
    )(x)
    x = tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding="same", name="pool3"
    )(x)

    x = tf.keras.layers.Conv2D(
        512,
        (3, 3),
        strides=(1, 1),
        activation="relu",
        padding="same",
        name="conv4/conv4_1",
    )(x)
    x = tf.keras.layers.Conv2D(
        512,
        (3, 3),
        strides=(1, 1),
        activation="relu",
        padding="same",
        name="conv4/conv4_2",
    )(x)
    x = tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding="same", name="pool4"
    )(x)

    if include_top:
        x = tf.keras.layers.Flatten(name="flatten_")(x)
        x = tf.keras.layers.Dense(4096, activation="relu", name="vggish_fc1/fc1_1")(x)
        x = tf.keras.layers.Dense(4096, activation="relu", name="vggish_fc1/fc1_2")(x)
        x = tf.keras.layers.Dense(out_dim, activation="relu", name="vggish_fc2")(x)
    else:
        if pooling == "avg":
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = tf.keras.layers.GlobalMaxPooling2D()(x)

    model = tf.keras.models.Model(aud_input, x, name="VGGish")

    if load_weights:
        if weights == "audioset":
            if include_top:
                model.load_weights("vggish_audioset_weights.h5")
            else:
                model.load_weights("vggish_audioset_weights_without_fc2.h5")
        else:
            print("failed to load weights")

    return model


# In[3]:


def get_model():
    audio_input_tensor = tf.keras.layers.Input(shape=AUDIO_SIZE)
    audio_backbone_model = VGGish(
        include_top=False, input_shape=AUDIO_SIZE, input_tensor=audio_input_tensor
    )
    audio_output = audio_backbone_model.get_layer(index=-1).output

    video_backbone_model = tf.keras.applications.VGG16(
        include_top=False, input_shape=(VIDEO_SIZE[0], VIDEO_SIZE[1], 3)
    )
    video_input = video_backbone_model.get_layer(index=0).input
    video_output = video_backbone_model.get_layer("block4_conv3").output

    feature_fusion_block_filters = [512, 512, 512, 256, 128, 64]
    for filter_size in feature_fusion_block_filters:
        video_output = tf.keras.layers.Conv2D(
            filters=filter_size,
            kernel_size=(3, 3),
            dilation_rate=(2, 2),
            padding="same",
        )(video_output)
        video_output = tf.keras.layers.BatchNormalization()(video_output)

        gamma = tf.keras.layers.Dense(filter_size)(audio_output)
        tiled_gamma = tf.keras.layers.RepeatVector(
            VIDEO_SIZE[0] // 8 * VIDEO_SIZE[1] // 8
        )(gamma)
        tiled_gamma = tf.reshape(
            tiled_gamma, (-1, VIDEO_SIZE[0] // 8, VIDEO_SIZE[1] // 8, filter_size)
        )
        video_output = tf.keras.layers.Multiply()([video_output, tiled_gamma])

        beta = tf.keras.layers.Dense(filter_size)(audio_output)
        tiled_beta = tf.keras.layers.RepeatVector(
            VIDEO_SIZE[0] // 8 * VIDEO_SIZE[1] // 8
        )(beta)
        tiled_beta = tf.reshape(
            tiled_beta, (-1, VIDEO_SIZE[0] // 8, VIDEO_SIZE[1] // 8, filter_size)
        )
        video_output = tf.keras.layers.Add()([video_output, tiled_beta])

        video_output = tf.keras.layers.ReLU()(video_output)

    video_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1))(video_output)

    model = tf.keras.models.Model(
        inputs=[audio_input_tensor, video_input], outputs=video_output
    )
    return model


# In[4]:


def get_dataset_split(split):
    examples = {}
    for file_name in os.listdir(WAVEFORMS_PATH):
        waveform_path = os.path.join(WAVEFORMS_PATH, file_name)
        if os.path.isfile(waveform_path) and file_name.endswith(".wav"):
            key = ".".join(file_name.split(".")[:-1])
            if key not in examples:
                examples[key] = {}
            examples[key]["waveform_path"] = waveform_path
    for file_name in os.listdir(IMAGES_PATH):
        image_path = os.path.join(IMAGES_PATH, file_name)
        if os.path.isfile(image_path) and file_name.endswith(".jpg"):
            key = ".".join(file_name.split(".")[:-1])
            if key not in examples:
                examples[key] = {}
            examples[key]["image_path"] = image_path
    for file_name in os.listdir(TRAIN_DENSITY_MAPS_PATH):
        density_map_path = os.path.join(TRAIN_DENSITY_MAPS_PATH, file_name)
        if os.path.isfile(density_map_path) and file_name.endswith(".mat"):
            key = ".".join(file_name.split(".")[:-1])
            if key not in examples:
                examples[key] = {}
            examples[key]["density_map_path"] = density_map_path
            examples[key]["split"] = "train"
    for file_name in os.listdir(VAL_DENSITY_MAPS_PATH):
        density_map_path = os.path.join(VAL_DENSITY_MAPS_PATH, file_name)
        if os.path.isfile(density_map_path) and file_name.endswith(".mat"):
            key = ".".join(file_name.split(".")[:-1])
            if key not in examples:
                examples[key] = {}
            examples[key]["density_map_path"] = density_map_path
            examples[key]["split"] = "val"
    for file_name in os.listdir(TEST_DENSITY_MAPS_PATH):
        density_map_path = os.path.join(TEST_DENSITY_MAPS_PATH, file_name)
        if os.path.isfile(density_map_path) and file_name.endswith(".mat"):
            key = ".".join(file_name.split(".")[:-1])
            if key not in examples:
                examples[key] = {}
            examples[key]["density_map_path"] = density_map_path
            examples[key]["split"] = "test"
    final_examples = []
    for key, info in examples.items():
        if "split" in info and info["split"] == split:
            final_examples.append(info)
    return final_examples


def visualize_data(image, density_map, spectrogram):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    fig.delaxes(ax4)
    ax1.imshow(image)
    ax2.imshow(density_map)
    ax3.imshow(spectrogram)
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    plt.show()


def get_gaussian_kernel(kernel_size, sigma):
    """
    Returns a 2D Gaussian kernel.
    from: https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    x = np.linspace(-sigma, sigma, kernel_size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


def precompute_batches(recreate=False):
    gaussian_kernel = get_gaussian_kernel(15, 4)
    split_lens = []
    resize_errors = []
    for split in ["train", "val", "test"]:
        infos = get_dataset_split(split)
        infos_len = len(infos)
        split_lens.append(infos_len)
        for index in range(infos_len):
            sys.stdout.write("\r%d" % (index + 1))
            sys.stdout.flush()

            all_path = os.path.join(CACHE_DIR, "%s_%d.pkl" % (split, index))

            if recreate or not os.path.exists(all_path):
                info = infos[index]
                image = plt.imread(info["image_path"], format="jpeg")
                resized_image = resize(image, VIDEO_SIZE)

                waveform, _ = librosa.load(
                    info["waveform_path"], sr=DISCO_SAMPLING_RATE
                )
                mel_spectrogram = preprocess_sound(waveform, DISCO_SAMPLING_RATE)
                mel_spectrogram = np.moveaxis(mel_spectrogram, 0, -1)

                head_annotation = scipy.io.loadmat(info["density_map_path"])["map"]
                density_map = signal.convolve2d(head_annotation, gaussian_kernel)
                old_size = density_map.shape
                new_size = (VIDEO_SIZE[0] // 8, VIDEO_SIZE[1] // 8)
                resize_factor = old_size[0] / new_size[0] * old_size[1] / new_size[1]
                resized_density_map = resize(density_map, new_size) * resize_factor
                resize_errors.append(
                    np.abs(np.sum(density_map) - np.sum(resized_density_map))
                )

                with open(all_path, "wb") as all_file:
                    pickle.dump(
                        {
                            "spectrogram": mel_spectrogram,
                            "image": resized_image,
                            "density_map": resized_density_map,
                        },
                        all_file,
                    )
        print()  # newline
    if resize_errors:
        print("Mean absolute resize error:", np.mean(resize_errors))
    return split_lens


class AVCCSequence(tf.keras.utils.Sequence):
    def __init__(self, split, split_len, batch_size):
        self.split = split
        self.split_len = split_len
        self.batch_size = batch_size
        self.random_permutation = np.random.permutation(self.split_len)

    def __len__(self):
        return math.ceil(self.split_len / self.batch_size)

    def on_epoch_end(self):
        self.random_permutation = np.random.permutation(self.split_len)

    def __getitem__(self, index):
        spectrograms = []
        images = []
        density_maps = []
        for random_index in self.random_permutation[
            index * self.batch_size : (index + 1) * self.batch_size
        ]:
            all_path = os.path.join(CACHE_DIR, "%s_%d.pkl" % (self.split, random_index))
            with open(all_path, "rb") as all_file:
                data = pickle.load(all_file)
                spectrograms.append(data["spectrogram"])
                images.append(data["image"])
                density_maps.append(data["density_map"])

        return [np.array(spectrograms), np.array(images)], np.array(density_maps)


# In[5]:


def l2_loss(y, y_hat):
    return tf.reduce_sum(tf.square(y - y_hat), axis=[1, 2])


def mae(y, y_hat):
    return tf.reduce_mean(
        tf.abs(tf.reduce_sum(y, axis=[1, 2]) - tf.reduce_sum(y_hat, axis=[1, 2])),
        axis=-1,
    )


# In[6]:


def lr_scheduler(epoch, lr):
    return lr * 0.99


def train_backbone(epochs):
    tf.keras.backend.clear_session()

    batch_size = 4 * NUM_GPUS
    split_lens = precompute_batches()
    train_sequence = AVCCSequence("train", split_lens[0], batch_size)
    val_sequence = AVCCSequence("val", split_lens[1], batch_size)
    test_sequence = AVCCSequence("test", split_lens[2], batch_size)

    with DISTRIBUTED_STRATEGY.scope():
        backbone_model = get_model()
        backbone_model.compile(
            optimizer=tfa.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
            loss=l2_loss,
            metrics=[mae],
        )

    lr_reduce = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    model_checkpoint_file = "av_cc_backbone_v10.h5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_checkpoint_file,
        monitor="val_mae",
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        mode="min",
        save_freq="epoch",
    )

    history = backbone_model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=epochs,
        shuffle=True,
        callbacks=[lr_reduce, checkpoint],
        verbose=1,
    )

    backbone_model.evaluate(test_sequence)
    return backbone_model


# In[ ]:


backbone_model = train_backbone(100)


# In[ ]:
