# Generated from: MedGen.ipynb
# Converted at: 2026-03-06T04:22:48.634Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Setup


# Functionality
import os
from glob import glob
from google.colab import files

# Basics
import numpy as np
import pandas as pd

# Visualization
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# Python ≥3.5
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20
import sklearn
assert sklearn.__version__ >= "0.20"

try:
    %tensorflow_version 2.x
    IS_COLAB = True
except Exception:
    IS_COLAB = False

# TensorFlow ≥2.0
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. CNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")

# Install Kaggle library
!pip install -q kaggle
!mkdir -p ~/.kaggle

# Upload Kaggle API key file
files.upload()

# Download our dataset
!cp kaggle.json ~/.kaggle/
!kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

# Unzip it
!unzip skin-cancer-mnist-ham10000.zip

# # Processing


# Set base directory, importing to later identify image paths
base_skin_dir = "/content/"

# Take a glimpse at the metadata of our images
df_meta = pd.read_csv(os.path.join(base_skin_dir,'HAM10000_metadata.csv'))
df_meta.head()

# Functionality so that we can find the image path for each metadata entry
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

# Functionality so that we can identify the type of lesion
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# First of all, we read in our metadata
tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

# Secondly, we map the image path directory for the entry
tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)

# Thirdly, we specify the type of lesion for the entry
tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get)

# Finally, we encode the type of lesion - to later use as label as onehotencoded
tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes
tile_df.sample(3)





# Overview of our data
tile_df.describe(exclude=[np.number])

# Now we look at the distribution of our dataset, to identify which
# categories we will use for our CNN classification and category for GAN
fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))
tile_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)

# Let's choose Melanoma, benign lesions, and the carcinoma for our CNN and
# the carcinoma for our GAN synthetic data generation, due to its having
# significantly less data in proportion to the former two categories.
categories = ['mel', 'bkl', 'bcc']
tile_df = tile_df[tile_df.dx.isin(categories)]

# Load in all of the images for each entry
from skimage.io import imread
tile_df['image'] = tile_df['path'].map(imread)

# Image size distributions
tile_df['image'].map(lambda x: x.shape).value_counts()

# We sample 5 images from each category for visual inspection
n_samples = 5 # How many samples we want to see from each category

# Code for visualization
fig, m_axs = plt.subplots(3, n_samples, figsize = (4*n_samples, 3*3))
for n_axs, (type_name, type_rows) in zip(m_axs,
                                         tile_df.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=2018).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)

# RGB means for greater insight into our dataset
rgb_info_df = tile_df.apply(lambda x: pd.Series({'{}_mean'.format(k): v for k, v in
                                  zip(['Red', 'Green', 'Blue'],
                                      np.mean(x['image'], (0, 1)))}),1)
gray_col_vec = rgb_info_df.apply(lambda x: np.mean(x), 1)
for c_col in rgb_info_df.columns:
    rgb_info_df[c_col] = rgb_info_df[c_col]/gray_col_vec
rgb_info_df['Gray_mean'] = gray_col_vec
rgb_info_df.sample(3)

# Visualization of the RGB means, for greater insight into our dataset
for c_col in rgb_info_df.columns:
    tile_df[c_col] = rgb_info_df[c_col].values

sns.pairplot(tile_df[['Red_mean', 'Green_mean', 'Blue_mean', 'Gray_mean', 'cell_type']],
             hue='cell_type', plot_kws = {'alpha': 0.5})

# # Data Preparation


# Global parameters
TEST_IMG_COUNT = 700
IMG_SIZE = (299, 299) # [(224, 224), (384, 384), (512, 512), (640, 640)]
BATCH_SIZE = 64 # [1, 8, 16, 24]
EPOCHS = 2000
RGB_FLIP = 1 # should RGB be flipped when rendering images
#SAMPLE_PER_GROUP = 300

from sklearn.model_selection import train_test_split

# Splitting up our data into training and testing set
df_train, df_test = train_test_split(tile_df, # Feel free to add stratification
                                     test_size = 0.3)

df_train = df_train.reset_index(drop = True)
# Possibly ensure that the size is divisible by batch size if errors ensue

# Uncomment if you wish to have equal sample groups
#df_train = df_train.groupby('cell_type').\
#                    apply(lambda x: x.sample(SAMPLE_PER_GROUP,
#                                             replace=True)).\
#                    reset_index(drop = True)

# df_train = df_train.reset_index(drop=True)
# df_test = df_test.reset_index(drop=True)

# Checking volume for each category after the split
print("train mel: " + str(len(df_train[df_train.dx == "mel"])) + # Melanoma
      "\ntrain bkl: " + str(len(df_train[df_train.dx == "bkl"])) + # Beign keratosis-like lesions
      "\ntrain bcc: " + str(len(df_train[df_train.dx == "bcc"])) + # Basal cell carcinoma
      "\ntrain size: " + str(df_train.shape[0]))

print("\ntest mel: " + str(len(df_test[df_test.dx == "mel"])) +
      "\ntest bkl: " + str(len(df_test[df_test.dx == "bkl"])) +
      "\ntest bcc: " + str(len(df_test[df_test.dx == "bcc"])) +
      "\ntest size: " + str(df_test.shape[0]))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Arguments for our training data generator
dg_args = dict(featurewise_center = False,
               samplewise_center = False,
               rescale = 1./255,
               rotation_range = 45,
               width_shift_range = 0.1,
               height_shift_range = 0.1,
               shear_range = 0.01,
               zoom_range = [0.9, 1.25],
               brightness_range = [0.7, 1.3],
               horizontal_flip = True,
               vertical_flip = False,
               fill_mode = 'reflect',
               data_format = 'channels_last'
#               preprocessing_function = preprocess_input
               )

# Arguments for our testing data generator
test_args = dict(rescale = 1./255,
                 fill_mode = 'reflect',
                 data_format = 'channels_last'
#                 preprocessing_function = preprocess_input
                 )

# Setting up the image data generators
core_idg = ImageDataGenerator(**dg_args)
test_idg = ImageDataGenerator(**test_args)

# Consult Keras Image Preprocessing documentation,
# specifically section "flow_from_dataframe"
# https://keras.io/preprocessing/image/

def flow_from_dataframe(img_data_gen, raw_df, path_col, y_col, **dflow_args):

    in_df = raw_df.copy()
    in_df[path_col] = in_df[path_col].map(str)
    in_df[y_col] = in_df[y_col].map(lambda x: np.array(x))
    df_gen = img_data_gen.flow_from_dataframe(in_df,
                                              x_col=path_col,
                                              y_col=y_col,
                                              class_mode = 'raw',
                                              **dflow_args)

    # posthoc correction
    df_gen._targets = np.stack(df_gen.labels, 0)
    return df_gen

# Setting up training data generator and loader
train_gen = flow_from_dataframe(core_idg, df_train,
                                path_col = 'path',
                                y_col = 'cell_type_idx',
                                target_size = IMG_SIZE,
                                color_mode = 'rgb',
                                batch_size = BATCH_SIZE)

# Setting up testing data generator and loader
test_x, test_y = next(flow_from_dataframe(test_idg,
                                            df_test,
                                            path_col = 'path',
                                            y_col = 'cell_type_idx',
                                            target_size = IMG_SIZE,
                                            color_mode = 'rgb',
                                            batch_size = TEST_IMG_COUNT))

# Checking shape
print(test_x.shape, test_y.shape)

from skimage.util import montage

# Acquire next batch of data from our generator
t_x, t_y = next(train_gen)

# Print out their shapes
print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())

# Visualize them using the montage utility
fig, (ax1) = plt.subplots(1, 1, figsize = (10, 10))
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
ax1.imshow(montage_rgb((t_x-t_x.min())/(t_x.max()-t_x.min()))[:, :, ::RGB_FLIP])
ax1.set_title('images')

# Show the labels for this batch
print(t_y, ", labels")

# # Convolutional Neural Network Setup


# Functionality to ease the clutter of constructing the CNN
from functools import partial
DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")

# Convolutional neural network construction
model = keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=10, activation='softmax'),
])

from sklearn.model_selection import train_test_split
import numpy as np

# Load images and labels from the dataframe
X = np.stack(tile_df['image'].values)
y = tile_df['cell_type_idx'].values

# Resize images to match CNN input shape (28, 28)
from skimage.transform import resize
from skimage.color import rgb2gray
X = np.array([resize(rgb2gray(img), (28, 28, 1)) for img in X])

# Split into train, valid, test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(X_train.shape, X_valid.shape, X_test.shape)

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=200, validation_data=[X_valid, y_valid])
score = model.evaluate(X_test, y_test)

# # Deep Convolutional GAN


def plot_multiple_images(images, n_cols=8):
    n_cols = min(len(images), n_cols)
    n_rows = (len(images) - 1) // n_cols + 1
    images = images.numpy() if hasattr(images, 'numpy') else images
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            img = images[idx].squeeze()
            ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.tight_layout()

codings_size = 100

# Generator construction
generator = keras.models.Sequential([
    keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
    keras.layers.Reshape([7, 7, 128]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME",
                                 activation="selu"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="SAME",
                                 activation="tanh"),
])

# Discriminator construction
discriminator = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME",
                        activation=keras.layers.LeakyReLU(0.2),
                        input_shape=[28, 28, 1]),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME",
                        activation=keras.layers.LeakyReLU(0.2)),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation="sigmoid")
])

# Final construction and compilation
dcgan = keras.models.Sequential([generator, discriminator])
discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
dcgan.compile(loss="binary_crossentropy", optimizer="rmsprop")

batch_size = 64
X_train = X_train.astype('float32')
dataset = tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size, drop_remainder=True)

# Function to train our DCGAN
def train_gan(gan, dataset, batch_size, codings_size, n_epochs=200):
    generator, discriminator = gan.layers

    # may the epochs commence
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))
        for X_batch in dataset:

            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)

            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)

        plot_multiple_images(generated_images, 8) # generate images for viz.
        plt.show() # visualize our generated images

# Training our DCGAN
train_gan(dcgan, dataset, batch_size, codings_size, n_epochs=2000)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('CNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Compare real vs generated pixel distributions
real_batch_np = real_batch.numpy().flatten()
fake_batch_np = generated_batch.numpy().flatten()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(real_batch_np, bins=50, color='steelblue', alpha=0.7, label='Real')
plt.hist(fake_batch_np, bins=50, color='salmon', alpha=0.7, label='Generated')
plt.title('Real vs Generated Pixel Distribution')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(['DCGAN'], [fid_score], color='steelblue', width=0.3)
plt.title(f'FID Score: {fid_score:.2f} (lower is better)')
plt.ylabel('FID Score')
plt.ylim(0, 20)

plt.tight_layout()
plt.show()

# Generate new synthetic images
n_images = 64
noise = tf.random.normal(shape=[n_images, codings_size])
generated_images = generator(noise, training=False)

# Plot them
fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for idx, ax in enumerate(axes.flat):
    img = generated_images[idx].numpy().squeeze()
    ax.imshow(img, cmap='gray')
    ax.axis('off')

plt.suptitle('Synthetically Generated Skin Lesion Images', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('generated_skin_lesions.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved as generated_skin_lesions.png")