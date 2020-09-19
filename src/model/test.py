import tensorflow as tf

from tensorflow_examples.models.pix2pix import pix2pix

from IPython.display import clear_output
import matplotlib.pyplot as plt

from ..data import data


def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask


@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


#TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
#BUFFER_SIZE = 1000
#STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

#train = dataset['train'].map(
    #load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#test = dataset['test'].map(load_image_test)

#train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
#train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#test_dataset = test.batch(BATCH_SIZE)


################################################################################
image_gen = data.default_ImageDataGenerator(validation_split=0.15)

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    cnt = (i for i in range(1, 10000000000))

    print(len(display_list))
    for i, image in enumerate(display_list):
        for c in range(image.shape[2]):
            print(i, c)
            plt.subplot(len(display_list), image.shape[2], next(cnt))
            plt.title(title[c])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(image[:,:,c:c+1]))
            plt.axis('off')
    plt.show()


if 1:
    images = data.load_images_w_masks(data.dirs.mitochondria)
    images = image_gen.flow_from_generator(images, shuffle=False)
    #display_list = [*next(images)]
    #display(display_list)
################################################################################


######################
##                  ##
## DEFINE THE MODEL ##
##                  ##
######################

OUTPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
#base_model = tf.keras.applications.MobileNetV2(input_shape=[256, 256, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False


up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  #inputs = tf.keras.layers.Input(shape=[512, 512, 1])
  x = tf.keras.layers.Conv2D(3, 3, strides=2, padding='same')(inputs)
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  #for _ in range(2): # 128 -> 256 -> 512
      # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


######################
##                  ##
##  TRAIN THE MODEL ##
##                  ##
######################

if 1:
    model = unet_model(OUTPUT_CHANNELS)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


if 1:
    tf.keras.utils.plot_model(model, "test_plot_model.png", show_shapes=True)


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  #else:
  #  display([sample_image, sample_mask,
  #           create_mask(model.predict(sample_image[tf.newaxis, ...]))])


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    if 0: show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 20
VAL_SUBSPLITS = 5
#VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

images = data.load_images_w_masks(data.dirs.mitochondria)
images = image_gen.flow_from_generator(images, shuffle=False)
#x, y = images

model_history = model.fit(
    #train_dataset, epochs=EPOCHS,
    images, epochs=EPOCHS,
    #steps_per_epoch=STEPS_PER_EPOCH,
    #validation_steps=VALIDATION_STEPS,
    #validation_data=test_dataset,
    #callbacks=[DisplayCallback()]
)


loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()


########################
##                    ##
##  MAKE PREDICTIONS  ##
##                    ##
########################

show_predictions(test_dataset, 3)

