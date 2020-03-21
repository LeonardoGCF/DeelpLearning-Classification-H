from IPython.core.interactiveshell import InteractiveShell
from keras.engine.training_generator import predict_generator

InteractiveShell.ast_node_interactivity = "all"

import os
import tensorflow as tf
import numpy as np
import pandas as pd

from CNNClassifier import CNNClassifier
from splitValidationSet import *

SEED = 1234
tf.random.set_seed( SEED )

# =================================================================
#             step 1 : initialized the ImageDataGenerator
# =================================================================

from tensorflow.keras.preprocessing.image import ImageDataGenerator

apply_data_augmentation = True

# Create training ImageDataGenerator object
if apply_data_augmentation:
    train_data_gen = ImageDataGenerator( rotation_range=10,
                                         width_shift_range=10,
                                         height_shift_range=10,
                                         zoom_range=0.3,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         fill_mode='constant',
                                         cval=0,
                                         rescale=1. / 255 )
else:
    train_data_gen = ImageDataGenerator( rescale=1. / 255 )

# Create validation and test ImageDataGenerator objects
valid_data_gen = ImageDataGenerator( rescale=1. / 255 )
test_data_gen = ImageDataGenerator( rescale=1. / 255 )
# =================================================================
#   step 2 : Create validation set（ only  use once to produce the set)
## =================================================================
#filePath, child_list =eachFile("Classification_Dataset/training/")
#for i in child_list:
#    path = 'Classification_Dataset/split-train/' + i
#    mkdir( path )
#    path = 'Classification_Dataset/split-validation/' + i
#    mkdir( path )
#
#train_pic_dir = []
#validation_pic_dir = []
#for i in filePath:
#    pic_dir, pic_name = eachFile1( i )
#    random.shuffle( pic_dir )
#    train_list = pic_dir[0:int( 0.8 * len( pic_dir ) )]
#    validation_list = pic_dir[int( 0.8 * len( pic_dir ) ):]
#    for j in train_list:
#        fromImage = Image.open( j )
#        j=j.replace('training', 'split-train')
#        #print('pic'+j+'stored in')
#        fromImage.save( j )
#        #print(j+'successful')
#    for k in validation_list:
#        fromImage = Image.open( k )
#        k=k.replace('training', 'split-validation')
#        #print('pic'+k+'stored in')
#        fromImage.save( k )
#        # print(k+'successful')

# =================================================================
#         step 2 : call flow_from_directory class method
# =================================================================

# Create generators to read images from dataset directory
# -------------------------------------------------------
dataset_dir = os.path.join( os.getcwd(), 'Classification_Dataset' )

# Batch size
bs = 8

# img shape
img_h = 256
img_w = 256

num_classes = 20

decide_class_indices = False
if decide_class_indices:
    classes = ['owl',  # 0
               'galaxy',  # 1
               'lightning',  # 2
               'wine-bottle',  # 3
               't-shirt',  # 4
               'waterfall',  # 5
               'sword',  # 6
               'school-bus',  # 7
               'calculator',  # 8
               'sheet-music',  # 9
               'airplanes',  # 10
               'lightbulb',  # 11
               'skyscraper',  # 12
               'mountain-bike',  # 13
               'fireworks',  # 14
               'computer-monitor',  # 15
               'bear',  # 16
               'grand-piano',  # 17
               'kangaroo',  # 18
               'laptop']  # 19
else:
    classes = None

# Training
training_dir = os.path.join( dataset_dir, 'split-train' )
train_gen = train_data_gen.flow_from_directory( training_dir,
                                                batch_size=bs,
                                                color_mode='rgb',
                                                classes=classes,
                                                class_mode='categorical',
                                                shuffle=True,
                                                seed=SEED )  # targets are directly converted into one-hot vectors

# Validation
validation_dir = os.path.join(dataset_dir, 'split-validation')
valid_gen = valid_data_gen.flow_from_directory(validation_dir,
                                               batch_size=bs,
                                               color_mode='rgb',
                                               class_mode='categorical',
                                               classes=classes,
                                               shuffle=False,
                                               seed=SEED)

# Test
test_dir = os.path.join( dataset_dir, 'test' )
test_gen = test_data_gen.flow_from_directory( test_dir,
                                              batch_size=bs,
                                              color_mode='rgb',
                                              classes=classes,
                                              class_mode='categorical',
                                              shuffle=False,
                                              seed=SEED )

# =================================================================
#         step 3 : creat a tf.data.Dataset object
# =================================================================

# Training
train_dataset = tf.data.Dataset.from_generator( lambda: train_gen,
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=([None, img_h, img_w, 3], [None, num_classes]) )
# Repeat
# Without calling the repeat function the dataset
# will be empty after consuming all the images
train_dataset = train_dataset.repeat()

# Validation
# ----------
valid_dataset =tf.data.Dataset.from_generator
valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=([None, img_h, img_w, 3], [None, num_classes]))

# Repeat
valid_dataset = valid_dataset.repeat()

# Test
# ----
test_dataset = tf.data.Dataset.from_generator( lambda: test_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]) )

# Repeat
test_dataset = test_dataset.repeat()

# =================================================================
#         step 4 : parameters for the model
# =================================================================

depth = 5
start_f = 8

# Create Model instance
model = CNNClassifier( depth=depth,
                       start_f=start_f,
                       num_classes=num_classes )

# Build Model (Required)
model.build( input_shape=(None, img_h, img_w, 3) )

## Visualize created model as a table
# model.feature_extractor.summary()
#
## Visualize initialized weights
# print(model.weights)

# Optimization params
# Loss
loss = tf.keras.losses.CategoricalCrossentropy()

# learning rate
lr = 1e-3
optimizer = tf.keras.optimizers.Adam( learning_rate=lr )
# -------------------

# Validation metrics
# ------------------

metrics = ['accuracy']
# ------------------

# Compile Model
model.compile( optimizer=optimizer, loss=loss, metrics=metrics )

# =================================================================
#         step 5 : training with call back
# =================================================================
from datetime import datetime
import os
from datetime import datetime

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

cwd = os.getcwd()

exps_dir = os.path.join( cwd, 'test_experiments' )
if not os.path.exists( exps_dir ):
    os.makedirs( exps_dir )

now = datetime.now().strftime( '%b%d_%H-%M-%S' )

model_name = 'CNN'

exp_dir = os.path.join( exps_dir, model_name + '_' + str( now ) )
if not os.path.exists( exp_dir ):
    os.makedirs( exp_dir )

callbacks = []

# Model checkpoint
# ----------------
ckpt_dir = os.path.join( exp_dir, 'ckpts' )
if not os.path.exists( ckpt_dir ):
    os.makedirs( ckpt_dir )

ckpt_callback = tf.keras.callbacks.ModelCheckpoint( filepath=os.path.join( ckpt_dir, 'cp_{epoch:02d}.ckpt' ),
                                                    save_weights_only=True )  # False to save the model directly
callbacks.append( ckpt_callback )

# Visualize Learning on Tensorboard
# ---------------------------------
tb_dir = os.path.join( exp_dir, 'tb_logs' )
if not os.path.exists( tb_dir ):
    os.makedirs( tb_dir )

# By default shows losses and metrics for both training and validation
tb_callback = tf.keras.callbacks.TensorBoard( log_dir=tb_dir,
                                              profile_batch=0,
                                              histogram_freq=1 )  # if 1 shows weights histograms
callbacks.append( tb_callback )

# Early Stopping
# --------------
early_stop = False
if early_stop:
    es_callback = tf.keras.callback.EarlyStopping( monitor='val_loss', patience=10 )
    callbacks.append( es_callback )

model.fit( x=train_dataset,
           epochs=100,  #### set repeat in training dataset
           steps_per_epoch=len( train_gen ),
           validation_data=valid_dataset,
           validation_steps=len(valid_gen),
           callbacks=callbacks )

# How to visualize Tensorboard

# 1. tensorboard --logdir EXPERIMENTS_DIR --port PORT     <- from terminal
# 2. localhost:PORT   <- in your browser


# img = os.path.join( test_dir, 'IMG_48' )
# img_arr = np.expand_dims( np.array( img ), 0 )
# print(tf.argmax(model.predict(x=img_arr, verbose=0), -1))


predict_class = model.predict_generator( test_gen, steps=len( test_gen ), verbose=0 )
predicted_class = tf.argmax( predict_class, -1 )

# =================================================================
#              step 7 : output the RESULT
# =================================================================

# test_img_dir = os.path.join( test_dir,'test_data' )
#img_filenames = next( os.walk( 'Classification_Dataset/test/test_data' ) )[2]
img_filenames = test_gen.filenames

match_name = []
for i in test_gen.filenames:
  path = os.path.dirname(i)
  filename = i[len(path)+1: ]
  match_name.append(filename)


#matching the classes to the true predicted classes
matching_class = []
pn = predicted_class.numpy()
# print(pn)

for i in range(len(pn)):
  if(pn[i] == 0) : matching_class.append(10)
  elif(pn[i] == 1): matching_class.append(16)
  elif(pn[i] == 2): matching_class.append(8)
  elif(pn[i] == 3): matching_class.append(15)
  elif(pn[i] == 4): matching_class.append(14)
  elif(pn[i] == 5): matching_class.append(1)
  elif(pn[i] == 6): matching_class.append(17)
  elif(pn[i] == 7): matching_class.append(18)
  elif(pn[i] == 8): matching_class.append(19)
  elif(pn[i] == 9): matching_class.append(11)
  elif(pn[i] == 10): matching_class.append(2)
  elif(pn[i] == 11): matching_class.append(13)
  elif(pn[i] == 12): matching_class.append(0)
  elif(pn[i] == 13): matching_class.append(7)
  elif(pn[i] == 14): matching_class.append(9)
  elif(pn[i] == 15): matching_class.append(12)
  elif(pn[i] == 16): matching_class.append(6)
  elif(pn[i] == 17): matching_class.append(4)
  elif(pn[i] == 18): matching_class.append(5)
  elif(pn[i] == 19): matching_class.append(3)



df = pd.DataFrame( {'Id': match_name, 'Category': matching_class} )
df.to_csv( 'submission.csv', index=False )


