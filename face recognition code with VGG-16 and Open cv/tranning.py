from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt 

train_path = "/dataset/train/**"
test_path ="dataset/test/**"
IMAGE_SIZE = [244,244]
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights = 'imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False
        
folders = glob('E:/face recognition code with VGG-16 and Open cv/face recognition code with VGG-16 and Open cv/dataset/train/*')
folders 

x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)

model.summary()

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
    )

train_datagen = ImageDataGenerator( rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
    )
test_datagen=ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                  target_size= (244,244),
                                                  batch_size = 32,
                                                  class_mode = 'categorical'
                                                  )

test_set = test_datagen.flow_from_directory('dataset/test',
                                                  target_size= (244,244),
                                                  batch_size = 32,
                                                  class_mode = 'categorical'
                                                  )

r = model.fit(
    training_set,
    validation_data = test_set,
    epochs = 20,
    steps_per_epoch = len(training_set),
    validation_steps = len(test_set)
    )
plt.plot(r.history['loss'], label = 'train loss')
plt.plot(r.history['val_loss'], label= 'val loss')
plt.legend()
plt.show()
model.save('face_recog.h5')

