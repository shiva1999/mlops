{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "from keras.layers import Convolution2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.layers import MaxPooling2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.layers import Flatten"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.layers import Dense"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import Sequential"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.preprocessing.image import ImageDataGenerator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Convolution2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(64,64,3) ))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(MaxPooling2D())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Flatten())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Dense(units=128, activation= 'relu' ))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_1\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_1 (Conv2D)            (None, 62, 62, 32)        896       \n",
      "_________________________________________________________________\n",
      "max_pooling2d_1 (MaxPooling2 (None, 31, 31, 32)        0         \n",
      "_________________________________________________________________\n",
      "flatten_1 (Flatten)          (None, 30752)             0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 128)               3936384   \n",
      "=================================================================\n",
      "Total params: 3,937,280\n",
      "Trainable params: 3,937,280\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Dense(units=3, activation= 'softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_1\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_1 (Conv2D)            (None, 62, 62, 32)        896       \n",
      "_________________________________________________________________\n",
      "max_pooling2d_1 (MaxPooling2 (None, 31, 31, 32)        0         \n",
      "_________________________________________________________________\n",
      "flatten_1 (Flatten)          (None, 30752)             0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 128)               3936384   \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 3)                 387       \n",
      "=================================================================\n",
      "Total params: 3,937,667\n",
      "Trainable params: 3,937,667\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.optimizers import Adam"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "optim= Adam(learning_rate=0.001)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer= optim, loss= 'binary_crossentropy', metrics= ['accuracy'], )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_datagen= ImageDataGenerator(rescale= 1./255 ,zoom_range= 0.2, horizontal_flip= True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_datagen= ImageDataGenerator(rescale= 1./255)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 2707 images belonging to 3 classes.\n"
     ]
    }
   ],
   "source": [
    "train_set= train_datagen.flow_from_directory('C:/Users/Dell/Desktop/mlops/animals_classification/animals/train_animals', target_size=(64,64),class_mode='categorical',\n",
    "    batch_size=32 )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 293 images belonging to 3 classes.\n"
     ]
    }
   ],
   "source": [
    "test_set= test_datagen.flow_from_directory('C:/Users/Dell/Desktop/mlops/animals_classification/animals/test_animals', \n",
    "                                           target_size=(64,64),\n",
    "                                           class_mode='categorical',\n",
    "                                           batch_size=32 )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.callbacks import ModelCheckpoint , EarlyStopping"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "checkpoint= ModelCheckpoint('checkpoint.h5',\n",
    "                            monitor='val_loss',\n",
    "                            verbose=1,\n",
    "                            mode= 'min')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "earlystop= EarlyStopping(monitor='val_loss',\n",
    "                         min_delta=0,\n",
    "                         patience=10,\n",
    "                         verbose=1,\n",
    "                         restore_best_weights= True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "callbacks= [earlystop, checkpoint]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_samples= 2706"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_samples= 294"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "902/902 [==============================] - 307s 340ms/step - loss: 0.3670 - accuracy: 0.8177 - val_loss: 0.4590 - val_accuracy: 0.8186\n",
      "Epoch 2/10\n",
      "902/902 [==============================] - 274s 304ms/step - loss: 0.2549 - accuracy: 0.8841 - val_loss: 0.6597 - val_accuracy: 0.8260\n",
      "Epoch 3/10\n",
      "902/902 [==============================] - 272s 302ms/step - loss: 0.1881 - accuracy: 0.9201 - val_loss: 0.2779 - val_accuracy: 0.8543\n",
      "Epoch 4/10\n",
      "902/902 [==============================] - 274s 304ms/step - loss: 0.1306 - accuracy: 0.9476 - val_loss: 1.0949 - val_accuracy: 0.8247\n",
      "Epoch 5/10\n",
      "902/902 [==============================] - 284s 315ms/step - loss: 0.0916 - accuracy: 0.9654 - val_loss: 0.0060 - val_accuracy: 0.8328\n",
      "Epoch 6/10\n",
      "902/902 [==============================] - 320s 355ms/step - loss: 0.0670 - accuracy: 0.9752 - val_loss: 0.6202 - val_accuracy: 0.8295\n",
      "Epoch 7/10\n",
      "902/902 [==============================] - 302s 335ms/step - loss: 0.0494 - accuracy: 0.9826 - val_loss: 0.6965 - val_accuracy: 0.8321\n",
      "Epoch 8/10\n",
      "902/902 [==============================] - 277s 307ms/step - loss: 0.0412 - accuracy: 0.9855 - val_loss: 1.0565 - val_accuracy: 0.8341\n",
      "Epoch 9/10\n",
      "902/902 [==============================] - 326s 362ms/step - loss: 0.0318 - accuracy: 0.9888 - val_loss: 0.9938 - val_accuracy: 0.8271\n",
      "Epoch 10/10\n",
      "902/902 [==============================] - 284s 315ms/step - loss: 0.0336 - accuracy: 0.9886 - val_loss: 0.2688 - val_accuracy: 0.8225\n"
     ]
    }
   ],
   "source": [
    "record= model.fit(\n",
    "        train_set,\n",
    "        steps_per_epoch= train_samples // 3,\n",
    "        epochs=9,\n",
    "        validation_data=test_set,\n",
    "        validation_steps= test_samples // 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "acc= record.history['accuracy']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "length= len(acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.98858136\n"
     ]
    }
   ],
   "source": [
    "final_acc= acc[length-1]\n",
    "print(final_acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "98.85813593864441\n"
     ]
    }
   ],
   "source": [
    "final_acc_new= 100 * final_acc\n",
    "print(final_acc_new)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "final_acc_str= str(final_acc_new)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "d= open(\"C:/Users/Dell/Desktop/mlops/acc.txt\", \"w\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "17"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "d.write(final_acc_str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "d.close()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
