# -*- coding: utf-8 -*-
import platform as plat
import os
import time
import keras as kr
import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization ,GRU# , Flatten
from keras.layers import Lambda, TimeDistributed, Activation,Conv2D, MaxPooling2D, Add#, Merge
from keras import backend as K
from keras.optimizers import SGD, Adadelta, Adam
from keras.layers.merge import add, concatenate

MS_OUTPUT_SIZE = 1424
label_max_string_length = 64
AUDIO_LENGTH = 1600
AUDIO_FEATURE_LENGTH = 200
 
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def lstmModel():
    '''
    定义CNN/LSTM/CTC模型，使用函数式模型
    输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
    隐藏层一：3*3卷积层
    隐藏层二：池化层，池化窗口大小为2
    隐藏层三：Dropout层，需要断开的神经元的比例为0.2，防止过拟合
    隐藏层四：循环层、LSTM/GRU层
    隐藏层五：Dropout层，需要断开的神经元的比例为0.2，防止过拟合
    隐藏层六：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数，
    输出层：自定义层，即CTC层，使用CTC的loss作为损失函数，实现连接性时序多输出

    '''
    # 每一帧使用13维mfcc特征及其13维一阶差分和13维二阶差分表示，最大信号序列长度为1500
    input_data = Input(name='the_input', shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1))

    layer_ = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(input_data) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_0 = layer_
    layer_ = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)    
    layer_ = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_) # 池化层

    layer_ = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_0 = layer_
    layer_ = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_0 = layer_
    layer_ = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_ = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_) # 池化层
    
    #add
    layer_ = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_0 = layer_
    layer_ = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_0 = layer_
    layer_ = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_ = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_) # 池化层
    
    layer_ = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_0 = layer_
    layer_ = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_0 = layer_
    layer_ = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_ = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_) # 池化层    
    #add

    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_0 = layer_
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_0 = layer_    
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_ = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_) # 池化层

    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_0 = layer_
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_0 = layer_    
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_ = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_) # 池化层
    
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) 
    layer_ = BatchNormalization()(layer_)
    layer_0 = layer_
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) 
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_0 = layer_     
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) 
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_ = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_)
    
    #add
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) 
    layer_ = BatchNormalization()(layer_)
    layer_0 = layer_
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) 
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_0 = layer_    
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) 
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_ = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_)
    
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) 
    layer_ = BatchNormalization()(layer_)
    layer_0 = layer_
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) 
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_0 = layer_    
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) 
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_ = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_)
    
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) 
    layer_ = BatchNormalization()(layer_)
    layer_0 = layer_
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) 
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_0 = layer_    
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) 
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_ = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_)
    
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) 
    layer_ = BatchNormalization()(layer_)
    layer_0 = layer_
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) 
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_0 = layer_    
    layer_ = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_) 
    layer_ = BatchNormalization()(layer_)
    layer_= Add()([layer_0,layer_])
    layer_ = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_)    
    #add    

    layer_ = Reshape((200, 3200))(layer_) #Reshape层
    layer_ = Dropout(0.4)(layer_)
    layer_ = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_) # 全连接层
    layer_ = BatchNormalization()(layer_)
    layer_ = Dense(MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(layer_) # 全连接层
    layer_ = BatchNormalization()(layer_)
    y_pred = Activation('softmax', name='Activation0')(layer_)
    
    model_data = Model(inputs = input_data, outputs = y_pred)

    labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    model.summary()

    #ada_d = Adadelta(lr = 0.01, rho = 0.95, epsilon = 1e-06)
    #opt = SGD(lr=0.01, momentum=0., decay=0., nesterov=False)
    opt = Adam(lr = 0.1, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = opt)

    test_func = K.function([input_data], [y_pred])
    
    return model, model_data