# -*- coding: utf-8 -*-
import numpy as np
from process import *
from keras import backend as K
import tensorflow as tf
from keras.layers import Input
import os
import time
from skip_model import lstmModel

_model,base_model = lstmModel()
_model.load_weights('./model/asr44.model')
base_model.load_weights('./model/asr.model44.base')
vocab = GetSymbolList()

data = ['1','2','3','4','5','6','7','8','9','10']

start = 0#9500
end = 16015#25015
x = base_model.output    # [batch_sizes, series_length, classes]
input_length = Input(batch_shape=[None], dtype='int32')
ctc_decode = K.ctc_decode(x, input_length=input_length,greedy = True, beam_width=100, top_paths=2)
decode = K.function([base_model.input, input_length], [ctc_decode[0][0]])
lens = []

for j in range(1):
    wavsignal_all,fs = read_wav_data('./1.wav')
    wavsignal = wavsignal_all[:,start:end]
    data_input = GetFrequencyFeature(wavsignal,fs)    
    data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
    if 1600 < len(data_input):
        data_input = data_input[0:1600,:,:]
    input_len = data_input.shape[0] // 8   
    X = np.zeros((1600, 200, 1), dtype = np.float)
    X[0:len(data_input)] = data_input
    in_len = np.zeros((1),dtype = np.int32)        
    in_len[0] = input_len        
    x_in = X.reshape(1,X.shape[0],X.shape[1],1)
    
    out = decode([x_in, in_len])
    
    #base_pred = base_model.predict(x = x_in)
    #r = K.ctc_decode(base_pred, in_len, greedy = True, beam_width=100, top_paths=2)    
    #r1 = K.get_value(r[0][0])
    #pre = r1[0]
    
    pre = out[0][0]
    pinyin = []
    for i in pre:
        pinyin.append(vocab[i])
    print('语音转拼音结果：','/'.join(pinyin))