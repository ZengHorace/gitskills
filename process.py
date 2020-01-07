# -*- coding: utf-8 -*-
import os
import wave
import numpy  
import math
import time
from scipy.fftpack import fft
import scipy.io.wavfile as wav
from collections import Counter
import difflib
import collections
from scipy import signal
import numpy as np

sample_freq = 16000
freq_threshold = [600 * 2 / sample_freq, 2800 * 2 / sample_freq]
hipass_filter = signal.butter(8, freq_threshold[0], 'highpass')
lopass_filter = signal.butter(8, freq_threshold[1], 'lowpass')

error_obj=open('errorpath.txt','r',encoding='UTF-8')
error_text=error_obj.read()
error_lines=error_text.split('\n')
error_num=[]
for i in error_lines:
    error_l=i.split('\t')
    error_num.append(error_l[0])
error_obj.close()


def get_wav_list(filename):
    txt_obj=open(filename,'r')
    txt_text=txt_obj.read()
    txt_lines=txt_text.split('\n')
    dic_filelist={}
    list_wavmark=[]
    for i in txt_lines:
        if(i!=''):
            txt_l=i.split(' ')
            dic_filelist[txt_l[0]] = txt_l[1]
            if txt_l[0] not in error_num:
                list_wavmark.append(txt_l[0])
    txt_obj.close()
    return dic_filelist,list_wavmark

def get_wav_symbol(filename):
    txt_obj=open(filename,'r',encoding='utf-8')
    txt_text=txt_obj.read()
    txt_lines=txt_text.split('\n')
    dic_symbol_list={}
    list_symbolmark=[]
    for i in txt_lines:
        if(i!=''):
            txt_l=i.split(' ')
            dic_symbol_list[txt_l[0]]=txt_l[1:]
            list_symbolmark.append(txt_l[0])
    txt_obj.close()
    return dic_symbol_list

def get_list(filename):
    txt_obj=open(filename,'r',encoding='utf-8')
    txt_text=txt_obj.read()
    txt_lines=txt_text.split('\n')
    dic_filelist={}
    dic_symbollist={}
    list_wavmark=[]
    if 'wav' in filename:
        m = 1
        for i in txt_lines:
            txt_l=i.split('\t')
            dic_filelist[txt_l[0]] = txt_l[1]
            dic_symbollist[txt_l[0]] = txt_l[2].split(' ')
            list_wavmark.append(txt_l[0])    
    else:
        m = 1
        for i in txt_lines:
            txt_l=i.split('\t')
            dic_filelist[txt_l[0]] = txt_l[1]
            dic_symbollist[txt_l[0]] = txt_l[2].split(' ')
            #if txt_l[0] not in error_num:
            list_wavmark.append(txt_l[0])    
    txt_obj.close()
    return dic_filelist,list_wavmark,dic_symbollist

def data_list(type):
    if(type=='train'):
        #filename_wavlist_thchs30 = 'dataset/thchs30' + '/' + 'train.wav.lst'
        #filename_wavlist_stcmds = 'dataset/st-cmds' + '/' + 'train.wav.txt'
        #filename_wavlist_aishell = 'dataset/aishell' + '/' + 'train.wav.lst'
        #filename_wavlist_primewords = 'dataset/primewords' + '/' + 'train.wav.lst'        
        #filename_symbollist_thchs30 = 'dataset/thchs30' + '/' + 'train.syllable.txt'
        #filename_symbollist_stcmds = 'dataset/st-cmds' + '/' + 'train.syllable.txt'
        #filename_symbollist_aishell = 'dataset/aishell' + '/' + 'train.syllabel.txt'
        #filename_symbollist_primewords = 'dataset/primewords' + '/' + 'train.syllabel.txt'
        filename_list_aishell = 'datalist' + '/' + 'aishell.txt'
        filename_list_prime = 'datalist' + '/' + 'prime.txt'
        filename_list_stcmd = 'datalist' + '/' + 'stcmd.txt'
        filename_list_thchs = 'datalist' + '/' + 'thchs.txt'        
        filename_list_magicdata = 'datalist' + '/' + 'magicdata.txt'
        filename_list_aidatatang_200 = 'datalist' + '/' + 'aidatatang_200.txt'        
    elif(type=='dev'):
        filename_wavlist_thchs30 = 'dataset/thchs30' + '/' + 'cv.wav.lst'
        filename_wavlist_stcmds = 'dataset/st-cmds' + '/' + 'dev.wav.txt'
        filename_symbollist_thchs30 = 'dataset/thchs30' + '/' + 'cv.syllable.txt'
        filename_symbollist_stcmds = 'dataset/st-cmds' + '/' + 'dev.syllable.txt'
        filename_list_wav = 'dataset/wav' + '/' + 'wav.txt'
    dic_wavlist = {}
    dic_symbollist = {}
    if(type=='train'):
        #dic_wavlist_1,numlist_1 = get_wav_list(filename_wavlist_thchs30)
        #dic_symbollist_1 = get_wav_symbol(filename_symbollist_thchs30)    
        #dic_wavlist_2,numlist_2 = get_wav_list(filename_wavlist_stcmds)
        #dic_symbollist_2 = get_wav_symbol(filename_symbollist_stcmds)
        #dic_wavlist_3,numlist_3 = get_wav_list(filename_wavlist_aishell)
        #dic_symbollist_3 = get_wav_symbol(filename_symbollist_aishell)    
        #dic_wavlist_4,numlist_4 = get_wav_list(filename_wavlist_primewords)
        #dic_symbollist_4 = get_wav_symbol(filename_symbollist_primewords)
        dic_wavlist_1,numlist_1,dic_symbollist_1 = get_list(filename_list_aishell)
        dic_wavlist_2,numlist_2,dic_symbollist_2 = get_list(filename_list_prime)
        dic_wavlist_3,numlist_3,dic_symbollist_3 = get_list(filename_list_stcmd)
        dic_wavlist_4,numlist_4,dic_symbollist_4 = get_list(filename_list_thchs)        
        dic_wavlist_5,numlist_5,dic_symbollist_5 = get_list(filename_list_magicdata)
        dic_wavlist_6,numlist_6,dic_symbollist_6 = get_list(filename_list_aidatatang_200)
        dic_wavlist.update(dic_wavlist_1)
        dic_wavlist.update(dic_wavlist_2)
        dic_wavlist.update(dic_wavlist_3)
        dic_wavlist.update(dic_wavlist_4)
        dic_wavlist.update(dic_wavlist_5)
        dic_wavlist.update(dic_wavlist_6)
        dic_symbollist.update(dic_symbollist_1)
        dic_symbollist.update(dic_symbollist_2)
        dic_symbollist.update(dic_symbollist_3)
        dic_symbollist.update(dic_symbollist_4)
        dic_symbollist.update(dic_symbollist_5)
        dic_symbollist.update(dic_symbollist_6)
        numlist = numlist_1 + numlist_2 + numlist_3 + numlist_4 + numlist_5 + numlist_6       
    elif(type=='dev'):
        dic_wavlist_1,numlist_1 = get_wav_list(filename_wavlist_thchs30)
        dic_symbollist_1 = get_wav_symbol(filename_symbollist_thchs30)    
        dic_wavlist_2,numlist_2 = get_wav_list(filename_wavlist_stcmds)
        dic_symbollist_2 = get_wav_symbol(filename_symbollist_stcmds)
        dic_wavlist_7,numlist_7,dic_symbollist_7 = get_list(filename_list_wav)
        dic_wavlist.update(dic_wavlist_1)
        dic_wavlist.update(dic_wavlist_2)
        dic_wavlist.update(dic_wavlist_7)
        dic_symbollist.update(dic_symbollist_1)
        dic_symbollist.update(dic_symbollist_2)
        dic_symbollist.update(dic_symbollist_7)
        numlist = numlist_7# + numlist_2
    return dic_wavlist,dic_symbollist,numlist
  
def GetSymbolList():
    txt_obj=open('dict.txt','r',encoding='UTF-8')
    txt_text=txt_obj.read()
    txt_lines=txt_text.split('\n')
    list_symbol=[]
    for i in txt_lines:
        if(i!=''):
            txt_l=i.split('\t')
            list_symbol.append(txt_l[0])
    txt_obj.close()
    list_symbol.append('_')
    return list_symbol

def read_wav_data(filename):
    wav = wave.open(filename,"rb") # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes() # 获取帧数
    num_channel=wav.getnchannels() # 获取声道数
    framerate=wav.getframerate() # 获取帧速率
    num_sample_width=wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame) # 读取全部的帧
    wav.close() # 关闭流
    wave_data = np.fromstring(str_data, dtype = np.short) # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T # 将矩阵转置
    return wave_data, framerate

x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) ) # 汉明窗

def GetFrequencyFeature(wavsignal, fs):
    time_window = 25 # 单位ms
    wavsignal = wavsignal[0]
    pre_emphasis = 0.97
    wavsignal = np.append(wavsignal[0], wavsignal[1:] - pre_emphasis * wavsignal[:-1])
    window_length = int(fs / 1000 * time_window) # 计算窗长度的公式，目前全部为400固定值
    wav_arr = np.array(wavsignal)
    wav_length = wav_arr.shape[0]
    range0_end = int(len(wavsignal)/fs*1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, window_length // 2), dtype = np.float) # 用于存放最终的频率特征数据
    data_line = np.zeros((1, window_length), dtype = np.float)
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[p_start:p_end]
        data_line = data_line * w # 加窗
        data_line = np.abs(fft(data_line)) / wav_length
        data_input[i]=data_line[0: window_length // 2] # 设置为400除以2的值（即200）是取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
    return data_input

def word2id(line, vocab):
    feat_out=[]
    for i in line:
        if(''!=i and '?' != i and '。' != i and '，' != i and '？' != i and 'ａ' != i and 'ｂ' != i and 'ｃ' != i and 'ｋ' != i and 'ｔ' != i and '/' != i and '"' != i and '》' != i and '《' != i and '：' != i and '；' != i and ' ' != i and '？' != i and '，' != i and '。' != i and '！' != i and '?' != i and '“' != i and '”' != i and ',' != i and '·' != i and '…' != i):
            n=vocab.index(i)
            feat_out.append(n)
    data_label = np.array(feat_out)
    return data_label

def wav_padding(batch_size,wav_data_lst):
    wav_lens = [len(data) for data in wav_data_lst]
    wav_max_len = 1600#max(max(wav_lens),800)
    x = np.zeros((batch_size, wav_max_len, 200, 1), dtype = np.float)
    for i in range(len(wav_data_lst)):
        x[i,0:len(wav_data_lst[i])] = wav_data_lst[i]
    return x

def label_padding(batch_size,label_data_lst):
    label_lens = np.array([len(label) for label in label_data_lst])
    max_label_len = 64#max(label_lens)
    y = np.zeros((batch_size, max_label_len), dtype=np.int16)
    for i in range(len(label_data_lst)):        
        y[i,0:len(label_data_lst[i])] = label_data_lst[i]
    return y

def data_generatornew(dataset,batch_size, shuffle_list, dic_wavlist, dic_symbollist, numlist, vocab):
    labels = np.zeros((batch_size,1), dtype = np.float)
    while True:
        for i in range(len(dic_wavlist)//batch_size):
            wav_data_lst = []
            label_data_lst = []
            input_length = []
            label_length = []
            begin = i * batch_size
            end = begin + batch_size
            sub_list = shuffle_list[begin:end]
            for index in sub_list:
                data_input = compute_log_mel_fbank(dataset + '/' + dic_wavlist[numlist[index]])
                data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
                if 1600 < len(data_input):
                    data_input = data_input[:1600,:,:]
                data_label = word2id(dic_symbollist[numlist[index]], vocab)
                if 64 < len(data_label):
                    data_label = data_label[:64]                
                wav_data_lst.append(data_input)
                label_data_lst.append(data_label)
                length = data_input.shape[0] // 8 + data_input.shape[0] % 8
                if 200 < length:
                    length = 200
                if length < 65:
                    length = 65                    
                input_length.append(length)
                label_length.append([len(data_label)])
                
            X = wav_padding(batch_size,wav_data_lst)
            y = label_padding(batch_size,label_data_lst)
            label_length = np.matrix(label_length)
            input_length = np.array([input_length]).T        
            yield [X, y, input_length, label_length ], labels

def data_generator(dataset,batch_size, shuffle_list, dic_wavlist, dic_symbollist, numlist, vocab):
    labels = np.zeros((batch_size,1), dtype = np.float)
    while True:
        for i in range(len(dic_wavlist)//batch_size):
            wav_data_lst = []
            label_data_lst = []
            input_length = []
            label_length = []
            begin = i * batch_size
            end = begin + batch_size
            sub_list = shuffle_list[begin:end]
            for index in sub_list:
                wavsignal,fs = read_wav_data(dataset + '/' + dic_wavlist[numlist[index]])
                data_input = GetFrequencyFeature(wavsignal,fs)
                data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
                if 1600 < len(data_input):
                    data_input = data_input[:1600,:,:]
                data_label = word2id(dic_symbollist[numlist[index]], vocab)
                if 64 < len(data_label):
                    data_label = data_label[:64]                
                wav_data_lst.append(data_input)
                label_data_lst.append(data_label)
                length = data_input.shape[0] // 8 + data_input.shape[0] % 8
                if 200 < length:
                    length = 200
                if length < 65:
                    length = 65                    
                input_length.append(length)
                label_length.append([len(data_label)])
                
            X = wav_padding(batch_size,wav_data_lst)
            y = label_padding(batch_size,label_data_lst)
            label_length = np.matrix(label_length)
            input_length = np.array([input_length]).T        
            yield [X, y, input_length, label_length ], labels

def dev_datanew(path):
    data_input = compute_log_mel_fbank('data' + '/' + path)
    data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
    return data_input

def dev_data(path):
    wavsignal,fs = read_wav_data('data' + '/' + path)
    data_input = GetFrequencyFeature(wavsignal,fs)
    data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
    return data_input

def GetEditDistance(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            leven_cost += (j2-j1)
        elif tag == 'delete':
            leven_cost += (i2-i1)
    return leven_cost

def compute_log_mel_fbank(wav_file):
    """
    计算音频文件的fbank特征
    :param wav_file: 音频文件
    :return:
    """
    # 1.数据读取
    sample_rate, signal = wav.read(wav_file)
    # print('sample rate:', sample_rate, ', frame length:', len(signal))
    signal = signal[:128120]

    # 2.预增强
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # 3.分帧
    frame_size, frame_stride = 0.025, 0.01
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1

    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    indices = np.arange(0, frame_length).reshape(1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1,
                                                                                                                    1)
    frames = pad_signal[indices]

    # 4.加窗
    hamming = np.hamming(frame_length)
    frames *= hamming

    # 5.N点快速傅里叶变换（N-FFT）
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # 获取能量谱

    # 6.提取mel Fbank
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)

    n_filter = 80  # mel滤波器组的个数, 影响每一帧输出维度，通常取40或80个
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filter + 2)  # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    fbank = np.zeros((n_filter, int(NFFT / 2 + 1)))  # 各个mel滤波器在能量谱对应点的取值
    bin = (hz_points / (sample_rate / 2)) * (NFFT / 2)  # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
    for i in range(1, n_filter + 1):
        left = int(bin[i - 1])
        center = int(bin[i])
        right = int(bin[i + 1])
        for j in range(left, center):
            fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
        for j in range(center, right):
            fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])

    # 7.提取log mel Fbank
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB

    return filter_banks
    