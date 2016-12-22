#coding=utf-8
import numpy
import os

import numpy
import os

from train import train

def main(job_id, params):
    print(params)
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=1000,
                     batch_size=100,
                     valid_batch_size=100,
                     validFreq=2000,
                     dispFreq=10,
                     saveFreq=2000,
                     sampleFreq=1000,
                     datasets='../data/traindataRnn',#path to train data
                     valid_datasets='../data/validdataRnn',#path to valid data
                     dictionaries='../data/traindataRnn.pickle',#path to dictionary
                     numofs = 1,#model1 use 1
                     use_dropout=params['use-dropout'][0],
                     overwrite=False)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['/path/to/save/model1'],
        'dim_word': [500],
        'dim': [1024],
        'n-words': [7167],#dict length
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})
