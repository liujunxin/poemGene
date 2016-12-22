#coding=utf-8
import pickle as pkl
from data_iterator import dataIterator
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
def create_yvec(y, vacab_size):
    vec = numpy.zeros(vacab_size, dtype=numpy.float32)
    vec[y] = 1
    return numpy.asarray(vec, dtype=numpy.float32)
def prepare_traindata(seqs_x, vacab_size):
    lengths_x = [len(s) for s in seqs_x]
    x = []
    y = []
    for idx, s_x in enumerate(seqs_x):
        slen = lengths_x[idx] // 4#每句的长度（5或7）
        x.append([2] + s_x[0:slen-1])
        y.append(s_x[0:slen])
        x.append([2] + s_x[slen:2*slen-1])
        y.append(s_x[slen:2*slen])
        x.append([2] + s_x[2*slen:3*slen-1])
        y.append(s_x[2*slen:3*slen])
        x.append([2] + s_x[3*slen:4*slen-1])
        y.append(s_x[3*slen:4*slen])
    x = pad_sequences(x)
    y = pad_sequences(y)
    y_vec = []
    for sentence in y:
        vec_temp = []
        for word in sentence:
            vec_temp.append(create_yvec(word, vacab_size))
        y_vec.append(vec_temp)
    return x, y_vec
def train(dataset='../data/traindataRnn',#path to train data
          dictionary='../data/traindataRnnlm.pickle',#path to rnnlm dictionary
          batch_size=50,
          max_epochs=15,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          dim_word=100,  # word vector dimensionality
          dim=1000,
          save_path='/path/to/save/model'):

    charDict = {}
    with open(dictionary, 'rb') as f:
        charDict = pkl.load(f)
    charDict_r = {}
    for kk, vv in charDict.items():
        charDict_r[vv] = kk
    vocab_size = len(charDict)

    traindata = dataIterator(dataset, dictionary, batch_size)

    #sentencelen = 7
    model = Sequential()
    #model.add(Embedding(vocab_size, dim_word, input_length=sentencelen))
    model.add(Embedding(vocab_size, dim_word, mask_zero=True))
    model.add(LSTM(output_dim=dim, return_sequences=True, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(LSTM(output_dim=dim, return_sequences=True, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    update = 0
    for epochid in range(max_epochs):
        for x in traindata:
            x, y = prepare_traindata(x, vocab_size)
            train_loss = model.train_on_batch(x, y)
            update += 1
            if update % dispFreq == 0:
                print "Epoch:\t%d\tUPdate:\t%d\tloss:\t" % (epochid, update),
                print(train_loss)
            if update >= finish_after:
                break
        print("save model!")
        save_name = save_path + "rnnlm_epoch%d.h5" % epochid
        model.save(save_name)
        if update >= finish_after:
            break

if __name__ == '__main__':
    train(dataset='../data/traindataRnn', dictionary='../data/traindataRnnlm.pickle', save_path='/path/to/save/model')
