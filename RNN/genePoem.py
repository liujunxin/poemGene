#coding=utf-8
import argparse

import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy
import pickle as pkl

from rnnlib import (build_sampler, load_params,
                    init_params, init_tparams)

from UseRnnLm import UseRnnLm
from geneFirstlineCandidates import geneFirstlineCandidates

from Tonal import Tonal

def load_model(model):
    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    f_init, f_next = build_sampler(tparams, options, trng, use_noise)
    return f_init, f_next

def gene_oneline(f_init, f_next, dictlen, x, maxlen=7):
    line = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x)
    next_state, ctx = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    for ii in range(maxlen):
        inps = [next_w, ctx, next_state]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        next_p[0,0] = 0
        next_p[0,1] = 0
        next_ptemp = [next_p[0, jj] for jj in range(dictlen)]
        sump = sum(next_ptemp)
        next_ptemp = [next_ptemp[jj] / sump for jj in range(dictlen)]

        nw = numpy.argmax(numpy.random.multinomial(1, next_ptemp))
        next_w[0] = nw
        line.append(nw)

    return line

def gene_threeline(f_init, f_next, dictlen, firstline):
    poem = [numpy.asarray(firstline)[:, None]]
    for i in range(len(f_init)):
        oldlines = numpy.concatenate(poem)
        newline = gene_oneline(f_init[i], f_next[i], dictlen, oldlines, maxlen=len(firstline))
        poem.append(numpy.asarray(newline)[:, None])
    return poem

def gene_oneline_pz(f_init, f_next, x, charDict_r, ton, pzid, lineid, maxlen=7):
    line = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x)
    next_state, ctx = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    dictlen = len(charDict_r)
    for ii in range(maxlen):
        inps = [next_w, ctx, next_state]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        next_p[0,0] = 0
        next_p[0,1] = 0
        next_ptemp = [next_p[0, 0], next_p[0, 1]]
        for jj in range(2, dictlen):
            ch = charDict_r[jj]
            flag = ton.check_pz_onech(ch, pzid, lineid, ii, maxlen)
            if flag:
                next_ptemp.append(next_p[0, jj])
            else:
                next_ptemp.append(0.)

        sump = sum(next_ptemp)
        if sump < 1e-5:
            next_ptemp = [next_p[0, jj] for jj in range(dictlen)]
            sump = sum(next_ptemp)

        next_ptemp = [next_ptemp[jj] / sump for jj in range(dictlen)]

        nw = numpy.argmax(numpy.random.multinomial(1, next_ptemp))
        next_w[0] = nw
        line.append(nw)

    return line

def gene_threeline_pz(f_init, f_next, firstline, charDict_r, pzid = None, ton = None):
    if pzid == None or ton == None:
        print('pzid is None, genePoem with no pz')
        dictlen = len(charDict_r)
        return gene_threeline(f_init, f_next, dictlen, firstline)
    else:
        poem = [numpy.asarray(firstline)[:, None]]
        for i in range(len(f_init)):
            oldlines = numpy.concatenate(poem)
            newline = gene_oneline_pz(f_init[i], f_next[i], oldlines, charDict_r, ton, pzid, i+1, maxlen=len(firstline))
            poem.append(numpy.asarray(newline)[:, None])
        return poem


def main(modelsRnn, modelRnnlm, dictionaryRnn, dictionaryRnnlm, tonalfile, sxhyfile):

    # load dictionary and invert
    charDict = {}
    with open(dictionaryRnn, 'rb') as f:
        charDict = pkl.load(f)
    charDict_r = {}
    for kk, vv in charDict.items():
        charDict_r[vv] = kk
    dictlen = len(charDict)
    print('dictlen: %d' % dictlen)

    rnnlm = UseRnnLm(modelRnnlm, dictionaryRnnlm)

    f_init = [None] * len(models)
    f_next = [None] * len(models)
    for i in range(len(models)):
        print('Building model%d' % (i + 1))
        f_init[i], f_next[i] = load_model(models[i])
        print('Done')

    ton = Tonal(tonalfile)
    geneFirst = geneFirstlineCandidates(sxhyfile)
    usetonal = False
    while True:
        print('0退出，其它继续'.decode('utf-8'))
        ss = raw_input()
        if ss == '0':
            break
        print('1考虑平仄，否则不考虑平仄'.decode('utf-8'))
        ss = raw_input()
        if ss == '1':
            usetonal = True
        else:
            usetonal = False

        flCandidates = geneFirst.genefirstline()
        flCandidatesPZ = []
        if usetonal:
            for candidate in flCandidates:
                if ton.find_pz_firstline(candidate) != None:
                    flCandidatesPZ.append(candidate)

        if usetonal and len(flCandidatesPZ) > 0:
            scores = rnnlm.getscores(flCandidatesPZ)
            index = numpy.argsort(scores)[::-1]
            firstline = flCandidatesPZ[index[0]]
        else:
            scores = rnnlm.getscores(flCandidates)
            index = numpy.argsort(scores)[::-1]
            firstline = flCandidates[index[0]]

        pzid = ton.find_pz_firstline(firstline)
        if type(firstline) != str:
            firstline = firstline.encode('utf-8')
        firstline = [charDict[ch.encode('utf-8')] if ch.encode('utf-8') in charDict else 1 for ch in firstline.decode('utf-8')]

        if usetonal and pzid != None:
            poem = gene_threeline_pz(f_init, f_next, firstline, charDict_r, pzid, ton)
        else:
            poem = gene_threeline(f_init, f_next, dictlen, firstline)

        for line in poem:
            for i in range(line.shape[0]):
                if line[i, 0] in charDict_r:
                    print charDict_r[line[i, 0]],
                else:
                    print 'UNK',
            print


if __name__ == '__main__':
    models = []
    models.append('/path/to/model1')
    models.append('/path/to/model2')
    models.append('/path/to/model3')
    modelrnnlm = '/path/to/rnnlm/model'
    dictionaryRnn = '../data/traindataRnn.pickle'#/path/to/nmt/dictionary
    dictionaryRnnlm = '../data/traindataRnnlm.pickle'#/path/to/rnnlm/dictionary
    tonalfile = '../data/pingshui.txt'
    sxhyfile = '../data/shixuehanying.txt'
    main(models, modelrnnlm, dictionaryRnn, dictionaryRnnlm, tonalfile, sxhyfile)
