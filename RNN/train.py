#coding=utf-8
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import pickle as pkl
import ipdb
import numpy
import copy

import os
import sys
import time

from data_iterator import dataIterator

from rnnlib import (build_model, build_sampler, load_params,
                    init_params, init_tparams, itemlist, adadelta, zipp, unzip)

profile = False
# batch preparation
def prepare_data(seqs_x, numofs = 1):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    max_len = max(lengths_x)
    maxsxlen = max_len // 4 * numofs
    maxsylen = max_len // 4

    n_samples = len(seqs_x)
    if n_samples < 1:
        return None, None, None, None

    x = numpy.zeros((maxsxlen, n_samples)).astype('int64')
    y = numpy.zeros((maxsylen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxsxlen, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxsylen, n_samples)).astype('float32')

    for idx, s_x in enumerate(seqs_x):
        slen = lengths_x[idx] // 4#每句的长度（5或7）
        xend = slen * numofs
        yend = xend + slen
        x[:xend, idx] = s_x[:xend]
        x_mask[:xend, idx] = 1.
        y[:slen, idx] = s_x[xend:yend]
        y_mask[:slen, idx] = 1.

    return x, x_mask, y, y_mask


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams, f_init, f_next, x, options, maxlen=7, argmax=False):
    sample = []
    #sample_score = []
    sample_score = 0

    #live_k = 1
    #dead_k = 0

    #hyp_samples = [[]] * live_k
    #hyp_scores = numpy.zeros(live_k).astype('float32')
    #hyp_states = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x)
    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    for ii in range(maxlen):
        #ctx = numpy.tile(ctx0, [live_k, 1])
        inps = [next_w, ctx0, next_state]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if argmax:
            nw = next_p[0].argmax()
        else:
            nw = next_w[0]
        sample.append(nw)
        sample_score -= numpy.log(next_p[0, nw])

    return sample, sample_score

# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True, numofs = 1):
    probs = []

    n_done = 0

    for x in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, numofs = numofs)

        pprobs = f_log_probs(x, x_mask, y, y_mask)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print('%d samples computed' % (n_done))

    return numpy.array(probs)



def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          encoder='gru',
          decoder='gru_cond',
          patience=10,  # early stopping patience
          max_epochs=100,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          n_words=10000,  # vocabulary size
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets='../data/traindataRnn',
          valid_datasets='../data/vailiddataRnn',
          dictionaries='../data/traindataRnn.pickle',
          numofs = 1,
          use_dropout=False,
          reload_=False,
          overwrite=False):

    # Model options
    model_options = locals().copy()

    # load dictionaries and invert them
    charDict = {}
    with open(dictionaries, 'rb') as f:
        charDict = pkl.load(f)
    charDict_r = {}
    for kk, vv in charDict.items():
        charDict_r[vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        print('Reloading model options')
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    print('Loading data')
    train = dataIterator(datasets, dictionaries, batch_size)
    valid = dataIterator(valid_datasets, dictionaries, valid_batch_size)

    print('Building model')
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        print('Reloading model parameters')
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask]

    print('Building sampler')
    f_init, f_next = build_sampler(tparams, model_options, trng, use_noise)

    # before any regularizer
    print('Building f_log_probs...')
    f_log_probs = theano.function(inps, cost, profile=profile)
    print('Done')

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print('Building f_cost...')
    f_cost = theano.function(inps, cost, profile=profile)
    print('Done')

    print('Computing gradient...')
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print('Done')

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print('Building optimizers...')
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print('Done')

    print('Optimization')

    best_p = None
    bad_counter = 0
    uidx = 0
    estop = False
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        rmodel = numpy.load(saveto)
        history_errs = list(rmodel['history_errs'])
        if 'uidx' in rmodel:
            uidx = rmodel['uidx']

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    for eidx in range(max_epochs):
        n_samples = 0

        for x in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, y, y_mask = prepare_data(x, numofs = numofs)

            if x is None:
                print('Minibatch with zero sample')
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, y, y_mask)

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print('NaN detected')
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print('Epoch %d Update %d Cost %f' % (eidx, uidx, cost))


            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in range(numpy.minimum(5, x.shape[1])):
                    sample, score = gen_sample(tparams, f_init, f_next,
                                               x[:, jj][:, None],
                                               model_options,
                                               maxlen=7,
                                               argmax=False)
                    print 'Source %d: ' % jj,
                    for vv in x[:, jj]:
                        if vv in charDict_r:
                            print charDict_r[vv],
                        else:
                            print 'UNK',
                    print
                    print 'Truth %d: ' % jj,
                    for vv in y[:, jj]:
                        if vv in charDict_r:
                            print charDict_r[vv],
                        else:
                            print 'UNK',
                    print
                    print 'Sample %d: ' % jj,
                    ss = sample
                    for vv in ss:
                        if vv in charDict_r:
                            print charDict_r[vv],
                        else:
                            print 'UNK',
                    print

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid, numofs = numofs)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print('Early Stop!')
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                print('Valid %f' % valid_err)

            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
            if numpy.mod(uidx, saveFreq) == 0:
                print('Saving the best model...')
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print('Done')

                # save with uidx
                if not overwrite:
                    print('Saving the model at iteration {}...'.format(uidx))
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    numpy.savez(saveto_uidx, history_errs=history_errs,
                                uidx=uidx, **unzip(tparams))
                    print('Done')

            # finish after this many updates
            if uidx >= finish_after:
                print('Finishing after %d iterations!' % uidx)
                estop = True
                break

        print('Seen %d samples' % n_samples)

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = pred_probs(f_log_probs, prepare_data,
                           model_options, valid).mean()

    print('Valid %f' % valid_err)

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                uidx=uidx,
                **params)

    return valid_err


if __name__ == '__main__':
    pass
