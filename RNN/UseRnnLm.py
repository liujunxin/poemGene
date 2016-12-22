#coding=utf-8
from keras.models import load_model
import pickle as pkl
import numpy
class UseRnnLm:
    def __init__(self, model = '../path/to/rnnlm/model', dictionary = '../data/traindataRnnlm.pickle'):
        self.model = load_model(model)
        self.charDict = {}
        with open(dictionary, 'rb') as f:
            self.charDict = pkl.load(f)
    def getscore(self, poem = ''):
        if type(poem) == str:
            poem = poem.decode('utf-8')
        x = [0] + [self.charDict[ch.encode('utf-8')] for ch in poem[:-1]]
        x = numpy.asarray([x])
        y = [self.charDict[ch.encode('utf-8')] for ch in poem]
        probs = self.model.predict_proba(x)
        score = 1
        for idx, prob in enumerate(probs[0]):
            p = prob[y[idx]]
            score *= p
        return score
    def getscores(self, poems=[]):
        poemstemp = []
        for poem in poems:
            if type(poem) == str:
                poemstemp.append(poem.decode('utf-8'))
            else:
                poemstemp.append(poem)
        x = [[0] + [self.charDict[ch.encode('utf-8')] if ch.encode('utf-8') in self.charDict else 1 for ch in poem[:-1]] for poem in poemstemp]
        x = numpy.asarray(x)
        y = [[self.charDict[ch.encode('utf-8')] if ch.encode('utf-8') in self.charDict else 1 for ch in poem] for poem in poemstemp]
        scores = [1] * len(poemstemp)
        probs = self.model.predict_proba(x)
        #print(probs.shape)
        for ii, prob in enumerate(probs):
            for jj, pp in enumerate(prob):
                p = pp[y[ii][jj]]
                scores[ii] *= p
        return scores

if __name__ == '__main__':
    rnnlm = UseRnnLm('../path/to/rnnlm/model', '../data/traindataRnnlm.pickle')
    score = rnnlm.getscore('两个黄鹂鸣翠柳')
    print(score)
    poems = []
    poems.append('两个黄鹂鸣翠柳')
    scores = rnnlm.getscores(poems)
    print(scores)
    poems.append('一行白鹭上青天')
    poems.append('窗含西岭千秋雪')
    poems.append('门泊东吴万里船')
    scores = rnnlm.getscores(poems)
    print(scores)
