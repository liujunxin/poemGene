#coding=utf-8
from PairProb import PairProb
from geneFirstlineCandidates import geneFirstlineCandidates
from Tonal import Tonal
import numpy
import jieba
import math
import itertools
import pickle
import os
class genePoemBigram:
    def __init__(self, traindata, sxhyfile, tonalfile, usetonal = False):
        self.pairPb = PairProb(traindata)
        self.geneFirst = geneFirstlineCandidates(sxhyfile)
        self.tonal = Tonal(tonalfile)
        self.usetonal = usetonal
    #add one!
    def BigramProb(self, sentence):
        words = list(jieba.cut(sentence))
        if len(words[0]) in self.pairPb.wordFreq and words[0] in self.pairPb.wordFreq[len(words[0])]:
            score = (self.pairPb.wordFreq[len(words[0])][words[0]] + 1.) / (self.pairPb.wordFreq['<allnum>'] + self.pairPb.wordVacabSize)
        else:
            score = 1. / (self.pairPb.wordFreq['<allnum>'] + self.pairPb.wordVacabSize)
        score = math.log(score)
        for i in range(len(words)-1):
            if words[i] in self.pairPb.wordPairFreq1 and words[i+1] in self.pairPb.wordPairFreq1[words[i]]:
                temp = (self.pairPb.wordPairFreq1[words[i]][words[i+1]] + 1.) / (self.pairPb.wordPairFreq1[words[i]]['<num>'] + self.pairPb.wordVacabSize)
            elif words[i] in self.pairPb.wordPairFreq1:
                temp = 1. / (self.pairPb.wordPairFreq1[words[i]]['<num>'] + self.pairPb.wordVacabSize)
            else:
                temp = 1. / self.pairPb.wordVacabSize
            score += math.log(temp)
        return score
    def BigramBest(self, candidates, lineid=0, pzid=None):
        scores = []
        for sentence in candidates:
            score = self.BigramProb(sentence)
            scores.append(score)
        idx = numpy.argsort(scores)[::-1]

        if self.usetonal:
            for index in idx:
                if lineid == 0:
                    if self.tonal.find_pz_firstline(candidates[index]) != None:
                        return candidates[index]
                elif pzid != None:
                    if self.tonal.check_pz_oneline(pzid, lineid, candidates[index]):
                        return candidates[index]

        best = candidates[idx[0]]
        return best
    def geneCandidates(self, wordslist):
        candidate = wordslist[0]
        for i in range(1, len(wordslist)):
            candidate = itertools.product(candidate, wordslist[i])
            candidate = list(candidate)
            candidate = [''.join(j) for j in candidate]
        return candidate
    def geneNextline(self, firstline, lineid=1, pzid=None):
        bestwordnum = 10
        bestcharnum = 5
        templist = []
        if type(firstline) == str:
            firstline = firstline.decode('utf-8')
        words = list(jieba.cut(firstline))
        chset = set([ch for ch in firstline])
        wordset = set()
        for word in words:
            if word in self.pairPb.wordPairFreq2:
                pairslist = sorted(self.pairPb.wordPairFreq2[word].items(), key=lambda x:(x[1]), reverse = True)
                temp = []
                for i in range(len(pairslist)):
                    if pairslist[i][0] == '<num>':
                        continue
                    if word == pairslist[i][0] or word in wordset:
                        continue
                    temp.append(pairslist[i][0])
                    wordset.add(pairslist[i][0])
                    chset = chset | set([ch for ch in pairslist[i][0]])
                    if len(temp) >= bestwordnum:
                        break
                #temp = [pairslist[i][0] for i in range(1, min(bestwordnum + 1, len(pairslist)))]
                templist.append(temp)
            else:
                charnum = len(word)
                flag = True
                for i in range(charnum):
                    if not word[i] in self.pairPb.charPairFreq2:
                        flag = False
                        break
                if flag:
                    for i in range(charnum):
                        pairslist = sorted(self.pairPb.charPairFreq2[word[i]].items(), key=lambda x:(x[1]), reverse = True)
                        temp = []
                        for j in range(len(pairslist)):
                            if pairslist[j][0] == '<num>':
                                continue
                            if pairslist[j][0] in chset:
                                continue
                            else:
                                temp.append(pairslist[j][0])
                                chset.add(pairslist[j][0])
                                if len(temp) >= bestcharnum:
                                    break
                        #temp = [pairslist[j][0] for j in range(1, min(bestcharnum + 1, len(pairslist)))]
                        templist.append(temp)
                else:
                    pairslist = sorted(self.pairPb.wordFreq[charnum].items(), key=lambda x:(x[1]), reverse = True)
                    temp = []
                    for i in range(len(pairslist)):
                        flag1 = True
                        for ch in pairslist[i][0]:
                            if ch in chset:
                                flag1 = False
                                break
                        if flag1:
                            temp.append(pairslist[i][0])
                            chset = chset | set([ch for ch in pairslist[i][0]])
                        if len(temp) >= bestwordnum:
                            break
                    #temp = [pairslist[i][0] for i in range(0, min(bestwordnum, len(pairslist)))]
                    templist.append(temp)
        candidates = self.geneCandidates(templist)
        return self.BigramBest(candidates, lineid, pzid)

    def geneline3(self, firstline, lineid=2, pzid=None):
        bestwordnum = 10
        bestcharnum = 5
        templist = []
        if type(firstline) == str:
            firstline = firstline.decode('utf-8')
        words = list(jieba.cut(firstline))
        chset = set([ch for ch in firstline])
        wordset = set()
        for word in words:
            if word in self.pairPb.wordPairFreq3:
                pairslist = sorted(self.pairPb.wordPairFreq3[word].items(), key=lambda x:(x[1]), reverse = True)
                temp = []
                for i in range(len(pairslist)):
                    if pairslist[i][0] == '<num>':
                        continue
                    if word == pairslist[i][0] or word in wordset:
                        continue
                    temp.append(pairslist[i][0])
                    wordset.add(pairslist[i][0])
                    chset = chset | set([ch for ch in pairslist[i][0]])
                    if len(temp) >= bestwordnum:
                        break
                #temp = [pairslist[i][0] for i in range(1, min(bestwordnum + 1, len(pairslist)))]
                templist.append(temp)
            else:
                charnum = len(word)
                flag = True
                for i in range(charnum):
                    if not word[i] in self.pairPb.charPairFreq3:
                        flag = False
                        break
                if flag:
                    for i in range(charnum):
                        pairslist = sorted(self.pairPb.charPairFreq3[word[i]].items(), key=lambda x:(x[1]), reverse = True)
                        temp = []
                        for j in range(len(pairslist)):
                            if pairslist[j][0] == '<num>':
                                continue
                            if pairslist[j][0] in chset:
                                continue
                            else:
                                temp.append(pairslist[j][0])
                                chset.add(pairslist[j][0])
                                if len(temp) >= bestcharnum:
                                    break
                        #temp = [pairslist[j][0] for j in range(1, min(bestcharnum + 1, len(pairslist)))]
                        templist.append(temp)
                else:
                    pairslist = sorted(self.pairPb.wordFreq[charnum].items(), key=lambda x:(x[1]), reverse = True)
                    temp = []
                    for i in range(len(pairslist)):
                        flag1 = True
                        for ch in pairslist[i][0]:
                            if ch in chset:
                                flag1 = False
                                break
                        if flag1:
                            temp.append(pairslist[i][0])
                            chset = chset | set([ch for ch in pairslist[i][0]])
                        if len(temp) >= bestwordnum:
                            break
                    #temp = [pairslist[i][0] for i in range(0, min(bestwordnum, len(pairslist)))]
                    templist.append(temp)
        candidates = self.geneCandidates(templist)
        return self.BigramBest(candidates, lineid, pzid)

    def genePoems(self):
        while True:
            print('0退出，其它继续'.decode('utf-8'))
            ss = raw_input()
            if ss == '0':
                break
            print('1考虑平仄，否则不考虑平仄'.decode('utf-8'))
            ss = raw_input()
            if ss == '1':
                self.usetonal = True
            else:
                self.usetonal = False
            flCandidates = self.geneFirst.genefirstline()
            firstline = self.BigramBest(flCandidates)
            if self.usetonal:
                pzid = self.tonal.find_pz_firstline(firstline)
                secondline = self.geneNextline(firstline, 1, pzid)
                thirdline = self.geneline3(secondline, 2, pzid)
                lastline = self.geneNextline(thirdline, 3, pzid)
            else:
                secondline = self.geneNextline(firstline)
                thirdline = self.geneline3(secondline)
                lastline = self.geneNextline(thirdline)
            print(firstline.decode('utf-8'))
            print(secondline)
            print(thirdline)
            print(lastline)


if __name__ == '__main__':
    traindata = '../data/traindataBigram'
    sxhyfile = '../data/shixuehanying.txt'
    tonalfile = '../data/pingshui.txt'

    if os.path.exists('../data/bigram'):
        print('载入数据中，请稍候'.decode('utf-8'))
        with open('../data/bigram', 'rb') as f:
            genePoem = pickle.load(f)
            genePoem.genePoems()
    else:
        genePoem = genePoemBigram(traindata, sxhyfile, tonalfile, True)
        with open('../data/bigram', 'wb') as f:
            pickle.dump(genePoem, f)
        genePoem.genePoems()


