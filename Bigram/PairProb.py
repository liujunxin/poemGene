#coding=utf-8
import jieba
def addDictHelper(dic, word1, word2):
    dic['<allnum>'] += 1
    if word1 in dic:
        if word2 in dic[word1]:
            dic[word1][word2] += 1
        else:
            dic[word1][word2] = 1
        dic[word1]['<num>'] += 1
    else:
        dic[word1] = {'<num>' : 1}
        dic[word1][word2] = 1
class PairProb:
    def __init__(self, traindata):
        #字典内保存的是u''类型
        self.wordPairFreq1, self.wordPairFreq2, self.wordPairFreq3 = self.calWordPairFreq(traindata)
        self.charPairFreq1, self.charPairFreq2, self.charPairFreq3 = self.calCharPairFreq(traindata)
        self.wordFreq = self.calWordFreq(traindata)
        self.charFreq = self.calCharFreq(traindata)
        self.wordVacabSize = 0
        for key in self.wordFreq.keys():
            if key == '<allnum>':
                continue
            self.wordVacabSize += len(self.wordFreq[key])
        self.charVacabSize = len(self.charFreq) - 1
    def calWordPairFreq(self, traindata):
        wordPairFreq1 = {'<allnum>' : 0}#同一行
        wordPairFreq2 = {'<allnum>' : 0}#1对2，3对4...
        wordPairFreq3 = {'<allnum>' : 0}#2对3
        with open(traindata, 'r') as f:
            num = 0
            for line in f:
                sentences = line.strip().split()
                sentences = [list(jieba.cut(sentence)) for sentence in sentences]
                sentenceNum = len(sentences)
                for i in range(0, sentenceNum, 2):
                    for j in range(len(sentences[i])-1):
                        addDictHelper(wordPairFreq1, sentences[i][j], sentences[i][j+1])
                    if i<sentenceNum-1:
                        for j in range(len(sentences[i+1])-1):
                            addDictHelper(wordPairFreq1, sentences[i+1][j], sentences[i+1][j+1])

                    if i<sentenceNum-1 and len(sentences[i]) == len(sentences[i+1]):
                        for j in range(len(sentences[i])):
                            if len(sentences[i][j]) == len(sentences[i+1][j]):
                                addDictHelper(wordPairFreq2, sentences[i][j], sentences[i+1][j])
                            else:
                                break

                    if i<sentenceNum-2 and len(sentences[i+1]) == len(sentences[i+2]):
                        for j in range(len(sentences[i+1])):
                            if len(sentences[i+1][j]) == len(sentences[i+2][j]):
                                addDictHelper(wordPairFreq3, sentences[i+1][j], sentences[i+2][j])
                            else:
                                break
                num += 1
                if num % 1000 == 0:
                    print('complete:\t%d\tlines' % num)
        return wordPairFreq1, wordPairFreq2, wordPairFreq3

    def calCharPairFreq(self, traindata):
        charPairFreq1 = {'<allnum>' : 0}#同一行
        charPairFreq2 = {'<allnum>' : 0}#1对2，3对4...
        charPairFreq3 = {'<allnum>' : 0}#2对3
        with open(traindata, 'r') as f:
            num = 0
            for line in f:
                sentences = line.decode('utf-8').strip().split()
                sentenceNum = len(sentences)
                for i in range(0, sentenceNum, 2):
                    for j in range(len(sentences[i])-1):
                        addDictHelper(charPairFreq1, sentences[i][j], sentences[i][j+1])
                    if i<sentenceNum-1:
                        for j in range(len(sentences[i+1])-1):
                            addDictHelper(charPairFreq1, sentences[i+1][j], sentences[i+1][j+1])

                    if i<sentenceNum-1 and len(sentences[i]) == len(sentences[i+1]):
                        for j in range(len(sentences[i])):
                            addDictHelper(charPairFreq2, sentences[i][j], sentences[i+1][j])

                    if i<sentenceNum-2 and len(sentences[i+1]) == len(sentences[i+2]):
                        for j in range(len(sentences[i+1])):
                            addDictHelper(charPairFreq3, sentences[i+1][j], sentences[i+2][j])

                num += 1
                if num % 1000 == 0:
                    print('complete:\t%d\tlines' % num)
        return charPairFreq1, charPairFreq2, charPairFreq3

    def calWordFreq(self, traindata):
        wordFreq = {'<allnum>' : 0}
        with open(traindata, 'r') as f:
            num = 0
            for line in f:
                sentences = line.strip().split()
                sentences = [list(jieba.cut(sentence)) for sentence in sentences]
                for sentence in sentences:
                    for word in sentence:
                        if len(word) in wordFreq:
                            if word in wordFreq[len(word)]:
                                wordFreq[len(word)][word] += 1
                            else:
                                wordFreq[len(word)][word] = 1
                        else:
                            wordFreq[len(word)] = {word : 1}
                        wordFreq['<allnum>'] += 1
                num += 1
                if num % 1000 == 0:
                    print('complete:\t%d\tlines' % num)
        return wordFreq

    def calCharFreq(self, traindata):
        charFreq = {'<allnum>' : 0}
        with open(traindata, 'r') as f:
            num = 0
            for line in f:
                sentences = line.decode('utf-8').strip().split()
                for sentence in sentences:
                    for ch in sentence:
                        if ch in charFreq:
                            charFreq[ch] += 1
                        else:
                            charFreq[ch] = 1
                        charFreq['<allnum>'] += 1
                num += 1
                if num % 1000 == 0:
                    print('complete:\t%d\tlines' % num)
        return charFreq





if __name__ == '__main__':
    pairPb = PairProb('../data/traindataBigram')
    print(pairPb.wordVacabSize)
    print(pairPb.charVacabSize)
    #print(pairPb.wordPairFreq1)
    print(pairPb.wordFreq[7])
    # print(pairPb.wordFreq[1])
