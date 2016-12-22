#coding=utf-8
import sys
import numpy
import pickle
def main(filepath):
    charsDict = {}
    with open(filepath, 'r') as f:
        i = 0
        for line in f:
            chars = line.strip().split()
            for ch in chars:
                if ch in charsDict:
                    charsDict[ch] += 1
                else:
                    charsDict[ch] = 1
            i += 1
            if i % 1000 == 0:
                print('complete:\t%d\tlines' % i)
    chars = list(charsDict.keys())
    freqs = list(charsDict.values())
    index = numpy.argsort(freqs)
    charslist = [chars[i] for i in index[::-1]]
    charsDict = {}
    charsDict['MASK'] = 0
    charsDict['UNK'] = 1
    charsDict['begin'] = 2
    i = 3
    print(len(charslist))
    for char in charslist:
        charsDict[char] = i
        i += 1
        if i % 1000 == 0:
            print('complete:\t%d\tchars' % i)
    with open('%slm.pickle' % filepath, 'wb') as f:
        pickle.dump(charsDict, f)
    print(len(charsDict))

if __name__ == '__main__':
    main(sys.argv[1])
