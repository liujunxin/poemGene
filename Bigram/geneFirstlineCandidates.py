#coding=utf-8
from itertools import permutations
from Tonal import Tonal
import numpy
import sys

def readsxhy(sxhyfile):
    classes = []
    keywords = []
    with open(sxhyfile, 'r') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            if line == '':
                continue
            elif line[0:2] == '<b':
                classes.append([line.split('\t')[2].encode('utf-8')])
                keywords.append([])
            elif line[0:2] == '<e':
                continue
            else:
                temp = line.split('\t')
                if len(temp) < 3:
                    continue
                classes[-1].append(temp[1].encode('utf-8'))
                keywords[-1].append([temp[1].encode('utf-8')])
                words = temp[2].encode('utf-8').split()
                for word in words:
                    keywords[-1][-1].append(word)
    return classes, keywords
class geneFirstlineCandidates:
    def __init__(self, sxhyfile):
        self.classes, self.keywords = readsxhy(sxhyfile)
    def gene_first_candidates(self, bigclassid, smallclassid, maxlen=7):
        candidates1 = []
        candidates2 = []
        keywords = self.keywords[bigclassid][smallclassid]
        for i in range(1, 4):
            for temp in permutations(keywords, i):
                line = ''.join(temp)
                if len(line.decode('utf-8')) == maxlen:
                    candidates1.append(line)
        for candidate in candidates1:
            chars = [ch for ch in candidate.decode('utf-8')]
            charsset = set(chars)
            if len(chars) != len(charsset):
                continue
            candidates2.append(candidate)
        if len(candidates2)>0:
            return candidates2
        else:
            return candidates1
    def genefirstline(self):
        while True:
            print('请输入欲生成的诗句的长度(仅限5言或7言诗)'.decode('utf-8'))
            maxlen = input()
            if maxlen == 5 or maxlen == 7:
                break
            else:
                print('输入错误！本系统只限于生成5言或7言诗'.decode('utf-8'))
        while True:
            print('请选择欲生成的诗句的大类'.decode('utf-8'))
            print('0:\t自行输入第一句诗'.decode('utf-8'))
            for i in range(len(self.classes)):
                print('%d:\t%s' % (i + 1, self.classes[i][0].decode('utf-8')))
            bigclassid = input()
            if bigclassid == 0:
                while True:
                    print('请输入第一句诗句'.decode('utf-8'))
                    firstline = raw_input()
                    firstline = firstline.decode(sys.stdin.encoding)
                    firstline = firstline.encode('utf-8')
                    if len(firstline.decode('utf-8')) != maxlen:
                        print('输入的诗句长度错误！请重新输入！'.decode('utf-8'))
                        continue
                    print('请稍候'.decode('utf-8'))
                    return [firstline]
            if 1<=bigclassid and bigclassid<=len(self.classes):
                while True:
                    print('请选择欲生成的诗句的小类'.decode('utf-8'))
                    for j in range(1, len(self.classes[bigclassid-1])):
                        print('%d:\t%s' % (j, self.classes[bigclassid-1][j].decode('utf-8')))
                    smallclassid = input()
                    if 1<=smallclassid and smallclassid<len(self.classes[bigclassid-1]):
                        break
                    else:
                        print('输入错误！请重新选择小类！'.decode('utf-8'))
                        continue
            else:
                print('输入错误！请重新选择大类！'.decode('utf-8'))
                continue
            break
        outinfo = '您选择的诗歌类别为:\t%s-%s' % (self.classes[bigclassid-1][0], self.classes[bigclassid-1][smallclassid])
        print(outinfo.decode('utf-8'))
        print('请稍候'.decode('utf-8'))
        candidates = self.gene_first_candidates(bigclassid-1, smallclassid-1, maxlen)
        #print(len(candidates))
        return candidates


if __name__ == '__main__':
    gene = geneFirstlineCandidates('../data/shixuehanying.txt')
    candidates = gene.genefirstline()
