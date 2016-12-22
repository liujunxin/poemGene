#coding=utf-8
import re
import sys
def main(corpus_file, trainfile, vaildfile):
    poemslist = []
    with open(corpus_file, 'r') as f:
        flag = False
        f.readline()
        f.readline()
        f.readline()
        while True:
            poem = ""
            while True:
                temp = f.readline()
                if temp == "":
                    flag = True
                    break
                if temp.rfind('<http:') != -1:
                    temp = f.readline()
                    temp = f.readline()
                    if temp.rfind('<http:') != -1:
                        temp = f.readline()
                        temp = f.readline()
                    break
                temp = temp.replace('\n', '')
                poem += temp
            poemslist.append(poem)
            if len(poemslist) % 1000 == 0:
                print("complete:\t%d\tsamples" % len(poemslist))
            if flag:
                break
    f = open(trainfile, 'w')
    num = 0
    for poem in poemslist:
        sentences = re.split('。|？|！|；|（|）', poem)
        poemtemp = []
        lens = 0
        for s in sentences:
            if s.rfind('<img') != -1:
                continue
            if s.rfind('□') != -1:
                continue
            s = s.strip().replace(':', '')
            s = s.replace('：', '')
            s = s.replace('、', '')
            s = s.replace('“', '')
            s = s.replace('”', '')
            s = s.replace('<', '')
            s = s.replace('>', '')
            s = s.replace('《', '')
            s = s.replace('》', '')
            s = s.replace('　', '')
            temp = s.split('，')
            while len(temp) > 1:
                if lens == 0:
                    if len(temp[0].decode('utf-8')) == 5 or len(temp[0].decode('utf-8')) == 7:
                        lens = len(temp[0].decode('utf-8'))
                temp1 = temp.pop(0)
                temp2 = temp.pop(0)
                if lens:
                    if len(temp1.decode('utf-8')) == lens and len(temp2.decode('utf-8')) == lens:
                        poemtemp.append(temp1)
                        poemtemp.append(temp2)
        for i in range(0, len(poemtemp), 4):
            if i + 3 < len(poemtemp):
                f.write(''.join([word.encode('utf-8') + ' ' for word in poemtemp[i].decode('utf-8')]).strip() + ' ')
                f.write(''.join([word.encode('utf-8') + ' ' for word in poemtemp[i+1].decode('utf-8')]).strip() + ' ')
                f.write(''.join([word.encode('utf-8') + ' ' for word in poemtemp[i+2].decode('utf-8')]).strip() + ' ')
                f.write(''.join([word.encode('utf-8') + ' ' for word in poemtemp[i+3].decode('utf-8')]).strip() + '\n')
        num = num + 1
        if num % 1000 == 0:
            print("complete:\t%d\tsamples" % num)
    f.close()
    f1 = open(trainfile, 'r')
    with open(vailidfile, 'w') as f2:
        for i in range(2000):
            poem = f1.readline()
            f2.write(poem)
    f1.close()


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf-8')
    corpus_file = u'../data/唐诗语料库.txt'
    trainfile = '../data/traindataRnn'
    vailidfile = '../data/vailddataRnn'
    main(corpus_file, trainfile, vailidfile)
