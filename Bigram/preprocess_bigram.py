#coding=utf-8
import re
import sys
def main(corpus_file, outfile):
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
    f = open(outfile, 'w')
    i = 0
    for poem in poemslist:
        sentences = re.split('。|？|！|；|（|）', poem)
        flag = False
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
                temp1 = temp.pop(0)
                temp2 = temp.pop(0)
                if temp1.strip() != "" and temp2.strip() != "":
                    if flag:
                        f.write(" ")
                    else:
                        flag = True
                    f.write(temp1 + " " + temp2)
        f.write('\n')
        i = i + 1
        if i % 1000 == 0:
            print("complete:\t%d\tsamples" % i)
    f.close()


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf-8')
    corpus_file = u'../data/唐诗语料库.txt'
    outfile = '../data/traindataBigram'
    main(corpus_file, outfile)
