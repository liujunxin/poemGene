#coding=utf-8
class Tonal:
    def __init__(self, tonalfile = '../data/pingshui.txt'):
        # 0 represents either, 1 represents Ping, -1 represents Ze
        self.FIVE_PINGZE = [[[0, -1, 1, 1, -1], [1, 1, 0, -1, 1], [0, 1, 1, -1, -1], [0, -1, -1, 1, 1]],
                [[0, -1, -1, 1, 1], [1, 1, 0, -1, 1], [0, 1, 1, -1, -1], [0, -1, -1, 1, 1]],
                [[0, 1, 1, -1, -1], [0, -1, -1, 1, 1], [0, -1, 1, 1, -1], [1, 1, 0, -1, 1]],
                [[1, 1, 0, -1, 1], [0, -1, -1, 1, 1], [0, -1, 1, 1, -1], [1, 1, 0, -1, 1]]]

        self.SEVEN_PINGZE = [[[0, 1, 0, -1, -1, 1, 1], [0, -1, 1, 1, 0, -1, 1], [0, -1, 0, 1, 1, -1, -1], [0, 1, 0, -1, -1, 1, 1]],
                [[0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, 0, -1, 1], [0, -1, 0, 1, 1, -1, -1], [0, 1, 0, -1, -1, 1, 1]],
                [[0, -1, 1, 1, 0, -1, 1], [0, 1, 0, -1, -1, 1, 1], [0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, 0, -1, 1]],
                [[0, -1, 0, 1, 1, -1, -1], [0, 1, 0, -1, -1, 1, 1], [0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, 0, -1, 1]]]

        self.tonalDict = {}
        with open(tonalfile, 'r') as f:
            f.readline()
            pingze = 1
            while True:
                line = f.readline().decode('utf-8').strip()
                if line == '':
                    break
                if line[0] == '/':
                    pingze = -1
                    continue
                for ch in line:
                    self.tonalDict[ch.encode('utf-8')] = pingze

    def check_pingze(self, ch, pingze = 0):
        if type(ch) != str:
            ch = ch.encode('utf-8')
        if not pingze:
            return True
        if ch in self.tonalDict:
            if pingze == self.tonalDict[ch]:
                return True
            else:
                return False
        return False

    def check_pz_onech(self, ch, pzid, lineid, chid, maxlen=7):
        if type(ch) != str:
            ch = ch.encode('utf-8')
        if maxlen == 5:
            try:
                pz = self.FIVE_PINGZE[pzid][lineid][chid]
            except IndexError:
                return False
        elif maxlen == 7:
            try:
                pz = self.SEVEN_PINGZE[pzid][lineid][chid]
            except IndexError:
                return False
        else:
            return False
        return self.check_pingze(ch, pz)

    def check_pz_oneline(self, pzid, lineid, linestr):
        if type(linestr) == str:
            linestr = linestr.decode('utf-8')
        linelen = len(linestr)
        for i in range(linelen):
            if not self.check_pz_onech(linestr[i].encode('utf-8'), pzid, lineid, i, linelen):
                return False

        return True

    def find_pz_firstline(self, firstline):
        if type(firstline) == str:
            firstline = firstline.decode('utf-8')

        linelen = len(firstline)
        PZmodel = []
        if linelen == 5:
            PZmodel = [i for i in range(len(self.FIVE_PINGZE))]
        elif linelen == 7:
            PZmodel = [i for i in range(len(self.SEVEN_PINGZE))]
        else:
            return None
        for i in PZmodel:
            flag = True
            for j in range(linelen):
                flag = self.check_pz_onech(firstline[j].encode('utf-8'), i, 0, j, linelen)
                if not flag:
                    break
            if flag:
                return i
        return None

def main():
    tt = Tonal('../data/pingshui.txt')
    poem = []
    ss = '两个黄鹂鸣翠柳'
    poem.append(ss)
    ss = '一行白鹭上青天'
    poem.append(ss)
    ss = '窗含西岭千秋雪'
    poem.append(ss)
    ss = '门泊东吴万里船'
    poem.append(ss)
    pzid = tt.find_pz_firstline(poem[0])
    if pzid == None:
        print('no tonal')
        return
    print('pzmodel:\t%d' % pzid)
    for i in range(1, 4):
        flag = tt.check_pz_oneline(pzid, i, poem[i])
        if not flag:
            print(('第%d句不押韵' % (i+1)).decode('utf-8'))
        else:
            print(flag)

if __name__ == '__main__':
    main()
