#coding=utf-8
import pickle

class dataIterator:
    def __init__(self, filepath, dictpath, batch_size=128, n_words=-1):
        self.file = open(filepath, 'r')
        with open(dictpath, 'rb') as f:
            self.dict = pickle.load(f)
        self.batch_size = batch_size
        self.n_words = n_words
        self.buffer = []
        self.buff_size = batch_size * 20

    def __iter__(self):
        return self

    def reset(self):
        self.file.seek(0)

    def next(self):
        poems = []
        if len(self.buffer) == 0:
            for i in range(self.buff_size):
                temp = self.file.readline()
                if temp == '':
                    break
                self.buffer.append(temp.strip().split())

        if len(self.buffer) == 0:
            self.reset()
            raise StopIteration

        for i in range(self.batch_size):
            try:
                temp = self.buffer.pop(0)
            except IndexError:
                break
            temp = [self.dict[ch] if ch in self.dict else 1 for ch in temp]
            if self.n_words > 0:
                temp = [ch if ch < self.n_words else 1 for ch in temp]
            poems.append(temp)

        if len(poems) == 0:
            self.reset()
            raise StopIteration
        return poems
