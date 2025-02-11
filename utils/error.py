# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 15:27
# @Author  : Zhou
# @FileName: error.py
# @Software: PyCharm

class OutofIndexError(Exception):
    def __init__(self, func):
        super(OutofIndexError, self).__init__(self)
        self.errorinfo = func + 'out of index'

    def __str__(self):
        return self.errorinfo

class IllegalConnectionError(Exception):
    def __init__(self, circuit1, circuit2):
        super(IllegalConnectionError, self).__init__(self)
        self.errorinfo = circuit1 + "and" + circuit2 + "have illegal connection"
        print('len {} is {}'.format(circuit1, len(circuit1)))
        print('len {} is {}'.format(circuit2, len(circuit2)))

    def __str__(self):
        return self.errorinfo