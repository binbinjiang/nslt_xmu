# -*- coding: utf-8 -*-
import numpy

def wer_help(groundtruth_list, hypothesis_list, debug=True):
    '''
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    :param groundtruth_list: list of ground truth
    :param hypothesis_list: list of hypsotesis
    :param debug: print debug info
    :return:
    '''
    r = groundtruth_list
    h = hypothesis_list
    # build the matrix
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8).reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)

    return d[len(r)][len(h)], len(r)
