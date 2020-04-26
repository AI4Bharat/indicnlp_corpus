# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np


def read(file, max_voc, threshold=0, vocabulary=None, dtype='float'):
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    wordset = set()
    if count<max_voc:
        max_voc=0
    if max_voc:
        matrix = np.empty((max_voc, dim), dtype=dtype)
    else:
        matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
    if max_voc==0:
        for i in range(count):
            word, vec = file.readline().split(' ', 1)
            if vocabulary is None:
                words.append(word)
                matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
            elif word in vocabulary:
                words.append(word)
                matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
    else:
        wc=0
        for i in range(count):
            if wc==max_voc:
                break
            word, vec = file.readline().split(' ', 1)
            
            if word.lower() in wordset:
                continue
            words.append(word.lower())
            wordset.add(word.lower())
            
            if vocabulary is None:
                matrix[wc] = np.fromstring(vec, sep=' ', dtype=dtype)
            elif word in vocabulary:
                matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
                
            wc+=1
            
    return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))

def read_vocab(file, max_voc):
    header = file.readline().split(' ')
    count = int(header[0])
    dim = int(header[1])
    words = []
    if count<max_voc:
        max_voc=0
    if max_voc==0:
        for i in range(count):
            word, vec = file.readline().split(' ', 1)
            words.append(word)
    else:
        wc=0
        for i in range(count):
            if wc==max_voc:
                break
            word, vec = file.readline().split(' ', 1)
            words.append(word)
            wc+=1
    return words

def write(words, matrix, file):
    m = np.array(matrix)
    print('%d %d' % m.shape, file=file)
    for i in range(len(words)):
        print(words[i] + ' ' + ' '.join(['%.6g' % x for x in m[i]]), file=file)


def length_normalize(matrix):
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    return matrix / norms[:, np.newaxis]


def mean_center(matrix):
    avg = np.mean(matrix, axis=0)
    return matrix - avg


def length_normalize_dimensionwise(matrix):
    norms = np.sqrt(np.sum(matrix**2, axis=0))
    norms[norms == 0] = 1
    return matrix / norms


def mean_center_embeddingwise(matrix):
    avg = np.mean(matrix, axis=1)
    return matrix - avg[:, np.newaxis]