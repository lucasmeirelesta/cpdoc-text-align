"""

This project brings the implementation of the Needleman-Wunsch to align words between texts.

"""

import time
import numpy as np
from utils import match_using_edit_distance


def nw_words(text_small, text_big, match=1, mismatch=-1, gapsmall=-1, gapbig=-1, distance=False):
    """
    Create a matrix that is part of the Needleman-Wunsch algorithm.
    Each column is a word of the same text.
    Each row is a word of the same text.
    The text from the row and the column should be different.
    Each information of this metrics is the maximum result of the score system.

    :param text_small: (String, default none) The smallest text, each row of the matrix will represent
    a word of this text, following the order that each word appears. That is, the first line will refer
    to the first word, the second line with the second word and etc...
    :param text_big: (String, default none) Analogous to the text_small, but the largest text will
    be represented by the columns.
    :param match: (Float, default 1) if there is a match between words in the position ixj of the matrix, this
    values will be added to (i-1)x(j-1) to be test as the highest candidate. If it is, it will be the value of ixj.
    :param mismatch: (Float, default -1) if there is a mismatch between words in the position ixj of the matrix, this
    values will be added to (i-1)x(j-1) to be test as the highest candidate. If it is, it will be the value of ixj.
    :param gapsmall: (Float, default -1) This values will be added to (i-1)x(j) to be test as the highest candidate.
    If it is, it will be the value of ixj.
    :param gapbig: (Float, default -1) This values will be added to (i)x(j-1) to be test as the highest candidate.
    If it is, it will be the value of ixj.
    :param distance: (Bool, default False) if True the edit distance will be used to consider a match between words.
    :return:
    A NxM Score Matrix. N is the number of words in text_small and M is the number of words in text_big
    """

    len_texts = [len(text_small), len(text_big)]

    nw_matrix = np.zeros(shape=(len_texts[0], len_texts[1]))
    nw_matrix[0, :] = (nw_matrix[0, :] + range(0, len_texts[1])) * gapbig  # Filling the values of the first row
    nw_matrix[:, 0] = (nw_matrix[:, 0] + range(0, len_texts[0])) * gapsmall  # Filling the values of the first columns

    for i in range(1, len_texts[0]):
        for j in range(1, len_texts[1]):

            if distance is False:
                if text_small[i-1] == text_big[j-1]:
                    aux_score = match
                else:
                    aux_score = mismatch
            else:
                if match_using_edit_distance(text_small[i-1], text_big[j-1]) is True:
                    aux_score = match
                else:
                    aux_score = mismatch

            # Scores of the surrounding cells
            diag = nw_matrix[i-1, j-1]
            up = nw_matrix[i-1, j]
            left = nw_matrix[i, j-1]

            aux_scalar = np.max((diag + aux_score, up + gapbig, left + gapsmall))

            nw_matrix[i, j] = aux_scalar  # This cells receive the highest candidate score

    return nw_matrix


def best_align(text_1, text_2, match=1, mismatch=-1, gapsmall=-1, gapbig=-1, distance=False):
    """
    Calculate the best align between two texts using de Score Matrix.

    :param text_1: (String, default none) A string of any size
    :param text_2: (String, default none) A string of any size
    :param match: (Float, default 1) if there is a match between words in the position ixj of the matrix, this
    values will be added to (i-1)x(j-1) to be test as the highest candidate. If it is, it will be the value of ixj.
    :param mismatch: (Float, default -1) if there is a mismatch between words in the position ixj of the matrix, this
    values will be added to (i-1)x(j-1) to be test as the highest candidate. If it is, it will be the value of ixj.
    :param gapsmall: (Float, default -1) This values will be added to (i-1)x(j) to be test as the highest candidate.
    If it is, it will be the value of ixj.
    :param gapbig: (Float, default -1) This values will be added to (i)x(j-1) to be test as the highest candidate.
    If it is, it will be the value of ixj.
    :param distance: (Bool, default False) if True the edit distance will be used to consider a match between words.
    :param distance: True if you want to use the distance between two words to consider a match
    :return:
    Returns 5 objects, respectively:
    The smallest text with gaps, the gaps are represented by #.
    the biggest text with gaps, the gaps are represented by #.
    A NxM Score Matrix. N is the number of words in text_small and M is the number of words in text_big.
    The path chosen by the algorithm, from the lower right diagonal to the upper left diagonal.
    Duration in seconds of the process
    """

    duration = time.time()

    # Test which text is bigger
    if len(text_1.split()) > len(text_2.split()):
        text_small = text_2.split()
        text_big = text_1.split()

    elif len(text_1.split()) == len(text_2.split()):
        text_small = text_2.split()
        text_big = text_1.split()

    else:
        text_small = text_1.split()
        text_big = text_2.split()

    nw_matrix = nw_words(text_small, text_big, match=match, mismatch=mismatch,
                         gapsmall=gapsmall, gapbig=gapbig, distance=distance)

    i = len(text_small)-1
    j = len(text_big)-1

    alignment_big = [text_big[j]]
    alignment_small = [text_small[i]]  # Matching the last word in the same position

    path = []

    while i > 0 or j > 0:

        if nw_matrix[i-1, j-1] >= nw_matrix[i-1, j] and nw_matrix[i-1, j-1] >= nw_matrix[i, j-1] and i > 0 and j > 0:
            i = i-1
            j = j-1

            # adding spacing for better visual comparison between the two texts
            if len(text_big[j]) > len(text_small[i]):
                text_small[i] = text_small[i] + ' '*(len(text_big[j])-len(text_small[i]))

            elif len(text_big[j]) < len(text_small[i]):
                text_big[j] = text_big[j] + ' '*(len(text_small[i])-len(text_big[j]))

            else:
                pass

            alignment_big.append(text_big[j])
            alignment_small.append(text_small[i])

            path.append("diag")

        elif nw_matrix[i-1,j] >= nw_matrix[i-1,j-1] and nw_matrix[i-1,j] >= nw_matrix[i,j-1] and i > 0:
            i = i-1

            alignment_big.append('#'*len(text_small[i]))  # adding hash for better visual comparison
            alignment_small.append(text_small[i])

            path.append("up")

        else:
            j = j-1

            alignment_big.append(text_big[j])
            alignment_small.append('#'*len(text_big[j]))  # adding hash for better visual comparison

            path.append("left")

    alignment_small.reverse()  # Correcting the text ordering
    alignment_big.reverse()
    path.reverse()

    alignment_small = ' '.join(alignment_small)
    alignment_big = ' '.join(alignment_big)

    duration = time.time() - duration

    return alignment_small, alignment_big, nw_matrix, path, duration