import numpy as np
import time

text_big = 'quero uma frase de teste que funcione para o nosso exemplo'
text_small = 'Não quero uma frase para o nosso exemplo'

text_big = text_big.split()
text_small = text_small.split()

def nw_forwords(text_small, text_big, match=1, mismatch=-1, w_insert=-1, w_delete=-1):

    len_texts = [len(text_small), len(text_big)]

    nw_matrix = np.zeros(shape=(len_texts[0], len_texts[1]))

    nw_matrix[0,:] = nw_matrix[0,:] + [i * w_insert for i in range(0, len_texts[1])]
    nw_matrix[:,0] = nw_matrix[:, 0] + [i * w_delete for i in range(0, len_texts[0])]

    for i in range(1, len_texts[0]):
        for j in range(1, len_texts[1]):

            if text_small[i-1] == text_big[j-1]:
                S = match
            else:
                S = mismatch

            diag = nw_matrix[i-1,j-1]
            up = nw_matrix[i-1, j]
            left = nw_matrix[i, j-1]

            aux_scalar = np.max((diag + S, up + w_delete, left + w_insert))

            nw_matrix[i,j] = aux_scalar

    return nw_matrix

nw_forwords(text_small, text_big)

def best_align(text_1, text_2, match=1, mismatch=-1, w_insert=-1, w_delete=-1):

    duration = time.time()

    if len(text_1.split()) > len(text_2.split()):
        text_small = text_2.split()
        text_big = text_1.split()
    elif len(text_1.split()) == len(text_2.split()):
        text_small = text_2.split()
        text_big = text_1.split()
        w_delete = w_insert
    else:
        text_small = text_1.split()
        text_big = text_2.split()


    nw_matrix = nw_forwords(text_small, text_big, match=match, mismatch=mismatch, w_insert=w_insert, w_delete=w_delete)

    i, j = [len(text_small)-1, len(text_big)-1]

    alignment_big = [text_big[j]]
    alignment_small = [text_small[i]]

    caminho = []

    while i > 0 or j > 0:

        if nw_matrix[i-1,j-1] >= nw_matrix[i-1,j] and nw_matrix[i-1,j-1] >= nw_matrix[i,j-1] and i > 0 and j > 0:
            i = i - 1
            j = j - 1

            if len(text_big[j]) > len(text_small[i]):
                text_small[i] = text_small[i] + ' '*(len(text_big[j])-len(text_small[i]))

            elif len(text_big[j]) < len(text_small[i]):
                text_big[j] = text_big[j] + ' '*(len(text_small[i])-len(text_big[j]))

            else:
                pass

            alignment_big.append(text_big[j])
            alignment_small.append(text_small[i])

            caminho.append("diag")

        elif nw_matrix[i-1,j] >= nw_matrix[i-1,j-1] and nw_matrix[i-1,j] >= nw_matrix[i,j-1] and i > 0:
            i = i - 1
            alignment_big.append('#'*len(text_small[i]))
            alignment_small.append(text_small[i])

            caminho.append("up")

        else:
            j = j - 1
            alignment_big.append(text_big[j])
            alignment_small.append('#'*len(text_big[j]))

            caminho.append("left")

    alignment_small.reverse()
    alignment_big.reverse()
    caminho.reverse()

    alignment_small = ' '.join(alignment_small)
    alignment_big = ' '.join(alignment_big)

    duration = time.time() - duration

    return alignment_small, alignment_big, nw_matrix, caminho, duration


#text_1 = 'quero uma frase de teste que funcione para o nosso exemplo'
#text_2 = 'Não quero uma frase para o nosso exemplo'

#sample_1, sample_2, nw_matrix, caminho, duration = best_align(text_1,text_2,1,-1,-1,-1)
#print(sample_1)
#print(sample_2)
#print(nw_matrix)
#print(caminho)
#print(duration)