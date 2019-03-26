import re
import numpy as np
import unidecode
from nltk import edit_distance


def parse_transcription_human(transcription):
    transcription = transcription+" \n"
    
    transcription = re.findall(r'–(.*?)\n', transcription)
    transcription = [i.strip() for i in transcription]
    transcription = ' '.join(transcription)
    transcription = re.sub(r'[^\w\s]', '', transcription)
    transcription = re.sub('  ', ' ', transcription)
    transcription = re.sub('\n', ' ', transcription)
    transcription = transcription.lower()
    transcription = unidecode.unidecode(transcription)

    return transcription


def parse_transcription_machine(transcription):
    transcription = re.findall(r'Transcript:(.*?)\nconfidence', transcription)
    transcription = [i.strip() for i in transcription]
    transcription = ' '.join(transcription)
    transcription = transcription.lower()
    transcription = unidecode.unidecode(transcription)

    return transcription


def score_match(phrase_1, phrase_2, distance = False):
    phrase_1 = phrase_1.split()
    phrase_2 = phrase_2.split()

    if distance == False:
        positive_score = [1 for i in range(0, len(phrase_1)) if phrase_1[i] == phrase_2[i]]
    else:
        positive_score = [1 for i in range(0, len(phrase_1)) \
                          if match_using_edit_distance(phrase_1[i],phrase_2[i]) == True]

    positive_score = np.sum(positive_score)

    negative_score = len(phrase_1) - positive_score

    total_score = positive_score - negative_score

    return total_score, positive_score, negative_score

def match_using_edit_distance(word_1, word_2):
    len_word_1 = len(word_1)
    len_word_2 = len(word_2)

    if len_word_1 + len_word_2 > 6:
        threshold = np.floor((len_word_1 + len_word_2) * 0.2)

        if threshold < 2:
            threshold = 2

        return edit_distance(word_1, word_2) <= threshold

    else:
        return edit_distance(word_1, word_2) == 0

def phrase_dic(phrase, list_word_time):
    phrase = phrase.split()
    dic = {}

    j = 0
    for i in range(0,len(phrase)):
        if phrase[i] == '':
            pass
            j=j-1
        else:
            if '#' in phrase[i]:
                word = len(phrase[i])+1

                start_time = dic[i-1]['end_time']
                end_time = np.round(dic[i-1]['end_time']+word*0.06,1)

                dic.update({i: {'start_time': float(start_time), 'end_time': float(end_time), 'word': phrase[i]}})
                j = j - 1
            else:
                start_time = re.search('start_time: (\d+\.\d+)', list_word_time[i+j]).group(1)
                end_time = re.search('end_time: (\d+\.\d+)', list_word_time[i+j]).group(1)
                dic.update({i: {'start_time': float(start_time), 'end_time': float(end_time), 'word': phrase[i]}})

    return dic

from bayes_opt import BayesianOptimization
