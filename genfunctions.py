"""

This project brings some general functions to support the project

"""

import re
import numpy as np
import unidecode
from nltk import edit_distance


def clean_transcription_human(transcription):
    """
    Retrieve the text from a human transcriptions of the CPDOC's historical archive.
    The pattern of each sentence begin with – and ending with \n .
    Also, passes string to lowercase and remove accents.

    :param transcription: (String, default none) human transcriptions from the CPDOC's Historical archive.
    :return:
    A String.
    """
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


def clean_transcription_machine(transcription):
    """
    Retrive the text from a machine transcriptions of the CPDOC's historical archive.
    The pattern of each sentence begin with Transcript: and ending with \nconfidence .
    Also, passes string to lowercase and remove accents.

    :param transcription: (String, default none) machine transcriptions from the CPDOC's Historical archive.
    :return:
    A String.
    """
    transcription = re.findall(r'Transcript:(.*?)\nconfidence', transcription)
    transcription = [i.strip() for i in transcription]
    transcription = ' '.join(transcription)
    transcription = transcription.lower()
    transcription = unidecode.unidecode(transcription)

    return transcription


def score_match(phrase_1, phrase_2, distance=False):
    """
    Return a score between two phrases. The score starts at 0, if in the same position i the two phrases
    have the same word, the score increase 1, otherwise, decrease 1.

    :param phrase_1: (String, default none) A string containing words.
    :param phrase_2: (String, default none) A string containing words.
    :param distance: (Bool, default False) if True the edit distance will be used to consider a match between words.
    :return:
    A tuple containing 3 interger: Final Score, Number of match and Number of mismatch.
    """
    phrase_1 = phrase_1.split()
    phrase_2 = phrase_2.split()

    if distance is False:
        positive_score = [1 for i in range(0, len(phrase_1)) if phrase_1[i] == phrase_2[i]]
    else:
        positive_score = [1 for i in range(0, len(phrase_1))
                          if match_using_edit_distance(phrase_1[i], phrase_2[i]) is True]

    positive_score = np.sum(positive_score)
    negative_score = len(phrase_1) - positive_score

    total_score = positive_score - negative_score

    return total_score, positive_score, negative_score


def match_using_edit_distance(word_1, word_2):
    """
    Calculate de Levenshtein distance between two words,
    if the distance is less than a threshold is considered a match.
    The Levenshtein distance is the number of edits that are require to change one word into the other.
    It's only calculate de distance if the sum of the length of the two words is greater than 6.

    :param word_1: (String, default none) A word of any size.
    :param word_2: (String, default none) A word of any size.
    :return:
    A Boolean, True if was considered a match, False otherwise.
    """
    len_word_1 = len(word_1)
    len_word_2 = len(word_2)

    if len_word_1 + len_word_2 > 6:
        threshold = np.floor((len_word_1 + len_word_2) * 0.2)  # Threshold malleable for larger words

        if threshold < 2:
            threshold = 2

        return edit_distance(word_1, word_2) <= threshold

    else:
        return edit_distance(word_1, word_2) == 0


def phrase_dic(phrase, list_word_time):
    """
    After the Needleman-Wuncsh the size of the machine transcription phrase change.
    It is necessary to know in which positions do we have the information of the time, because
    gaps are new 'words' without the time information.
    This function saves in a dictionary the position of each word and each gap as a key.
    Each key contains, if it is not a gap, a dictionary with the time that word is said, when it finished and the word.
    If it is a gap, contains a dictionary dummy.

    :param phrase: (String, default none) The Machine transcription after the align.
    :param list_word_time: (List, default none) List sorted by order of appearance of words.
    Each position with the word, start time and end time of which word.
    E.g. ['word_!, start_time: 0, end_time: 1', 'word_2, start_time: 1, end_time: 1.5',...]
    :return:
    Dictionary with the time, if exist, of which word by position.
    """
    phrase = phrase.split()
    dic = {}

    j = 0
    for i in range(0, len(phrase)):
        if phrase[i] == '':
            pass
            j = j+1
        else:
            if '#' in phrase[i]:
                #word = len(phrase[i])+1

                #start_time = dic[i-1]['end_time']
                #end_time = np.round(dic[i-1]['end_time']+word*0.06, 1)

                dic.update({i: {'start_time': 0, 'end_time': 0, 'word': phrase[i]}})
                j = j+1  # Counting the gaps
            else:
                # Because of j we will able to search the list of words that don't have a gap
                start_time = re.search('start_time: (\d+\.\d+)', list_word_time[i-j]).group(1)
                end_time = re.search('end_time: (\d+\.\d+)', list_word_time[i-j]).group(1)
                dic.update({i: {'start_time': float(start_time), 'end_time': float(end_time), 'word': phrase[i]}})

    return dic
