"""

This project brings some general functions to support the project

"""

import re
import time
import unidecode
import numpy as np
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
        threshold = np.floor((len_word_1 + len_word_2) * 0.2)  # Malleable threshold for larger words

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

                dic.update({i: {'start_time': 0, 'end_time': 0, 'word': phrase[i]}})
                j = j+1  # Counting the gaps

            else:

                # Because of j we will able to search the list of words that don't have a gap
                start_time = re.search('start_time: (\d+\.\d+)', list_word_time[i-j]).group(1)
                end_time = re.search('end_time: (\d+\.\d+)', list_word_time[i-j]).group(1)
                dic.update({i: {'start_time': float(start_time), 'end_time': float(end_time), 'word': phrase[i]}})

    return dic


def subtitle_gen(subtitle_name, human_transcription_align, machine_transcription_align, dic_pos_time):
    """
    This function create a subtitle using the human transcription with the time of the machine transcription.
    Sometimes the transcription of the machine accumulates time in one or more words,
    when it can't recognize some word or phrase spoken. E.g.: short word with two seconds of duration.
    We use the metric of characters per second as a measure of quality, if the sentence is below a certain
    value and above another value, are indications that it is not good.
    Size of which phrase at the subtitle is variable, starting at 10 words.
    Keep adding words as long as the number of characters per second is not good.
    If it gets too large, divide the phrase into N pieces and each piece gets a fraction of the total duration.

    :param subtitle_name: (String, default none) The name of the subtitle
    :param human_transcription: (List, default none) The aligned human transcription in list format,
    each word a position of the list
    :param machine_transcription: (List, default none) The aligned human transcription in list format,
    each word a position of the list
    :param dic_position_time: (Dic, default none) The dictionary with the time each position has.
    :return:
    A subtitle in srt format
    """

    with open(subtitle_name, "w") as file:

        i = 0
        j = 0

        while i <= len(human_transcription_align):
            file.write(str(i + 1) + '\n')  # Each sentence begins with its position in the subtitle

            # The majority of interview have a bip in the first minute, so I separated the first word
            if i == 0:
                start_time_format = "00:00:00,000"

                end_time = dic_pos_time[i]['end_time']
                end_time_format = time.strftime('%H:%M:%S,', time.gmtime(end_time)) + \
                                  re.findall('\.\d', str(end_time))[0][1] + '00'

                file.write(start_time_format + ' --> ' + end_time_format + '\n')
                file.write(human_transcription_align[i] + '\n\n')

                i = i+1

            # Last phrase
            elif i + j >= len(human_transcription_align):
                start_time = dic_pos_time[i - 1]['end_time']
                start_time_format = time.strftime('%H:%M:%S,', time.gmtime(start_time)) + \
                                    re.findall('\.\d', str(start_time))[0][1] + '00'

                end_time = dic_pos_time[len(human_transcription_align) - 1]['end_time']
                end_time_format = time.strftime('%H:%M:%S,', time.gmtime(end_time)) + \
                                  re.findall('\.\d', str(end_time))[0][1] + '00'

                file.write(start_time_format + ' --> ' + end_time_format + '\n')
                file.write(' '.join(human_transcription_align[i:len(human_transcription_align)]) + '\n\n')

                break

            # All other phrases
            else:
                start_time = dic_pos_time[i - 1]['end_time']
                start_time_format = time.strftime('%H:%M:%S,', time.gmtime(start_time)) + \
                                    re.findall('\.\d', str(start_time))[0][1] + '00'

                temp_char_per_second = 0
                aux_pos_best_char_per_second = False  # flag

                # Searching for the best combination of words with the lowest second character
                # although greater than 5.
                for j in range(10, 200):
                    if '#' not in machine_transcription_align[i + j]:
                        end_time = dic_pos_time[i + j]['end_time']
                        duration = end_time - start_time

                        if duration == 0:
                            continue
                        else:
                            length_char = len(' '.join(human_transcription_align[i:i + j + 1]))

                            if 5 <= length_char / duration <= 21:
                                position_best = j
                                aux_pos_best_char_per_second = True
                                break

                            elif temp_char_per_second < length_char / duration:
                                temp_char_per_second = length_char / duration
                                position_second_best = j

                if aux_pos_best_char_per_second is True:
                    j = position_best
                else:
                    j = position_second_best

                end_time = dic_pos_time[i + j]['end_time']
                end_time_format = time.strftime('%H:%M:%S,', time.gmtime(end_time)) + \
                                  re.findall('\.\d', str(end_time))[0][1] + \
                                  '00'

                temp_phrase = ' '.join(human_transcription_align[i:i + j + 1])  # Saves the actual phrase
                number_of_word = len(temp_phrase.split())

                # Tests if the phrase is too big.
                if number_of_word > 20:
                    total_time = end_time - start_time
                    time_per_word = total_time / number_of_word

                    number_partition = int(np.ceil(number_of_word / 20))  # Maximum of 30 words per slice
                    size_partition = int(np.ceil(number_of_word / number_partition))

                    # Create the new partitions, the number is defined by how many times it exceeds 30, rounded up
                    for part in range(0, number_partition):

                        if part == number_partition - 1:

                            new_start_time_format = new_end_time_format
                            new_phrase = ' '.join(temp_phrase.split()[part * size_partition:number_of_word])

                            file.write(new_start_time_format + ' --> ' + end_time_format + '\n')
                            file.write(new_phrase + '\n\n')

                        elif part == 0:

                            new_end_time = np.round(start_time + size_partition * time_per_word, 1)
                            new_end_time_format = time.strftime('%H:%M:%S,', time.gmtime(new_end_time)) + \
                                                  re.findall('\.\d', str(new_end_time))[0][1] + '00'

                            new_phrase = ' '.join(temp_phrase.split()[0:size_partition])

                            file.write(start_time_format + ' --> ' + new_end_time_format + '\n')
                            file.write(new_phrase + '\n\n')

                        else:
                            # Saving the ending time of the previous partition
                            new_start_time_format = new_end_time_format

                            # Calculate the new ending time adding seconds per words within the partition
                            new_end_time = np.round(new_end_time + size_partition * time_per_word, 1)
                            new_end_time_format = time.strftime('%H:%M:%S,', time.gmtime(new_end_time)) + \
                                                  re.findall('\.\d', str(new_end_time))[0][1] + \
                                                  '00'

                            new_phrase = ' '.join(temp_phrase.split()
                                                  [(part * size_partition):(part * size_partition + size_partition)])

                            file.write(new_start_time_format + ' --> ' + new_end_time_format + '\n')
                            file.write(new_phrase + '\n\n')

                else:
                    file.write(start_time_format + ' --> ' + end_time_format + '\n')
                    file.write(temp_phrase + '\n\n')

                i = i+j+1
