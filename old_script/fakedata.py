"""

This python file was used to generate texts with noise, so we could test our model in early stages.
Nowadays, I don't use it anymore but maybe in the future i could use it.

"""

import random
import numpy as np
from nltk.corpus import machado
import nwalgorithm

def fakedata(phrase, seed=42):
    """
    Receives a text and returns two texts. One with little noise, simulating a humanized transcript.
    The second with much noise, simulating a transcription of the machine.

    :param phrase: (String, default none) A text to be copied and divided into 2 texts with noises.
    :param seed: (Int, default 42) To fix the randomness to be able to compare experiments.
    :return:
    Two strings
    """

    phrase = phrase.split()

    phrase_fake_human = phrase
    phrase_fake_machine = phrase

    random.seed(seed)
    loss_human = np.random.uniform(0, 0.12)  # values were chosen arbitrarily
    random.seed(seed)
    loss_machine = np.random.uniform(0.1, 0.3)

    length_phrase = len(phrase)

    # Choosing a few words to be forgotten
    words_loss_human = sorted(np.random.choice(range(0, length_phrase),
                                               int(np.floor(length_phrase * loss_human))), reverse=True)
    words_loss_machine = sorted(np.random.choice(range(0, length_phrase),
                                                 int(np.floor(length_phrase * loss_machine))), reverse=True)

    phrase_fake_human = [phrase for i, phrase in enumerate(phrase_fake_human) if i not in words_loss_human]
    phrase_fake_machine = [phrase for i, phrase in enumerate(phrase_fake_machine) if i not in words_loss_machine]

    random.seed(seed)
    loss_machine = np.random.uniform(0.05, 0.1)
    random.seed(seed)
    words_change = np.random.choice(range(0, len(phrase_fake_machine)),
                                    int(np.floor(len(phrase_fake_machine) * loss_machine)),replace=False)
    random.seed(seed+1)
    move_to = np.random.choice(range(0, len(phrase_fake_machine)),
                               int(np.floor(len(phrase_fake_machine) * loss_machine)),replace=False)

    # Changing a few words of place.
    for i in range(0, len(move_to)):
        aux = phrase_fake_machine[move_to[i]]
        phrase_fake_machine[move_to[i]] = phrase_fake_machine[words_change[i]]
        phrase_fake_machine[words_change[i]] = aux

    phrase_fake_human = ' '.join(phrase_fake_human)
    phrase_fake_machine = ' '.join(phrase_fake_machine)


    return phrase_fake_human, phrase_fake_machine


raw_text = machado.raw('romance/marm05.txt')
raw_text = raw_text[998:1500]

fake_human, fake_machine = fakedata(raw_text)

sample_1, sample_2, nw_matrix, path, duration = nwalgorithm.best_align(fake_human, fake_machine, 1, -1, -1, -1)

print(sample_1)
print(sample_2)
print(duration)