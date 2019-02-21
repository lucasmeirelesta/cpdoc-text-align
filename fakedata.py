import random
import numpy as np
from nltk.corpus import machado
import nwalgorithm

def fakedata(phrase, seed=42):

    phrase = phrase.split()

    phrase_fake_human = phrase
    phrase_fake_machine = phrase

    random.seed(seed)
    loss_human = np.random.uniform(0,0.12)
    random.seed(seed)
    loss_machine = np.random.uniform(0.1, 0.3)

    length_phrase = len(phrase)

    words_loss_human = sorted(np.random.choice(range(0, length_phrase), int(np.floor(length_phrase * loss_human))), reverse=True)
    words_loss_machine = sorted(np.random.choice(range(0, length_phrase), int(np.floor(length_phrase * loss_machine))), reverse=True)

    phrase_fake_human = [phrase for _, phrase in enumerate(phrase_fake_human) if _ not in words_loss_human]
    phrase_fake_machine = [phrase for _, phrase in enumerate(phrase_fake_machine) if _ not in words_loss_machine]

    random.seed(seed)
    loss_machine = np.random.uniform(0.05, 0.1)

    random.seed(seed)
    words_change = np.random.choice(range(0, len(phrase_fake_machine)),int(np.floor(len(phrase_fake_machine) * loss_machine)),replace=False)
    random.seed(seed+1)
    move_to = np.random.choice(range(0, len(phrase_fake_machine)),int(np.floor(len(phrase_fake_machine) * loss_machine)),replace=False)

    for i in range(0, len(move_to)):
        aux = phrase_fake_machine[move_to[i]]
        phrase_fake_machine[move_to[i]] = phrase_fake_machine[words_change[i]]
        phrase_fake_machine[words_change[i]] = aux

    phrase_fake_human = ' '.join(phrase_fake_human)
    phrase_fake_machine = ' '.join(phrase_fake_machine)


    return phrase_fake_human, phrase_fake_machine


raw_text = machado.raw('romance/marm05.txt')
raw_text = raw_text[998:1500]

#print(machado.readme())

fake_human, fake_machine = fakedata(raw_text)

sample_1, sample_2, nw_matrix, caminho, duration = nwalgorithm.best_align(fake_human,fake_machine,1,0,-1,-1)

print(sample_1)
print(sample_2)
print(duration)