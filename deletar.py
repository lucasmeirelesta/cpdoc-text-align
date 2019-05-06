import nwalgorithm
import numpy as np
import utils
import subprocess
from os import listdir
import re

a = [i.split('.')[0] for i in listdir('Entrevistas/') if 'wav' in i]
a.remove('pho_2303_marcel_biato_2016-06-15_03')
aux = []
for i in a:

    name_interview = i

    with open("Entrevistas/" + name_interview + ".txt", 'r') as f:
        human_transcription = f.read()
        human_transcription = utils.clean_transcription_human(human_transcription)

    with open("transcription/transcricao-" + name_interview + ".txt", 'r') as f:
        machine_transcription = f.read()
        machine_transcription = utils.clean_transcription_machine(machine_transcription)

    aux.append(1-len(machine_transcription.split())/len(human_transcription.split()))

np.mean(aux)

from nltk import edit_distance

edit_distance(human_transcription, machine_transcription)

name_interview = "pho_2309_gala_irene_2016-06-06_01"

with open("transcription/transcricao-" + name_interview + ".txt", 'r') as f:
    transcription = f.read()
    transcription = re.findall(r'confidence:(.*?)\n\nWord:', transcription)
    transcription = [float(i.strip()) for i in transcription]

np.median(transcription)
np.percentile(transcription, 10)
np.min(transcription)
np.max(transcription)