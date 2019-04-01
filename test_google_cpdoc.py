import nwalgorithm
import numpy as np
import genfunctions

name_interview = "2347_carlos_santos_cruz_2016-10-21_01"

with open("Entrevistas/" + name_interview + ".txt", 'r') as f:
    human_transcription = f.read()
    human_transcription = genfunctions.clean_transcription_human(human_transcription)

with open("transcription/transcricao-" + name_interview + ".txt", 'r') as f:
    machine_transcription = f.read()
    machine_transcription = genfunctions.clean_transcription_machine(machine_transcription)

with open("transcription/transcricao-" + name_interview + ".txt", 'r') as f:
    machine_transcription_word = f.read()
    machine_transcription_word = machine_transcription_word.split('Word:')[1:]
    machine_transcription_word = [i.split('\n')[0].strip() for i in machine_transcription_word]

# Hyperparameters
match = 1
mismatch = -1
gapsmall = -1
gapbig = -1

# Align the two texts
sample_1, sample_2, nw_matrix, path, duration = nwalgorithm.best_align(human_transcription, machine_transcription,
                                                                       match, mismatch, gapsmall, gapbig,
                                                                       distance=False)

# Saving each position with the time that word appear, if not a gap.
dic_pos_time = genfunctions.phrase_dic(sample_1, machine_transcription_word)

human_transcription_align = sample_2.split()  # in our case, human transcription is a way bigger than the machine
machine_transcription_align = sample_1.split()

subtitle_name = name_interview + "_HP_{}{}{}{}.srt".format(str(match), str(mismatch), str(gapsmall), str(gapbig))

genfunctions.subtitle_gen('subtitle/'+subtitle_name, human_transcription_align, machine_transcription_align, dic_pos_time)


#Graphs
import seaborn as sns
import matplotlib.pyplot as plt


heatmap = sns.heatmap(nw_matrix, cmap = 'ocean')
cbar = heatmap.collections[0].colorbar
cbar.set_ticks([np.min(nw_matrix), np.max(nw_matrix)])
cbar.set_ticklabels(['Min','Max'])
plt.show(heatmap)
