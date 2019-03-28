import nwalgorithm
import genfunctions
from bayes_opt import BayesianOptimization
import time

#with open("Entrevistas/pho_165_juscelino_kubitschek_i_1974-03-01_01a.txt", 'r') as f:
with open("Entrevistas/2347_carlos_santos_cruz_2016-10-21_01.txt", 'r') as f:
    transcricao_humana = f.read()
    transcricao_humana = genfunctions.clean_transcription_human(transcricao_humana)

#with open("transcription/transcricao-pho_165_juscelino_kubitschek_i_1974-03-01_01a.txt", 'r') as f:
with open("transcription/transcricao-2347_carlos_santos_cruz_2016-10-21_01.txt", 'r') as f:
    transcricao_maquina = f.read()
    transcricao_maquina = genfunctions.clean_transcription_machine(transcricao_maquina)


def opt_hyperparameters(m, mm, wi, wd):

    sample_1, sample_2, _, _, _ = nwalgorithm.best_align(transcricao_humana, transcricao_maquina,
                                                                           m, mm, wi, wd, distance=False)
    score = genfunctions.score_match(sample_1, sample_2)[0]

    return score

bounds = {
    'm': (0, 4),
    'mm': (-3, 1),
    'wi': (-3, 1),
    'wd': (-3, 1)
}

duration = time.time()

optimizer = BayesianOptimization(
    f=opt_hyperparameters,
    pbounds=bounds,
    random_state=42,
)
optimizer.maximize(init_points=20, n_iter=50)

duration = time.time() - duration
print(duration)
