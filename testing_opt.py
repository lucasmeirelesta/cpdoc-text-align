import nwalgorithm
import functions
from bayes_opt import BayesianOptimization

with open("Entrevistas/2347_carlos_santos_cruz_2016-10-21_01.txt", 'r') as f:
    transcricao_humana = f.read()
    transcricao_humana = functions.parse_transcription_human(transcricao_humana)

with open("transcription/transcricao-2347_carlos_santos_cruz_2016-10-21_01.txt", 'r') as f:
    transcricao_maquina = f.read()
    transcricao_maquina = functions.parse_transcription_machine(transcricao_maquina)


def opt_hyperparameters(m, mm, wi, ws):

    sample_1, sample_2, _, _, _ = nwalgorithm.best_align(transcricao_humana, transcricao_maquina,
                                                                           m, mm, wi, ws, distance=False)
    score = functions.score_match(sample_1, sample_2)

    return score

bounds = {
    'm':(0, 10),
    'mm':(-10, 0) ,
    'wi':(-10, 0) ,
    'ws':(-10, 0)
}

optimizer = BayesianOptimization(
    f=opt_hyperparameters,
    pbounds=bounds,
    random_state=1,
)
optimizer.maximize(init_points=10, n_iter=50)