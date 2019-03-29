"""

This project test the Bayesian Optimization method

"""


import time
import genfunctions
from nwalgorithm import best_align
from bayes_opt import BayesianOptimization


with open("Entrevistas/pho_165_juscelino_kubitschek_i_1974-03-01_01a.txt", 'r') as f:
    human_transcription = f.read()
    human_transcription = genfunctions.clean_transcription_human(human_transcription)

with open("transcription/transcricao-pho_165_juscelino_kubitschek_i_1974-03-01_01a.txt", 'r') as f:
    machine_transcription = f.read()
    machine_transcription = genfunctions.clean_transcription_machine(machine_transcription)


def opt_hyperparameters(match, mismatch, gapsmall, gapbig):
    """
    Function to be used at the Bayesian Optimization.
    Run the Needleman-Wunsch algorithm and return a score, using the score_match function.

    :param match: (Float, default none) The weight that the algorithm will use for a
    match at the Needleman-Wunsch Algorithm.
    :param mismatch: (Float, default none) The weight that the algorithm will use for a
    mismatch at the Needleman-Wunsch Algorithm.
    :param gapsmall: (Float, default none) The weight that the algorithm will use for a
    gap at the Needleman-Wunsch Algorithm.
    :param gapbig: (Float, default none) The weight that the algorithm will use for a
    gap at the Needleman-Wunsch Algorithm.
    :return:
    A Score.
    """

    # Using underline for unnecessary information
    sample_1, sample_2, _, _, _ = best_align(human_transcription, machine_transcription,
                                             match, mismatch, gapsmall, gapbig, distance=False)

    score = genfunctions.score_match(sample_1, sample_2)[0]  # Only the first information is important in this case.

    return score

# Limits that will be tested the hyperparameters.
bounds = {
    'match': (0, 4),
    'mismatch': (-4, 0),
    'gapsmall': (-4, 0),
    'gapbig': (-4, 0)
}

optimizer = BayesianOptimization(
    f=opt_hyperparameters,
    pbounds=bounds,
    random_state=42
)


duration = time.time()

optimizer.maximize(init_points=20, n_iter=50)

duration = time.time() - duration

print(duration)
