from sklearn.model_selection import cross_val_score as cvs
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from vectorize import vectorize
import numpy as np
import pickle

def main():

    # Data file paths
    DATA_PATH = '' # Path of directory containing 'aps' files
    LABEL_PATH = '' # Path of csv file with labels

    # Pickle dump file paths
    VECTORS_PATH = ''
    SCORES_PATH = '' 

    NUM_CORES = 1 # Number of cores available on machine

    RANDOM_STATE = np.random.RandomState(25)

    try :

        vectors = pickle.load(open(VECTORS_PATH, 'rb'))
        xs, ys = vectors['xs'], vectors['ys']

    except FileNotFoundError:

        xs, ys = vectorize(DATA_PATH, LABEL_PATH)
        pickle.dump({ 'xs': xs, 'ys': ys }, open(VECTORS_PATH, 'wb')

    try:

        scores = pickle.load(open(SCORES_PATH, 'rb'))

    except FileNotFoundError:

        scores = {}
        for i in range(1, 18):
            x1, y1 = xs[i], ys[i]
            rfc = RandomForestClassifier(n_jobs=NUM_CORES, random_state=RANDOM_STATE)
            gbc = GradientBoostingClassifier(n_jobs=NUM_CORES, random_state=RANDOM_STATE)
            rscores = cvs(rfc, x1, y1, cv=5, n_jobs=NUM_CORES)
            gscores = cvs(gbc, x1, y1, cv=5, n_jobs=NUM_CORES)
            scores[i] = { 'rfc': rscores, 'gbc': gscores }

        pickle.dump(scores, open(SCORES_PATH, 'wb'))

if __name__ == '__main__' : main()
