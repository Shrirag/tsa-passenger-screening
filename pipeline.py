from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from vectorize import vectorize
import pickle, time
import numpy as np

# Data file paths
DATA_PATH = '' # Path of directory containing 'aps' files
LABEL_PATH = '' # Path of csv file with labels

# Pickle dump file paths
VECTORS_PATH = ''
RFC_GRID_SEARCH_PATH = ''
GBC_GRID_SEARCH_PATH = ''
FINAL_RESULTS_PATH = ''

# Result paths
RFC_GRID_SEARCH_GRAPH_PATH = ''
GBC_GRID_SEARCH_GRAPH_PATH = ''
FINAL_RESULTS_GRAPH_PATH = ''

# ScikitLearn parameters
NUM_CORES = 4 # Number of cores available on machine
GRID_SEARCH_CV_FOLDS = 3 # Number of folds for parameter selection cross validation
CV_FOLDS = 3 # Number of folds for final cross validation

RANDOM_STATE = np.random.RandomState(25)

def param_selection_heat_map(results, px_len, py_len, CV_FOLDS, GRAPH_PATH):

    params = results['params']
    param_scores = []
    for i in range(CV_FOLDS):
        param_scores.append(results['split%s_test_score' % (i)])

    param_scores = [sum(x)/len(x) for x in np.asarray(param_scores).transpose()]

    ps = [] 
    for i in range(len(params)): 
        ps.append({
            'max_depth': params[i]['max_depth'],
            'n_estimators': params[i]['n_estimators'],
            'score': param_scores[i]
        })

    x, y = [], []
    for i in range(len(ps)): 
        if ps[i]['max_depth'] not in x: x.append(ps[i]['max_depth'])
        if ps[i]['n_estimators'] not in y: y.append(ps[i]['n_estimators'])
    z = np.asarray([ps[i]['score'] for i in range(len(ps))]).ravel()

    arr = []
    for i in range(0, len(params), px_len):
        row = []
        for j in range(px_len): row.append(z[i+j])
        arr.append(row)

    fig, axarr = plt.subplots(1, 1, figsize=(10, 10))
    plot = axarr.imshow(arr, aspect='auto')
    axarr.set_xticks([i for i in range(len(y))], minor=False)
    axarr.set_yticks([i for i in range(len(x))], minor=False)
    axarr.set_xticklabels(y, minor=False)
    axarr.set_yticklabels(x, minor=False)
    axarr.set_xlabel('Number of Estimators')
    axarr.set_ylabel('Max Depth')
    fig.colorbar(plot)
    plt.savefig(GRAPH_PATH)

def main():


    ########################## 
    # Dataset initialization # 
    ########################## 

    print('Dataset initialization')
    
    try :

        vectors = pickle.load(open(VECTORS_PATH, 'rb'))
        xs, ys = vectors['xs'], vectors['ys']

    except FileNotFoundError:

        xs, ys = vectorize(DATA_PATH, LABEL_PATH)
        pickle.dump({ 'xs': xs, 'ys': ys }, open(VECTORS_PATH, 'wb'))

    ##########################
    # Parameter Optimization #
    ##########################

    print('Parameter Optimization')

    # xi, yi is the subset of the dataset used for optimizing the hyperparameters
    xi, yi = xs[1], ys[1]
    x_train, x_test, y_train, y_test = train_test_split(xi, yi, random_state=RANDOM_STATE)

    # Random Forest Parameter Grid
    rfc_param_grid = [{
        'n_estimators': [i for i in range(10, 50, 10)],
        'max_depth': [i for i in range(1, 5)],
        'n_jobs': [NUM_CORES],
        'random_state': [RANDOM_STATE] 
    }]
    rfc_px_len = len(rfc_param_grid[0]['n_estimators'])
    rfc_py_len = len(rfc_param_grid[0]['max_depth'])

    # Gradient Boost Parameter Grid
    gbc_param_grid = [{
        'n_estimators': [i for i in range(10, 50, 10)],
        'max_depth': [i for i in range(1, 5)],
        'random_state': [RANDOM_STATE] 
    }]
    gbc_px_len = len(gbc_param_grid[0]['n_estimators'])
    gbc_py_len = len(gbc_param_grid[0]['max_depth'])

    # Random Forest
    print('\tRandom Forest')

    try :

        rfc_results = pickle.load(open(RFC_GRID_SEARCH_PATH, 'rb'))
        param_selection_heat_map(rfc_results, rfc_px_len, rfc_py_len, GRID_SEARCH_CV_FOLDS, RFC_GRID_SEARCH_GRAPH_PATH)

    except FileNotFoundError:

        rfc = RandomForestClassifier()
        clf = GridSearchCV(estimator=rfc, param_grid=rfc_param_grid, cv=GRID_SEARCH_CV_FOLDS, n_jobs=NUM_CORES)
        clf.fit(x_train, y_train)
        rfc_results = clf.cv_results_
        pickle.dump(rfc_results, open(RFC_GRID_SEARCH_PATH, 'wb'))
        param_selection_heat_map(rfc_results, rfc_px_len, rfc_py_len, GRID_SEARCH_CV_FOLDS, RFC_GRID_SEARCH_GRAPH_PATH)

    # Gradient Boosted Trees 
    
    print('\tGradient Boosted Trees')

    try:

        gbc_results = pickle.load(open(GBC_GRID_SEARCH_PATH, 'rb')) 
        param_selection_heat_map(gbc_results, gbc_px_len, gbc_py_len, GRID_SEARCH_CV_FOLDS, GBC_GRID_SEARCH_GRAPH_PATH)

    except FileNotFoundError:

        gbc = GradientBoostingClassifier()
        clf = GridSearchCV(estimator=gbc, param_grid=gbc_param_grid, cv=GRID_SEARCH_CV_FOLDS, n_jobs=NUM_CORES)
        clf.fit(x_train, y_train)
        gbc_results = clf.cv_results_
        pickle.dump(gbc_results, open(GBC_GRID_SEARCH_PATH, 'wb'))
        param_selection_heat_map(gbc_results, gbc_px_len, gbc_py_len, GRID_SEARCH_CV_FOLDS, GBC_GRID_SEARCH_GRAPH_PATH)

    #################### 
    # Final Train/Test # 
    #################### 

    print('Final Train/Test')

    try:

        final_scores = pickle.load(open(FINAL_RESULTS_PATH, 'rb'))

    except FileNotFoundError:

        RFC_OPT_MAX_DEPTH = 1
        RFC_OPT_N_ESTIMATORS = 10
        GBC_OPT_MAX_DEPTH = 1
        GBC_OPT_N_ESTIMATORS = 10 

        final_scores = {}
        for i in range(1, 18):
            rfc = RandomForestClassifier(max_depth=RFC_OPT_MAX_DEPTH, n_estimators=RFC_OPT_N_ESTIMATORS, n_jobs=NUM_CORES, random_state=RANDOM_STATE)
            gbc = GradientBoostingClassifier(max_depth=GBC_OPT_MAX_DEPTH, n_estimators=GBC_OPT_N_ESTIMATORS, random_state=RANDOM_STATE)
            final_scores[i] = {}
            final_scores[i]['rfc'] = cross_val_score(rfc, xs[i], ys[i], cv=CV_FOLDS, n_jobs=NUM_CORES)
            final_scores[i]['gbc'] = cross_val_score(gbc, xs[i], ys[i], cv=CV_FOLDS, n_jobs=NUM_CORES)

        pickle.dump(final_scores, open(FINAL_RESULTS_PATH, 'wb'))

    fig, axarr = plt.subplots(5, 4, figsize=(25, 25))
    for i in range(1, 18):
        a, b = final_scores[i]['rfc'], final_scores[i]['gbc']
        row, col = int((i-1)/4), (i-1)%4
        axarr[row][col].boxplot([a, b])
        axarr[row][col].set_title('Body Zone %s' % (i))
        axarr[row][col].set_xticklabels(['RFC', 'GBC'])
        axarr[row][col].set_ylabel('Accuracy')
    for i in range(1, 4): axarr[4][i].axis('off')
    plt.savefig(FINAL_RESULTS_GRAPH_PATH)

if __name__ == '__main__' : main()
