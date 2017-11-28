from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import SparsePCA
from xgboost import XGBClassifier

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from vectorize import vectorize
import pandas as pd
import pickle, time
import numpy as np

# Data file paths
DATA_PATH = '' # Path of directory containing 'aps' files
LABEL_PATH = '' # Path of csv file with labels

# Pickle dump file paths
VECTORS_PATH = ''
RFC_GRID_SEARCH_PATH = ''
GBC_GRID_SEARCH_PATH = ''
XGB_GRID_SEARCH_PATH = ''
FINAL_RESULTS_PATH = ''

# Result paths
RFC_GRID_SEARCH_GRAPH_PATH = ''
GBC_GRID_SEARCH_GRAPH_PATH = ''
XGB_GRID_SEARCH_GRAPH_PATH = ''
FINAL_RESULTS_GRAPH_PATH = ''
CLASS_DIST_BAR_GRAPH_PATH = ''

# ScikitLearn parameters
NUM_CORES = # Number of cores available on machine
GRID_SEARCH_CV_FOLDS = # Number of folds for parameter selection cross validation
CV_FOLDS =  # Number of folds for final cross validation

RANDOM_STATE = np.random.RandomState(25)
RANDOM_STATE_XGB = 25

def class_dist_bar(LABEL_PATH):
    
    df = pd.read_csv(LABEL_PATH)
    
    zone_count = {}
    for i in range(1,18):
        temp = df[df['Id'].str.endswith('Zone'+str(i))]
        zone_count[i] = (np.sum(temp['Probability']==0), np.sum(temp['Probability']==1))

    neg_class = [zone_count[i][0] for i in zone_count]
    pos_class = [zone_count[i][1] for i in zone_count]

    plt.figure(figsize = (6,6))
    width = 0.8
    indices = np.arange(len(neg_class))
    plt.bar(indices, neg_class, width=width, 
            color='g', label='Negative Class')
    plt.bar(indices, pos_class, 
            width=0.4*width, color='r', alpha=0.5, label='Positive Class')
    plt.xticks(indices, 
               ['{}'.format(i+1) for i in range(len(neg_class))] )
    plt.xlabel("Body Zone")
    plt.ylabel("Instances")
    plt.title("Class Distribution Among Zones")
    plt.legend()
    plt.savefig(CLASS_DIST_BAR_GRAPH_PATH)

def optimize_hyper_params(model, param_grid, xs, ys):
	x_ps, y_ps = {}, {} 
	for i in range(1, 18):
		x_ps[i], y_ps[i] = [], []
		rng = len(xs[i])
		# count = int(len(xs[i]) * .40)
		count = len(xs[i])
		indices = np.random.choice([i for i in range(0, rng)], size=count, replace=False)
		for j in indices:
			x_ps[i].append(xs[i][j])
			y_ps[i].append(ys[i][j])

	results = {}
	for i in range(1, 18):
		print('\t\t%s\tzone\t%s' % (model, i))
		est = None
		if model == 'rfc': est = RandomForestClassifier()
		elif model == 'gbc': est = GradientBoostingClassifier()
		elif model == 'xgb': est = XGBClassifier()

		if model == 'rfc': clf = GridSearchCV(estimator=est, param_grid=param_grid, cv=CV_FOLDS, scoring='f1', n_jobs=NUM_CORES)
		elif model == 'gbc': clf = GridSearchCV(estimator=est, param_grid=param_grid, cv=CV_FOLDS, scoring='f1', n_jobs=NUM_CORES)
		elif model == 'xgb': clf = GridSearchCV(estimator=est, param_grid=param_grid, cv=CV_FOLDS)

		clf.fit(np.array(x_ps[i]), np.array(y_ps[i]))
		results[i] = clf.cv_results_

	return results

def param_selection_heat_map(results, px_len, py_len, CV_FOLDS, GRAPH_PATH, title):
    fig, axarr = plt.subplots(5, 4, figsize=(30, 30))
    for k in range(1, 18):

        params = results[k]['params']
        param_scores = []
        for i in range(CV_FOLDS):
            param_scores.append(results[k]['split%s_test_score' % (i)])

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

        row, col = int((k-1)/4), (k-1)%4
        img = axarr[row][col].imshow(arr)
        divider = make_axes_locatable(axarr[row][col])
        cax = divider.append_axes("right", size="10%", pad=0.05) 
        axarr[row][col].set_xticks([i for i in range(len(y))], minor=False)
        axarr[row][col].set_yticks([i for i in range(len(x))], minor=False)
        axarr[row][col].set_xticklabels(y, minor=False, fontsize=18)
        axarr[row][col].set_yticklabels(x, minor=False, fontsize=18)
        axarr[row][col].set_xlabel('Number of Estimators', fontsize=24)
        axarr[row][col].set_ylabel('Max Depth', fontsize=24)
        axarr[row][col].set_title('Zone %s' % (k), fontsize=28)
        cbar = plt.colorbar(img, cax)
        cbar.set_label('F1 Score',size=18)
        cbar.ax.tick_params(labelsize=14)

        for i in range(1,4): axarr[4][i].axis('off')
        
        plt.suptitle(title, fontsize=30, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=.95)
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
        
    print('Class Distribution Bar Graph')
    class_dist_bar(LABEL_PATH)

    ##########################
    # Parameter Optimization #
    ##########################

    print('Parameter Optimization')
    max_depth = int(len(xs[1]) * .40) - 1
    single = int(max_depth/5)

    Random Forest Parameter Grid
    rfc_param_grid = [{
        'n_estimators': [i for i in range(100, 1100, 100)],
        'max_depth': [i for i in range(2, 22, 2)],
        # 'n_jobs': [NUM_CORES],
        'random_state': [RANDOM_STATE] 
    }]
    rfc_px_len = len(rfc_param_grid[0]['n_estimators'])
    rfc_py_len = len(rfc_param_grid[0]['max_depth'])

    # Gradient Boost Parameter Grid
    gbc_param_grid = [{
        'n_estimators': [i for i in range(100, 1100, 100)],
        'max_depth': [i for i in range(2, 22, 2)],
        'random_state': [RANDOM_STATE] 
    }]
    gbc_px_len = len(gbc_param_grid[0]['n_estimators'])
    gbc_py_len = len(gbc_param_grid[0]['max_depth'])
      
    # XGBoost Parameter Grid
    xgb_param_grid = [{
        'nthread': [NUM_CORES], 
        'objective': ['binary:logistic'],
        'learning_rate': [0.05], 
        'n_estimators': [i for i in range(100, 1200, 100)],
        'max_depth': [i for i in range(2, 22, 2)],
        'seed': [RANDOM_STATE_XGB]
    }]

    xgb_px_len = len(xgb_param_grid[0]['n_estimators'])
    xgb_py_len = len(xgb_param_grid[0]['max_depth'])
    
    Random Forest
    print('\tRandom Forest')

    try :

        rfc_results = pickle.load(open(RFC_GRID_SEARCH_PATH, 'rb'))
        param_selection_heat_map(rfc_results, rfc_px_len, rfc_py_len, GRID_SEARCH_CV_FOLDS, RFC_GRID_SEARCH_GRAPH_PATH, 'Random Forest Classifier Parameter Selection')

    except FileNotFoundError:

        rfc_results = optimize_hyper_params('rfc', rfc_param_grid, xs, ys)
        pickle.dump(rfc_results, open(RFC_GRID_SEARCH_PATH, 'wb'))
        param_selection_heat_map(rfc_results, rfc_px_len, rfc_py_len, GRID_SEARCH_CV_FOLDS, RFC_GRID_SEARCH_GRAPH_PATH, 'Random Forest Classifier Parameter Selection')

    # Gradient Boosted Trees 
      
    print('\tGradient Boosted Trees')

    try:

        gbc_results = pickle.load(open(GBC_GRID_SEARCH_PATH, 'rb')) 
        param_selection_heat_map(gbc_results, gbc_px_len, gbc_py_len, GRID_SEARCH_CV_FOLDS, GBC_GRID_SEARCH_GRAPH_PATH, 'Gradient Boosted Trees Parameter Selection')

    except FileNotFoundError:

        gbc_results = optimize_hyper_params('gbc', gbc_param_grid, xs, ys)
        pickle.dump(gbc_results, open(GBC_GRID_SEARCH_PATH, 'wb'))
        param_selection_heat_map(gbc_results, gbc_px_len, gbc_py_len, GRID_SEARCH_CV_FOLDS, GBC_GRID_SEARCH_GRAPH_PATH, 'Gradient Boosted Trees Parameter Selection')

    # XGBoost

    print('\tXGBoost')
    
    try:

        xgb_results = pickle.load(open(XGB_GRID_SEARCH_PATH, 'rb')) 
        param_selection_heat_map(xgb_results, xgb_px_len, xgb_py_len, GRID_SEARCH_CV_FOLDS, XGB_GRID_SEARCH_GRAPH_PATH, 'XGBoost Parameter Selection')

    except FileNotFoundError:

        xgb_results = optimize_hyper_params('xgb', xgb_param_grid, xs, ys)
        pickle.dump(xgb_results, open(XGB_GRID_SEARCH_PATH, 'wb'))
        param_selection_heat_map(xgb_results, xgb_px_len, xgb_py_len, GRID_SEARCH_CV_FOLDS, XGB_GRID_SEARCH_GRAPH_PATH, 'XGBoost Trees Parameter Selection')

    #################### 
    # Final Train/Test # 
    #################### 

    print('Final Train/Test')

    try:

        final_scores = pickle.load(open(FINAL_RESULTS_PATH, 'rb'))

    except FileNotFoundError:

        opt_params = { 
            'rfc': {
                1:  { 'n_estimators': 500, 'max_depth': 10, 'random_state': RANDOM_STATE }, 
                2:  { 'n_estimators': 500, 'max_depth': 10, 'random_state': RANDOM_STATE },
                3:  { 'n_estimators': 500, 'max_depth': 10, 'random_state': RANDOM_STATE }, 
                4:  { 'n_estimators': 500, 'max_depth': 10, 'random_state': RANDOM_STATE },
                5:  { 'n_estimators': 500, 'max_depth': 10, 'random_state': RANDOM_STATE }, 
                6:  { 'n_estimators': 500, 'max_depth': 10, 'random_state': RANDOM_STATE },
                7:  { 'n_estimators': 500, 'max_depth': 10, 'random_state': RANDOM_STATE }, 
                8:  { 'n_estimators': 500, 'max_depth': 10, 'random_state': RANDOM_STATE },
                9:  { 'n_estimators': 500, 'max_depth': 10, 'random_state': RANDOM_STATE }, 
                10: { 'n_estimators': 500, 'max_depth': 10, 'random_state': RANDOM_STATE },
                11: { 'n_estimators': 100, 'max_depth': 20, 'random_state': RANDOM_STATE }, 
                12: { 'n_estimators': 500, 'max_depth': 10, 'random_state': RANDOM_STATE },
                13: { 'n_estimators': 500, 'max_depth': 10, 'random_state': RANDOM_STATE }, 
                14: { 'n_estimators': 500, 'max_depth': 10, 'random_state': RANDOM_STATE },
                15: { 'n_estimators': 500, 'max_depth': 10, 'random_state': RANDOM_STATE }, 
                16: { 'n_estimators': 100, 'max_depth': 16, 'random_state': RANDOM_STATE },
                17: { 'n_estimators': 500, 'max_depth': 10, 'random_state': RANDOM_STATE },
            },  
            'gbc': {
                1: { 'n_estimators': 100, 'max_depth': 2, 'random_state': RANDOM_STATE }, 
                2: { 'n_estimators': 100, 'max_depth': 2, 'random_state': RANDOM_STATE },
                3: { 'n_estimators': 600, 'max_depth': 4, 'random_state': RANDOM_STATE }, 
                4: { 'n_estimators': 100, 'max_depth': 2, 'random_state': RANDOM_STATE },
                5: { 'n_estimators': 100, 'max_depth': 10, 'random_state': RANDOM_STATE }, 
                6: { 'n_estimators': 100, 'max_depth': 8, 'random_state': RANDOM_STATE },
                7: { 'n_estimators': 100, 'max_depth': 4, 'random_state': RANDOM_STATE }, 
                8: { 'n_estimators': 200, 'max_depth': 12, 'random_state': RANDOM_STATE },
                9: { 'n_estimators': 100, 'max_depth': 8, 'random_state': RANDOM_STATE }, 
                10: { 'n_estimators': 100, 'max_depth': 10, 'random_state': RANDOM_STATE },
                11: { 'n_estimators': 900, 'max_depth': 4, 'random_state': RANDOM_STATE }, 
                12: { 'n_estimators': 200, 'max_depth': 2, 'random_state': RANDOM_STATE },
                13: { 'n_estimators': 100, 'max_depth': 2, 'random_state': RANDOM_STATE }, 
                14: { 'n_estimators': 300, 'max_depth': 4, 'random_state': RANDOM_STATE },
                15: { 'n_estimators': 100, 'max_depth': 2, 'random_state': RANDOM_STATE }, 
                16: { 'n_estimators': 300, 'max_depth': 8, 'random_state': RANDOM_STATE },
                17: { 'n_estimators': 100, 'max_depth': 10, 'random_state': RANDOM_STATE },
            }   
        }   

        final_scores = {}
        for i in range(1, 18):
            rfc = RandomForestClassifier()
            rfc.set_params(**opt_params['rfc'][i])
            gbc = GradientBoostingClassifier()
            gbc.set_params(**opt_params['gbc'][i])
            # xgb = XGBClassifier()
            final_scores[i] = {}
            final_scores[i]['rfc'] = cross_val_score(rfc, xs[i], ys[i], cv=CV_FOLDS, n_jobs=NUM_CORES, scoring='f1')
            final_scores[i]['gbc'] = cross_val_score(gbc, xs[i], ys[i], cv=CV_FOLDS, n_jobs=NUM_CORES, scoring='f1')
            # final_scores[i]['xgb'] = cross_val_score(xgb, np.array(xs[i]), np.array(ys[i]), cv=CV_FOLDS, scoring='f1')

        pickle.dump(final_scores, open(FINAL_RESULTS_PATH, 'wb'))

    fig, axarr = plt.subplots(5, 4, figsize=(25, 25))
    for i in range(1, 18):

        a, b = final_scores[i]['rfc'], final_scores[i]['gbc']
        # a, b, c = final_scores[i]['rfc'], final_scores[i]['gbc'], final_scores[i]['xgb']
        row, col = int((i-1)/4), (i-1)%4
        axarr[row][col].boxplot([a, b])
        # axarr[row][col].boxplot([a, b, c])
        axarr[row][col].set_title('Body Zone %s' % (i), fontsize=28)
        axarr[row][col].set_xticklabels(['RFC', 'GBC'], fontsize=24)
        # axarr[row][col].set_xticklabels(['RFC', 'GBC', 'XGB'])
        axarr[row][col].set_ylabel('Accuracy', fontsize=24)

    for i in range(1, 4): axarr[4][i].axis('off')
    plt.suptitle("Model Comparison", fontsize=30, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=.95)
    plt.savefig(FINAL_RESULTS_GRAPH_PATH)

if __name__ == '__main__' : main()
