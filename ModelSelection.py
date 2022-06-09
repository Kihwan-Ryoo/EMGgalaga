import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from pytictoc import TicToc

def Analysis(data, path, name):
    t = TicToc()
    t.tic()

    data.rename(columns={'class':'group'}, inplace=True)
    group_dummies = pd.get_dummies(data.group, prefix='group')
    group_dummies.drop(group_dummies.columns[0:3], axis=1, inplace=True)
    group_dummies.drop(group_dummies.columns[2:4], axis=1, inplace=True)
    data = pd.concat([data, group_dummies], axis=1)

    yd3 = ['channel1', 'channel2', 'channel3', 'channel4',
           'channel5', 'channel6', 'channel7', 'channel8', 'group_3']
    YD3 = data[yd3]
    yd4 = ['channel1', 'channel2', 'channel3', 'channel4',
           'channel5', 'channel6', 'channel7', 'channel8', 'group_4']
    YD4 = data[yd4]

    yd_x_full = ['channel1', 'channel2', 'channel3', 'channel4',
                 'channel5', 'channel6', 'channel7', 'channel8']
    X = data[yd_x_full]
    y3 = data.group_3
    y4 = data.group_4

    # Train Test Split
    X3_train, X3_test, y3_train, y3_test = train_test_split(X, y3, train_size=train_size, random_state=seed)
    X4_train, X4_test, y4_train, y4_test = train_test_split(X, y4, train_size=train_size, random_state=seed)

    '''
    1. Logistic Regression
    '''
    print('LR Analysis')
    ScaledLR = Pipeline([('Scaler', MinMaxScaler(feature_range=(0, 1))), ('LR', LogisticRegression(random_state=seed, max_iter=10000))])
    c_set = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    penalty_set = ['l2']
    param_grid = dict(LR__C=c_set, LR__penalty=penalty_set)

    '''
    1-1. LR Group3 (Wrist Flexion)
    '''
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=ScaledLR, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X3_train, y3_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_result.cv_results_['params']):
        print("mean:%0.3f (std:%0.03f) for %r"
              % (mean, std, params))
    fine_tuned_LR3 = grid_result.best_estimator_

    # AUROC
    Y3_score = fine_tuned_LR3.predict_proba(X3_test)
    Y3_PRED_LR = Y3_score[:, -1]
    fpr, tpr, _ = roc_curve(y3_test, Y3_PRED_LR)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.rcParams['figure.figsize'] = (6, 6)
    plt.rcParams['font.size'] = 12
    f1 = plt.figure()
    plt.plot(fpr, tpr, 'lawngreen', label='ROC curve of LR\nAUC:%.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC (Wrist Flexion) ' + name[:-4])
    plt.legend(loc="lower right")

    '''
    1-2. LR Group4 (Wrist Extension)
    '''
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=ScaledLR, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X4_train, y4_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_result.cv_results_['params']):
        print("mean:%0.3f (std:%0.03f) for %r"
              % (mean, std, params))
    fine_tuned_LR4 = grid_result.best_estimator_

    # AUROC
    Y4_score = fine_tuned_LR4.predict_proba(X4_test)
    Y4_PRED_LR = Y4_score[:, -1]
    fpr, tpr, _ = roc_curve(y4_test, Y4_PRED_LR)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.rcParams['figure.figsize'] = (6, 6)
    plt.rcParams['font.size'] = 12
    f2 = plt.figure()
    plt.plot(fpr, tpr, 'lawngreen', label='ROC curve of LR\nAUC:%.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC (Wrist Extension) ' + name[:-4])
    plt.legend(loc="lower right")

    '''
    2. Random Forest
    '''
    print('RF Analysis')
    ScaledRF = Pipeline([('Scaler', MinMaxScaler(feature_range=(0, 1))), ('RF', RandomForestClassifier(random_state=seed))])
    n_estimators_set = [100, 200]
    max_features_set = ['auto', 'sqrt']
    param_grid = dict(RF__n_estimators=n_estimators_set, RF__max_features=max_features_set)

    '''
    2-1. RF Group3 (Wrist Flexion)
    '''
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=ScaledRF, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X3_train, y3_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_result.cv_results_['params']):
        print("mean:%0.3f (std:%0.03f) for %r"
              % (mean, std, params))
    fine_tuned_RF3 = grid_result.best_estimator_

    # AUROC
    Y3_score = fine_tuned_RF3.predict_proba(X3_test)
    Y3_PRED_RF = Y3_score[:, -1]
    fpr, tpr, _ = roc_curve(y3_test, Y3_PRED_RF)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(f1)
    plt.plot(fpr, tpr, 'magenta', label='ROC curve of RF\nAUC:%.3f' % roc_auc)
    plt.legend(loc='lower right')

    importances = pd.Series(fine_tuned_RF3.steps[1][1].feature_importances_, index=YD3.columns[0:-1])
    plt.rcParams['figure.figsize'] = (26, 12)
    plt.rcParams['font.size'] = 12
    plt.figure()
    importances.plot(kind='bar')
    plt.title('Importances (Wrist Flexion) ' + name[:-4])
    importances.to_csv(path + '/importances_' + name[:-4] +'_WristFlexion.csv', index=True, header=True)
    plt.savefig(path + '/importances_' + name[:-4] +'_WristFlexion.png', bbox_inches='tight')

    '''
    2-2. RF Group4 (Wrist Extension)
    '''
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=ScaledRF, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X4_train, y4_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_result.cv_results_['params']):
        print("mean:%0.3f (std:%0.03f) for %r"
              % (mean, std, params))
    fine_tuned_RF4 = grid_result.best_estimator_

    # AUROC
    Y4_score = fine_tuned_RF4.predict_proba(X4_test)
    Y4_PRED_RF = Y4_score[:, -1]
    fpr, tpr, _ = roc_curve(y4_test, Y4_PRED_RF)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(f2)
    plt.plot(fpr, tpr, 'magenta', label='ROC curve of RF\nAUC:%.3f' % roc_auc)
    plt.legend(loc='lower right')

    importances = pd.Series(fine_tuned_RF4.steps[1][1].feature_importances_, index=YD4.columns[0:-1])
    plt.rcParams['figure.figsize'] = (26, 12)
    plt.rcParams['font.size'] = 12
    plt.figure()
    importances.plot(kind='bar')
    plt.title('Importances (Wrist Extension) ' + name[:-4])
    importances.to_csv(path + '/importances_' + name[:-4] + '_WristExtension.csv', index=True, header=True)
    plt.savefig(path + '/importances_' + name[:-4] + '_WristExtension.png', bbox_inches='tight')

    '''
    3. Multi Layer Perceptron
    '''
    print('MLP Analysis')
    ScaledMLP = Pipeline([('Scaler', MinMaxScaler(feature_range=(0, 1))), ('MLP', MLPClassifier(max_iter=100000))])
    alpha_set = [0.01, 0.1, 1]
    solver_set = ['adam']
    hidden_layer_sizes_set = [[10, 10],
                              [10, 10, 10]]
    param_grid = dict(MLP__alpha=alpha_set,
                      MLP__solver=solver_set,
                      MLP__hidden_layer_sizes=hidden_layer_sizes_set)

    '''
    3-1. MLP Group3 (Wrist Flexion)
    '''
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=ScaledMLP, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X3_train, y3_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_result.cv_results_['params']):
        print("mean:%0.3f (std:%0.03f) for %r"
              % (mean, std, params))
    fine_tuned_MLP3 = grid_result.best_estimator_

    # AUROC
    Y3_score = fine_tuned_MLP3.predict_proba(X3_test)
    Y3_PRED_MLP = Y3_score[:, -1]
    fpr, tpr, _ = roc_curve(y3_test, Y3_PRED_MLP)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(f1)
    plt.plot(fpr, tpr, 'r', label='ROC curve of MLP\nAUC:%.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.savefig(path + '/ROCcurves_' + name[:-4] + '_WristFlexion.png', bbox_inches='tight')

    '''
    3-2. MLP Group4 (Wrist Extension)
    '''
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=ScaledMLP, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X4_train, y4_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_result.cv_results_['params']):
        print("mean:%0.3f (std:%0.03f) for %r"
              % (mean, std, params))
    fine_tuned_MLP4 = grid_result.best_estimator_

    # AUROC
    Y4_score = fine_tuned_MLP4.predict_proba(X4_test)
    Y4_PRED_MLP = Y4_score[:, -1]
    fpr, tpr, _ = roc_curve(y4_test, Y4_PRED_MLP)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(f2)
    plt.plot(fpr, tpr, 'r', label='ROC curve of MLP\nAUC:%.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.savefig(path + '/ROCcurves_' + name[:-4] + '_WristExtension.png', bbox_inches='tight')

    t.toc()

Total = TicToc()
Total.tic()

# Variables
seed = 123
train_size = 0.7
num_folds = 5
scoring = 'accuracy'

common_path = 'C:/Users/RKH/PycharmProjects/BioMedicalSystemDesignProject/EMG_data_for_gestures-master'
direct = os.listdir(common_path)
subfolders = []
for i in direct:
    if os.path.isdir(common_path + '/' +i):
        subfolders.append(i)

file_list_txt = []
for file_name in subfolders:
    path = common_path + '/' + file_name
    file_list = os.listdir(path)

    # 파일 분석
    txt = [file for file in file_list if (file.endswith('.txt'))]
    file_list_txt = file_list_txt + txt
    data1 = pd.read_csv(path + '/' + txt[0], sep='\t')
    data1_name = txt[0]
    data2 = pd.read_csv(path + '/' + txt[1], sep='\t')
    data2_name = txt[1]
    print('Analyzing\t' + path[58:])
    print('\t' + data1_name)
    Analysis(data1, path, data1_name)
    print('\t' + data2_name)
    Analysis(data2, path, data2_name)

# 분석된 파일 종합
FIF = plt.figure('Importances per channel (Wrist Flexion)')
FIE = plt.figure('Importances per channel (Wrist Extension)')
FRF = plt.figure('ROC curves (Wrist Flexion)')
FRE = plt.figure('ROC curves (Wrist Extension)')
i = 1
csv_FLEXION = []
csv_EXTENSION = []
variables = ['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8']
for file_name in subfolders:
    path = common_path + '/' + file_name
    file_list = os.listdir(path)

    importances_flexion = [file for file in file_list if (file.startswith('importances') & file.endswith('Flexion.png'))] # Class 3
    importances_extension = [file for file in file_list if (file.startswith('importances') & file.endswith('Extension.png'))] # Class 4
    ROCs_flexion = [file for file in file_list if (file.startswith('ROCcurves') & file.endswith('Flexion.png'))]
    ROCs_extension = [file for file in file_list if (file.startswith('ROCcurves') & file.endswith('Extension.png'))]
    csv_flexion = [file for file in file_list if (file.startswith('importances') & file.endswith('Flexion.csv'))]
    csv_extension = [file for file in file_list if (file.startswith('importances') & file.endswith('Extension.csv'))]
    csv_FLEXION.extend([pd.read_csv(path + '/' + csv_flexion[0]).T, pd.read_csv(path + '/' + csv_flexion[1]).T])
    csv_EXTENSION.extend([pd.read_csv(path + '/' + csv_extension[0]).T, pd.read_csv(path + '/' + csv_extension[1]).T])

    if i <= 36:
        plt.figure(FIF)
        for j in range(2):
            plt.subplot(8, 9, 2*i - (1 - j))
            plt.title(2*i - (1 - j))
            plt.imshow(mpimg.imread(path + '/' + importances_flexion[j]))
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        plt.figure(FIE)
        for j in range(2):
            plt.subplot(8, 9, 2*i - (1 - j))
            plt.title(2 * i - (1 - j))
            plt.imshow(mpimg.imread(path + '/' + importances_extension[j]))
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        plt.figure(FRF)
        for j in range(2):
            plt.subplot(6, 12, 2 * i - (1 - j))
            plt.title(2 * i - (1 - j))
            plt.imshow(mpimg.imread(path + '/' + ROCs_flexion[j]))
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        plt.figure(FRE)
        for j in range(2):
            plt.subplot(6, 12, 2 * i - (1 - j))
            plt.title(2 * i - (1 - j))
            plt.imshow(mpimg.imread(path + '/' + ROCs_extension[j]))
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
    i += 1
FIF.suptitle('Importances per channel (Wrist Flexion)')
FIF.savefig(common_path + '/importances per channel of each data_WristFlexion.png', bbox_inches='tight', dpi=300)
FIE.suptitle('Importances per channel (Wrist Extension)')
FIE.savefig(common_path + '/importances per channel of each data_WristExtension.png', bbox_inches='tight', dpi=300)
FRF.suptitle('ROC curves (Wrist Flexion)')
FRF.savefig(common_path + '/ROCcurves of each data_WristFlexion.png', bbox_inches='tight', dpi=300)
FRE.suptitle('ROC curves (Wrist Extension)')
FRE.savefig(common_path + '/ROCcurves of each data_WristExtension.png', bbox_inches='tight', dpi=300)

csv_FLEXION_all = pd.concat(csv_FLEXION, axis=0, ignore_index=True)
csv_FLEXION_all = csv_FLEXION_all[1::2]
csv_FLEXION_all.columns = variables
csv_FLEXION_all.plot(kind='box')
plt.title('Importances per channel (Wrist Flexion)')
plt.savefig(common_path + '/importances per channel of all data_WristFlexion.png', bbox_inches='tight', dpi=300)
csv_EXTENSION_all = pd.concat(csv_EXTENSION, axis=0, ignore_index=True)
csv_EXTENSION_all = csv_EXTENSION_all[1::2]
csv_EXTENSION_all.columns = variables
csv_EXTENSION_all.plot(kind='box')
plt.title('Importances per channel (Wrist Extension)')
plt.savefig(common_path + '/importances per channel of all data_WristExtension.png', bbox_inches='tight', dpi=300)

Total.toc()

plt.show()