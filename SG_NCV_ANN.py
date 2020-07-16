import keras
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation, LeakyReLU
from RC_panel_data_new import split_fold, filepath, X, Y, seed_value
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
import numpy as np
import matplotlib
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from mlxtend.plotting import plot_confusion_matrix
import os
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC
from sklearn.metrics import roc_curve, auc
from numpy import interp
from itertools import cycle
from sklearn.utils import class_weight
import time
start_time = time.time()

os.chdir(os.getcwd())
##################################################
# Seed value (can actually be different for each attribution step)
seed_value = seed_value
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
# 5. Configure a new global `tensorflow` session
from tensorflow.compat.v1.keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# session_conf = tf.config.experimental(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


def create_model(neurons=20, lr=0.001, hidden_layers=1, dropout_rate=0.2, l2_rate=0.001,
                 activation='relu', initializer='he_uniform'):
    model_RC = Sequential()
    model_RC.add(
        Dense(neurons, kernel_initializer=initializer, input_dim=20, kernel_regularizer=regularizers.l2(l2_rate)))
    model_RC.add(BatchNormalization(momentum=0.90))
    model_RC.add(Activation(activation))
    model_RC.add(Dropout(rate=dropout_rate))

    if hidden_layers == 0:
        pass
    else:
        for i in range(hidden_layers):
            model_RC.add(Dense(neurons, kernel_initializer=initializer))
            model_RC.add(BatchNormalization(momentum=0.9))
            model_RC.add(Activation(activation))
            model_RC.add(Dropout(rate=dropout_rate))

    model_RC.add(Dense(4, kernel_initializer="uniform"))
    model_RC.add(BatchNormalization(momentum=0.90))
    model_RC.add(Activation('softmax'))
    # Compile model
    optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=10 ** -8)
    model_RC.compile(loss='categorical_crossentropy',
                     optimizer=optimizer,
                     metrics=['accuracy'])
    return model_RC


def objective_function(individual, fit_data):
    '''
    build and validate a model based on the parameters in an individual and return
    the f1 score value
    '''
    filepath = 'path'
    #  extract the values of the parameters from the individual chromosome
    lr = individual[0]
    batch_size = individual[1]
    hidden_layers = individual[2]
    neurons = individual[3]
    dropout_rate = individual[4]
    l2_rate = individual[5]

    X_train = fit_data[0]
    X_val = fit_data[1]
    Y_train = fit_data[2]
    Y_val = fit_data[3]

    # Cross-validation process
    model_RC = create_model(lr=lr, neurons=neurons, hidden_layers=hidden_layers, dropout_rate=dropout_rate,
                            l2_rate=l2_rate, initializer='he_uniform', activation='relu')
    filepath_com = filepath
    filepath_model = filepath_com + 'Loaded_model.h5'
    es = EarlyStopping(monitor='val_loss', mode='min', patience=80, verbose=0)
    mcp = ModelCheckpoint(filepath_model, monitor='val_accuracy', mode='max', save_best_only=True, verbose=0)
    callbacks = [es, mcp]

    Y_train = keras.utils.to_categorical(Y_train, num_classes=4)
    Y_val = keras.utils.to_categorical(Y_val, num_classes=4)
    model_RC.fit(X_train, Y_train,
                 batch_size=batch_size,
                 epochs=800,
                 verbose=0,
                 validation_data=(X_val, Y_val),
                 callbacks=callbacks)

    saved_model = load_model(filepath_model)
    Y_val_pred = saved_model.predict(X_val)
    Y_val_pred = np.argmax(Y_val_pred, axis=1)
    Y_val = np.argmax(Y_val, axis=1)

    train_acc = saved_model.evaluate(X_train, Y_train, verbose=0)
    print("Train accuracy: {:.2f}%".format(train_acc[1] * 100))
    acc_val = accuracy_score(Y_val, Y_val_pred)
    print("Validation accuracy: {:.2f}%".format(acc_val * 100))
    f1 = f1_score(Y_val, Y_val_pred, average='micro')
    print("Micro F1-score: {:.3f}".format(f1))
    print('-----------------------')
    return f1


def find_hyperpa(individual, findings='step_1'):

    out_kfold = StratifiedKFold(n_splits=split_fold, shuffle=True, random_state=seed_value)
    in_kfold = StratifiedKFold(n_splits=split_fold - 1, shuffle=True, random_state=seed_value)

    best_params_list = []
    best_fitness_list = []
    for i, (train_val_idx, test_idx) in enumerate(out_kfold.split(X, Y)):
        # ----Splitting train_val/test set-----
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        Y_train_val, Y_test = Y[train_val_idx], Y[test_idx]

        for j, (train_idx, val_idx) in enumerate(in_kfold.split(X_train_val, Y_train_val)):
            # ----Splitting train/val set-----
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            Y_train, Y_val = Y_train_val[train_idx], Y_train_val[val_idx]

            fit_data = [X_train, X_val, Y_train, Y_val]

            # ---------------------------optimizing hyper-parameters-----------------------------------------#
            if findings == 'step_1':
                # Design space
                learning_rate = [0.001, 0.002, 0.005, 0.01]
                batch_size = [8, 16, 32, 64]
                # Search: max f1 score
                stored_obj = {}
                for m in learning_rate:
                    for n in batch_size:
                        individual[0] = m
                        individual[1] = n
                        obj = objective_function(individual, fit_data)
                        stored_obj[m, n] = obj
                best_findings = max(stored_obj, key=stored_obj.get) 
                best_obj = max(stored_obj.values())
                # Store the findings with max obj
                print('Best findings:', best_findings)
                print('Best obj:', best_obj)
                best_params_list.append(best_findings)
                best_fitness_list.append(best_obj)

            if findings == 'step_2':
                # Design space
                hidden_layers = [0, 1, 2, 3]
                neurons = [5, 10, 15, 20, 25]
                # Search: max f1 score
                stored_obj = {}
                for m in hidden_layers:
                    for n in neurons:
                        individual[2] = m
                        individual[3] = n
                        obj = objective_function(individual, fit_data)
                        stored_obj[m, n] = obj
                best_findings = max(stored_obj, key=stored_obj.get)
                best_obj = max(stored_obj.values())
                # Store the findings with max obj
                print('Best findings:', best_findings)
                print('Best obj:', best_obj)
                best_params_list.append(best_findings)
                best_fitness_list.append(best_obj)

            if findings == 'step_3':
                # Design space
                dropout_rate = [0.1, 0.2, 0.3]
                l2_rate = [0.001, 0.01, 0.1]
                # Search: max f1 score
                stored_obj = {}
                for m in dropout_rate:
                    for n in l2_rate:
                        individual[4] = m
                        individual[5] = n
                        obj = objective_function(individual, fit_data)
                        stored_obj[m, n] = obj
                best_findings = max(stored_obj, key=stored_obj.get) 
                best_obj = max(stored_obj.values())
                # Store the findings with max obj
                print('Best findings:', best_findings)
                print('Best obj:', best_obj)
                best_params_list.append(best_findings)
                best_fitness_list.append(best_obj)
            ########## Finish searching #############

    ########## Selecting strategy for the best params #############
    print('List of best params:', best_params_list)
    print('List of best fitness:', best_fitness_list)
    # Find the most repeated params from the list of best params & fitness
    dupes = list(set([x for x in best_params_list if best_params_list.count(x) > 1]))  # List of repeated params
    print(dupes)
    if len(dupes) == 0:
        # No repeated params -> take the one with max obj
        best_params = best_params_list[best_fitness_list.index(max(best_fitness_list))]
    elif len(dupes) == 1:
        # Only 1 set of params that repeated
        best_params = dupes
    else:
        # Many repeated set of params
        # Find the params that have the most repetition with the highest mean value
        idx_dupes = []
        count = []
        for k in range(len(dupes)):
            idx_dupes.append([n for n, x in enumerate(best_params_list) if x == dupes[k]])  # Find index of all repeated params
            count.append(len(idx_dupes[k]))  # Count the number of repetition in each repeated params
        idx_dupes = [idx_dupes[j] for j in [n for n, i in enumerate(count) if i == max(count)]]  # Shrink the idx_dupes
        # If the found params have the same number of repetition -> select the one with highest mean of fitness
        best_fitness_list = np.array(best_fitness_list)
        avr = [np.mean(best_fitness_list[i]) for i in [sub for sub in idx_dupes]]  # Calculated mean of fitness
        final_dupes_idx = max(enumerate(avr))[0]  # Extract the index of the greatest mean of fitness
        max_mean_fitness = max(enumerate(avr))[1]  # Max mean of the fitness
        idx_dupes = idx_dupes[final_dupes_idx]  # Final idx_dupes: contain only the most repeated params with highest mean value
        # Selected best params is the one with max obj
        max_idx = idx_dupes[
            [best_fitness_list[l] for l in idx_dupes].index(max([best_fitness_list[l] for l in idx_dupes]))]
        best_params = best_params_list[max_idx]
    return best_params, max_mean_fitness 

def stepwise_search(individual, step):
    x = [individual[0], individual[1], individual[2], individual[3], individual[4], individual[5]]
    new_individual = [individual[0], individual[1], individual[2], individual[3], individual[4], individual[5]]
    for i in step:
        if i == 1:
            step1, obj1 = find_hyperpa(x, findings='step_1')
            print('Params step 1:', step1, 'Obj1:', obj1)
            # Update individual
            new_individual[0], new_individual[1] = step1[0], step1[1]
            x[0], x[1] = step1[0], step1[1]
            new_obj = obj1
        if i == 2:
            step2, obj2 = find_hyperpa(x, findings='step_2')
            print('Params step 2:', step2, 'Obj2:', obj2)
            # Update individual
            new_individual[2], new_individual[3] = step2[0], step2[1]
            x[2], x[3] = step2[0], step2[1]
            new_obj = obj2
        if i == 3:
            step3, obj3 = find_hyperpa(x, findings='step_3')
            print('Params step 3:', step3, 'Obj3:', obj3)
            # Update individual
            new_individual[4], new_individual[5] = step3[0], step3[1]
            new_obj = obj3

    return new_individual, new_obj

stored_fitness = []
ini_params = [0.0015, 21, 0, 11, 0.15, 0.0011]  # Initial params
step = [1, 2, 3]
new_params, new_fitness = stepwise_search(ini_params, step)
prev_params = ini_params
i = 0
print(i, '.Prev params:', prev_params)
print(i, '.New params:', new_params, 'New fitness:', new_fitness)
print('Current running time: {} seconds'.format(np.round(time.time() - start_time, 1)))
i = 1
stored_fitness.append(new_fitness)
while new_params != prev_params:
    curr_step = [None] * 3
    for n in range(len(new_params)):
        if new_params[n] != prev_params[n]:
            if n == 0 or n == 1:
                curr_step[0] = 1
            if n == 2 or n == 3:
                curr_step[1] = 2
            if n == 4 or n == 5:
                curr_step[2] = 3

    prev_params = new_params
    prev_fitness = new_fitness
    new_params, new_fitness = stepwise_search(prev_params, curr_step)
    stored_fitness.append(new_fitness)
    print(i, '.Prev params:', prev_params, '--Prev fitness:', prev_fitness)
    print(i, '.New params:', new_params, '--New fitness:', new_fitness)
    print('Current running time: {} seconds'.format(np.round(time.time() - start_time, 1)))
    if new_params == prev_params:
        final_params = new_params
        final_fitness = new_fitness
        print(i, '.Optimal params:', final_params, '--Optimal fitness:', final_fitness)
        break
    else:
        print('======= Update params =======>')
        pass
    i += 1
print('Running time: {} seconds'.format(np.round(time.time() - start_time, 1)))
print('Stored fitness:', stored_fitness)
