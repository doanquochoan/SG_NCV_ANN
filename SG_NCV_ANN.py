import keras
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation, LeakyReLU
from RC_panel_data_new import split_fold, filepath, X, Y, seed_value
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
import numpy as np
import matplotlib
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from mlxtend.plotting import plot_confusion_matrix
import os
from sklearn.metrics import roc_curve, auc
from numpy import interp
from itertools import cycle
import time
import random
import tensorflow as tf
start_time = time.time()

##################################################
os.chdir(os.getcwd())
seed_value = seed_value
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
random.seed(seed_value)  # 2. Set `python` built-in pseudo-random generator at a fixed value
np.random.seed(seed_value)  # 3. Set `numpy` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)  # 4. Set `tensorflow` pseudo-random generator at a fixed value
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1) # 5. Configure a new global `tensorflow` session
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

out_kfold = StratifiedKFold(n_splits=split_fold, shuffle=True, random_state=seed_value)
in_kfold = StratifiedKFold(n_splits=split_fold - 1, shuffle=True, random_state=seed_value)

#####################Optimization phase######################
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

#####################Testing phase######################
best_individual = final_params
lr = best_individual[0]
batch_size = best_individual[1]
hidden_layers = best_individual[2]
neurons = best_individual[3]
dropout_rate = best_individual[4]
l2_rate = best_individual[5]

fig_save_times = 1
best_model_list = {}
all_best_params = []
test_acc = []
train_val_acc = []
test_f1_score = []
test_class_0, test_class_1, test_class_2, test_class_3 = [], [], [], []

ann_tprs = []
aucs = []
ann_mean_fpr = np.linspace(0, 1, 100)
q = 0

ann_tprs_class_0 = []
aucs_class_0 = []
ann_tprs_class_1 = []
aucs_class_1 = []
ann_tprs_class_2 = []
aucs_class_2 = []
ann_tprs_class_3 = []
aucs_class_3 = []

for i, (train_val_idx, test_idx) in enumerate(out_kfold.split(X, Y)):
    # ----Splitting train_val/test set-----
    X_train_val, X_test = X[train_val_idx], X[test_idx]
    Y_train_val, Y_test = Y[train_val_idx], Y[test_idx]
   
    df_Y = pd.DataFrame(Y)
    print(df_Y[0].value_counts())
    df_ytrain = pd.DataFrame(Y_train_val)
    df_ytest = pd.DataFrame(Y_test)
    print(df_ytrain[0].value_counts())
    print(df_ytest[0].value_counts())

    model_RC = create_model(lr=lr, neurons=neurons, hidden_layers=hidden_layers, dropout_rate=dropout_rate,
                            l2_rate=l2_rate, initializer='he_uniform', activation='relu')
    filepath_com = 'path'
    filepath_model = filepath_com + 'Loaded_model_{}_{}.h5'.format(fig_save_times, i + 1)
    filepath = filepath_com
    es = EarlyStopping(monitor='val_loss', mode='min', patience=80, verbose=0)
    mcp = ModelCheckpoint(filepath_model, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    callbacks = [es, mcp]
    
    Y_train_val = keras.utils.to_categorical(Y_train_val, num_classes=4)
    Y_test = keras.utils.to_categorical(Y_test, num_classes=4)
    history_RC = model_RC.fit(X_train_val, Y_train_val,
                              batch_size=batch_size,
                              epochs=800,
                              verbose=0,
                              validation_data=(X_test, Y_test),
                              callbacks=callbacks)

    saved_model = load_model(filepath_model)

    # #-------Cross-validation and plot ROC curves--------
    probs = saved_model.predict_proba(X_test)
    n_classes = Y_test.shape[1]    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for c in range(n_classes):
        fpr[c], tpr[c], _ = roc_curve(Y_test[:, c], probs[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])

    ann_tprs_class_0.append(interp(ann_mean_fpr, fpr[0], tpr[0]))
    ann_tprs_class_0[-1][0] = 0.0
    aucs_class_0.append(roc_auc[0])
    ann_tprs_class_1.append(interp(ann_mean_fpr, fpr[1], tpr[1]))
    ann_tprs_class_1[-1][0] = 0.0
    aucs_class_1.append(roc_auc[1])
    ann_tprs_class_2.append(interp(ann_mean_fpr, fpr[2], tpr[2]))
    ann_tprs_class_2[-1][0] = 0.0
    aucs_class_2.append(roc_auc[2])
    ann_tprs_class_3.append(interp(ann_mean_fpr, fpr[3], tpr[3]))
    ann_tprs_class_3[-1][0] = 0.0
    aucs_class_3.append(roc_auc[3])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(Y_test.ravel(), probs.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # fpr, tpr, thresholds = roc_curve(Y_test, probs[:, 1])
    ann_tprs.append(interp(ann_mean_fpr, fpr['micro'], tpr['micro']))
    ann_tprs[-1][0] = 0.0
    # roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc['micro'])
    plt.plot(fpr['micro'], tpr['micro'], 'k--', lw=3, alpha=1, color='black', linestyle='solid',
             label='Micro-average ROC fold {} (AUC = {:.2f})'.format(q, roc_auc['micro']))
    colors = cycle(['brown', 'darkorange', 'cornflowerblue', 'green'])
    for cr, color in zip(range(n_classes), colors):
        plt.plot(fpr[cr], tpr[cr], color=color, lw=2, alpha=0.5,
                 label='ROC curve of class {0} (area = {1:0.2f})'''.format(cr, roc_auc[cr]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2, color='red')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ANN model - ROC curve fold {}'.format(q + 1))
    plt.legend(loc="lower right")
    plt.savefig(filepath + "ANN model - ROC_curve_fold_{}.jpg".format(q + 1), dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()
    q += 1

    # #-------Print classification metrics--------
    Y_pred = saved_model.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)

    train_acc = saved_model.evaluate(X_train_val, Y_train_val, verbose=0)
    print("Train accuracy: {:.2f}%".format(train_acc[1] * 100))
    train_val_acc.append(float("{:.2f}".format(train_acc[1] * 100)))
    acc_test = accuracy_score(Y_test, Y_pred)
    print("Test accuracy: {:.2f}%".format(acc_test * 100))
    test_acc.append(float("{:.2f}".format(acc_test * 100)))
    f1 = f1_score(Y_test, Y_pred, average='micro')
    print("Micro F1-score: {:.3f}".format(f1))
    test_f1_score.append(float("{:.2f}".format(f1)))

    cm = confusion_matrix(Y_test, Y_pred)
    matplotlib.rcParams.update({'font.size': 18})
    fig, ax = plot_confusion_matrix(conf_mat=cm, colorbar=True, show_normed=True)
    plt.savefig(filepath + "4. test_cm {}_{}.jpg".format(fig_save_times, i + 1), dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    class_acc = cm.diagonal() / cm.sum(axis=1)
    print("Class acc: ", class_acc)
    print("Average accuracy of classes: {:.2f}%\n".format(np.mean(class_acc * 100).astype(float)))
    class_i0, class_i1, class_i2, class_i3 = class_acc[0], class_acc[1], class_acc[2], class_acc[3]
    test_class_0.append(float("{:.2f}".format(class_i0)))
    test_class_1.append(float("{:.2f}".format(class_i1)))
    test_class_2.append(float("{:.2f}".format(class_i2)))
    test_class_3.append(float("{:.2f}".format(class_i3)))

    plt.figure(1)
    plt.plot(history_RC.history["accuracy"])
    plt.plot(history_RC.history["val_accuracy"])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Model accuracy')
    plt.legend(['Train', 'Dev'], loc='upper left')
    plt.savefig(filepath + "1. acc {}_{}.jpg".format(fig_save_times, i + 1), dpi=600, bbox_inches='tight')
    plt.close()

    plt.figure(2)
    plt.plot(history_RC.history["loss"])
    plt.plot(history_RC.history["val_loss"])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Dev'], loc='upper right')
    plt.savefig(filepath + "2. loss {}_{}.jpg".format(fig_save_times, i + 1), dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

print("\n\nTrain acc     : {}".format(train_val_acc))
print("Test acc: {}".format(test_acc))
print("Average accuracy of train set: {:.2f}%\n Std: +/- {:.2f}%".format(np.mean(train_val_acc).astype(float),
                                                                         np.std(train_val_acc).astype(float)))
print("Average accuracy of test set: {:.2f}%\n Std: +/- {:.2f}%".format(np.mean(test_acc).astype(float),
                                                                        np.std(test_acc).astype(float)))
print("All micro f1-score: {}\nAverage micro f1-score: {:.3f}".format(test_f1_score, np.mean(test_f1_score)))
print("Class 0 average acc: {:.2f}%\n"
      "Class 1 average acc: {:.2f}%\n"
      "Class 2 average acc: {:.2f}%\n"
      "Class 3 average acc: {:.2f}%\n"
      "Average class accuracy: {:.2f}%".format(np.mean(test_class_0) * 100, np.mean(test_class_1) * 100,
                                               np.mean(test_class_2) * 100, np.mean(test_class_3) * 100,
                                               (np.mean(test_class_0) + np.mean(test_class_1) +
                                                np.mean(test_class_2) + np.mean(test_class_3)) / 4 * 100))

print('Running time: {} seconds'.format(np.round(time.time() - start_time, 1)))
# #-------------Plot ROC-AUC curve------------------------------------------------------
# ----Chance line----
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Chance', alpha=.7)
# # -----Mean curve-------
ann_mean_tpr = np.mean(ann_tprs, axis=0)
ann_mean_tpr[-1] = 1.0
ann_mean_auc = auc(ann_mean_fpr, ann_mean_tpr)
ann_std_auc = np.std(aucs)
plt.plot(ann_mean_fpr, ann_mean_tpr, color='black', label=r'Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'
         .format(ann_mean_auc, ann_std_auc), lw=3, alpha=.8, linestyle='solid')
# # -----Class 0-------
ann_mean_tpr_class_0 = np.mean(ann_tprs_class_0, axis=0)
ann_mean_tpr_class_0[-1] = 1.0
ann_mean_auc_class_0 = auc(ann_mean_fpr, ann_mean_tpr_class_0)
ann_std_auc_class_0 = np.std(aucs_class_0)
plt.plot(ann_mean_fpr, ann_mean_tpr_class_0, color='brown', label=r'ROC curve of class 0 (AUC = {:.2f} $\pm$ {:.2f})'
         .format(ann_mean_auc_class_0, ann_std_auc_class_0), lw=2, alpha=.6, linestyle=(0, (5, 1)))
# # -----Class 1-------
ann_mean_tpr_class_1 = np.mean(ann_tprs_class_1, axis=0)
ann_mean_tpr_class_1[-1] = 1.0
ann_mean_auc_class_1 = auc(ann_mean_fpr, ann_mean_tpr_class_1)
ann_std_auc_class_1 = np.std(aucs_class_1)
plt.plot(ann_mean_fpr, ann_mean_tpr_class_1, color='green', label=r'ROC curve of class 1 (AUC = {:.2f} $\pm$ {:.2f})'
         .format(ann_mean_auc_class_1, ann_std_auc_class_1), lw=2, alpha=.6, linestyle='dashdot')
# # -----Class 2-------
ann_mean_tpr_class_2 = np.mean(ann_tprs_class_2, axis=0)
ann_mean_tpr_class_2[-1] = 1.0
ann_mean_auc_class_2 = auc(ann_mean_fpr, ann_mean_tpr_class_2)
ann_std_auc_class_2 = np.std(aucs_class_2)
plt.plot(ann_mean_fpr, ann_mean_tpr_class_2, color='blue', label=r'ROC curve of class 2 (AUC = {:.2f} $\pm$ {:.2f})'
         .format(ann_mean_auc_class_2, ann_std_auc_class_2), lw=2, alpha=.6, linestyle='dotted')
# # -----Class 3-------
ann_mean_tpr_class_3 = np.mean(ann_tprs_class_3, axis=0)
ann_mean_tpr_class_3[-1] = 1.0
ann_mean_auc_class_3 = auc(ann_mean_fpr, ann_mean_tpr_class_3)
ann_std_auc_class_3 = np.std(aucs_class_3)
plt.plot(ann_mean_fpr, ann_mean_tpr_class_3, color='purple', label=r'ROC curve of class 3 (AUC = {:.2f} $\pm$ {:.2f})'
         .format(ann_mean_auc_class_3, ann_std_auc_class_3), lw=2, alpha=.6, linestyle=(0, (3, 1, 1, 1, 1, 1)))

plt.fill_between(ann_mean_fpr, ann_mean_tpr, color='grey', alpha=0.2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ANN model - Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.savefig(filepath + "ANN model - Mean_ROC_curve.jpg", dpi=600, bbox_inches='tight')
plt.show()
