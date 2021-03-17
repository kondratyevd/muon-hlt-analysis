import os, sys
import argparse
import time

#from datasets.datasets_run2_1to7seeds import datasets, datasets_dict
from datasets.datasets_different import datasets, datasets_dict

from tqdm import tqdm
import glob
import uproot as uproot
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import awkward as ak
import dask
from dask.distributed import Client
import logging
logger = logging.getLogger("distributed.utils_perf")
logger.setLevel(logging.ERROR)

from utils import *
from tools import *

import tensorflow
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import BatchNormalization

import cmsml

OUT_PATH = '/home/dkondra/muon-hlt-analysis/plots/'
MODEL_PATH = '/home/dkondra/muon-hlt-analysis/models/'

def train(df, save=False, model_path='', opt_label=''):
    ignore_lbl = 'default'
    features = l2_branches
    truth_columns = []
    prediction_columns = []
    for c in df.columns:
        if ('pass' in c) and (c!='pass: default (Run 2)') and (c!='pass: default, dynamic SF (Run 2)'):
            truth_columns.append(c)
            prediction_columns.append(c.replace('pass', 'pred'))
    for c in prediction_columns:
        df[c] = -1
    if len(prediction_columns)<2:
        return df
    nfolds = 4
    for i in range(nfolds):
        label = f'dnn_{opt_label}_{i}'.replace(' ', '_')
        train_folds = [(i + f) % nfolds for f in [0, 1]]
        val_folds = [(i + f) % nfolds for f in [2]]
        eval_folds = [(i + f) % nfolds for f in [3]]

        print(f"Train classifier #{i + 1} out of {nfolds}")
        print(f"Training folds: {train_folds}")
        print(f"Validation folds: {val_folds}")
        print(f"Evaluation folds: {eval_folds}")

        train_filter = df.event.mod(nfolds).isin(train_folds)
        val_filter = df.event.mod(nfolds).isin(val_folds)
        eval_filter = df.event.mod(nfolds).isin(eval_folds)

        df_train = df[train_filter]
        df_val = df[val_filter]
        df_eval = df[eval_filter]

        x_train = df_train[features]
        y_train = df_train[truth_columns]
        x_val = df_val[features]
        y_val = df_val[truth_columns]
        x_eval = df_eval[features]
        y_eval = df_eval[truth_columns]
        
        input_dim = len(features)
        output_dim = len(truth_columns)
        inputs = Input(shape=(input_dim,), name=label+'_input')
        x = Dense(128, name=label+'_layer_1', activation='tanh')(inputs)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(64, name=label+'_layer_2', activation='tanh')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(32, name=label+'_layer_3', activation='tanh')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        outputs = Dense(output_dim, name=label+'_output',  activation='sigmoid')(x)

        dnn = Model(inputs=inputs, outputs=outputs)
        dnn.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=["accuracy"])
        dnn.summary()

        history = dnn.fit(
            x_train[features],
            y_train,
            epochs=100,
            batch_size=256,
            verbose=0,
            validation_data=(x_val[features], y_val),
            shuffle=True
        )
        if save:
            save_path = f"{model_path}/{label}.pb"
            print(f'Saving model to {save_path}')
            cmsml.tensorflow.save_graph(save_path, dnn, variables_to_constants=True)
            cmsml.tensorflow.save_graph(save_path+'.txt', dnn, variables_to_constants=True)
        plot_loss(history, label)
        prediction = pd.DataFrame(dnn.predict(x_eval))
        df.loc[eval_filter, prediction_columns] = prediction.values
    print(prediction_columns)
    df['best_guess_label'] = df[prediction_columns].idxmax(axis=1)
    df['best_guess_label'] = df.best_guess_label.str.replace('pred: ', 'pass: ')
    df[f'DNN recommendation {opt_label}'] = df.lookup(df.index, df.best_guess_label)
    return df


def plot_loss(history, label):
    fig = plt.figure()
    fig.clf()
    plt.rcParams.update({'font.size': 10})
    fig.set_size_inches(5, 4)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    out = f'{OUT_PATH}/loss_{label}'
    fig.savefig(out)
    print(f'Saved loss plot: {out}')

def plot_efficiencies(df, only_default_and_best=False, opt_label='', save=True):
    x_opts = ['eta', 'pt', 'nVtx', 'validHits']
    bin_opts = {
        'eta': regular(20, -2.4, 2.4),
        'pt': np.array([5, 7, 9, 12, 16, 20, 24, 27, 30, 35, 40, 45, 50, 60, 70, 90, 150]),,
        'nVtx': regular(50, 0, 100),
        'validHits': regular(30, 0, 60),
    }
    strategies = []
    for c in df.columns:
        if ('pass' in c):
            if not (only_default_and_best and ((c!='pass: default (Run 2)') and  (c!='pass: default, dynamic SF (Run 2)'))):
                strategies.append(c)
    strategies.append(f'DNN recommendation {opt_label}')
    plotting_data = {}
    for x in x_opts:
        plotting_data[x] = {}
        out_name = f'DNN_eff_vs_{x}'
        bins = bin_opts[x]
        values = {}
        errors_lo = {}
        errors_hi = {}
        for s in strategies:
            if s not in df.columns:
                continue
            values[s] = np.array([])
            errors_lo[s] = np.array([])
            errors_hi[s] = np.array([])
            for ibin in range(len(bins)-1):
                bin_min = bins[ibin]
                bin_max = bins[ibin+1]
                cut = (df[x] >= bin_min) & (df[x] < bin_max)
                total = df.loc[cut, s].shape[0]
                passed = df.loc[cut, s].sum().sum()
                if total == 0:
                    value = err_lo = err_hi = 0
                else:
                    value = passed / total

                # Clopper-Pearson errors
                lo, hi = clopper_pearson(total, passed, 0.327)
                err_lo = value - lo
                err_hi = hi - value

                values[s] = np.append(values[s], value)
                errors_lo[s] = np.append(errors_lo[s], err_lo)
                errors_hi[s] = np.append(errors_hi[s], err_hi)
        data = {
            'xlabel': x,
            'edges': bins,
            'values': values,
            'errors_lo': errors_lo,
            'errors_hi': errors_hi,
            'out_name': out_name,
        }
        plotting_data[x] = data
        if save:
            plot(
                data,
                out_name=out_name,
                ymin=0,
                ymax=1.01,
                ylabel='Efficiency',
                out_path=OUT_PATH
            )
    return plotting_data


def plot_strategies(df):
    ignore = 'default'
    x_opts = l2_branches
    bin_opts = {
        'eta': regular(20, -2.4, 2.4),
        'pt': np.array([5, 7, 9, 12, 16, 20, 24, 27, 30, 35, 40, 45, 50, 60, 70, 90, 150]),,
#        'nVtx': regular(10, 0, 10),
        'validHits': regular(30, 0, 60),
        'chi2': regular(50, 0, 5),
        'err0_IP': regular(50, 0, 0.015),
        'err1_IP': regular(50, 0, 0.02),
        'err2_IP': regular(50, 0, 0.05),
        'err3_IP': regular(50, 0.099, 0.1),
        'err4_IP': regular(50, 0, 5),
        'tsos_IP_pt': regular(50, 0, 100),
        'tsos_IP_eta': regular(50, -2.4, 2.4),
        'tsos_IP_pt_eta': regular(50, -2.4, 2.4),
        'err0_MuS': regular(50, 0, 0.015),
        'err1_MuS': regular(50, 0, 0.05),
        'err2_MuS': regular(50, 0, 0.25),
        'err3_MuS': regular(50, 0, 10),
        'err4_MuS': regular(50, 0, 15),
        'tsos_MuS_pt': regular(50, 0, 100),
        'tsos_MuS_eta': regular(50, -2.4, 2.4),
        'tsos_MuS_pt_eta': regular(50, -2.4, 2.4),
    }
    strategies = []
    for c in df.columns:
        if ('pass' in c) and (c!='pass: default (Run 2)') and (c!='pass: default, dynamic SF (Run 2)'):
            strategies.append(c)
    #strategies.append('DNN recommendation')
    for x in x_opts:
        out_name = f'DNN_strategy_vs_{x}'
        if x in bin_opts.keys():
            bins = bin_opts[x]
        elif x in df.columns:
            bins = regular(50, df.loc[df[x]>-999., x].min(), df[x].max())
        else:
            continue
        values = {}
        errors_lo = {}
        errors_hi = {}
        for s in strategies:
            values[s] = np.array([])
            errors_lo[s] = np.array([])
            errors_hi[s] = np.array([])
            for ibin in range(len(bins)-1):
                bin_min = bins[ibin]
                bin_max = bins[ibin+1]
                cut = (df[x] >= bin_min) & (df[x] < bin_max) & (df.best_guess_label==s)
                value = df[cut].shape[0]
                values[s] = np.append(values[s], value)
        data = {
            'xlabel': x,
            'edges': bins,
            'values': values,
        }
        plot(
            data,
            out_name=out_name,
            ylabel='',
            out_path=OUT_PATH
        )

        
def transform_plots(dicts, x_opts):
    out = {}
    for x in x_opts:
        out[x] = {
            'values': {},
            'errors_lo': {},
            'errors_hi': {},
        }
    for label, plots in dicts.items():
        for x, data in plots.items():
            out[x]['xlabel'] = data['xlabel']
            out[x]['edges'] = data['edges']
            out[x]['out_name'] = data['out_name']
            out[x]['values'].update(data['values'])
            out[x]['errors_lo'].update(data['errors_lo'])
            out[x]['errors_hi'].update(data['errors_hi'])
    return out

def run_training(**kwargs):
    args = kwargs.pop('args', None)
    datasets_ = kwargs.pop('datasets', datasets)
    label = kwargs.pop('label', 'test')
    # Get data for efficiency plots
    if args.dask:
        n = 22
        print(f'Using Dask with {n} local workers')
        client = dask.distributed.Client(
            processes=True,
            n_workers=n,
            threads_per_worker=1,
            memory_limit='1GB'
        )
        rets = client.gather(
            client.map(build_dataset, datasets_)
        )
    else:
        rets = []
        for ds in datasets_:
            rets.append(build_dataset(ds, progress_bar=True))
    
    df = preprocess(rets)
    print(df)
    if args.train:
        df = train(df, save=True, model_path=MODEL_PATH, opt_label=label)
    
    eff_plots = plot_efficiencies(df, only_default_and_best=True, opt_label=label, save=False)
    
#    if args.train:
#        plot_strategies(df)
    return eff_plots

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dask", action='store_true')
    parser.add_argument("-t", "--train", action='store_true')
    args = parser.parse_args()

    tick = time.time()

    #run_training(args=args, datasets=datasets)

    if True:
        eff_plots = {}
        for label, ds in datasets_dict.items():
            print(f"Training for {label}")
            eff_plots[label] = run_training(args=args, datasets=ds, label=label)
    
            x_opts = ['eta', 'pt', 'nVtx', 'validHits']
        eff_plots = transform_plots(eff_plots, x_opts)
        for x in x_opts:
            plot(
                eff_plots[x],
                out_name=eff_plots[x]['out_name'],
                ymin=0,
                ymax=1.01,
                ylabel='Efficiency',
                out_path=OUT_PATH
            )

    tock = time.time()
    print(f'Completed in {tock-tick} s.')


