import os, sys
import argparse
import time

#from datasets.datasets_run2_DY_1to7seeds import datasets, datasets_dict
#from datasets.datasets_run2_VariousMC_5seeds import datasets, datasets_dict
#from datasets.datasets_run2_first_studies import datasets
from datasets.datasets_phase2_DYPU140_5seeds_OIFromL2 import datasets, datasets_dict
#datasets_dict = {ds['name']:[ds] for ds in datasets}

from tqdm import tqdm
import glob
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

def train(df, **kwargs):
    save = kwargs.pop('save', True)
    model_path = kwargs.pop('model_path', './')
    opt_label = kwargs.pop('opt_label', 'test')
    plot_loss = kwargs.pop('plot_loss', False)

    features = l2_branches # imported from tools
    truth_columns = []
    prediction_columns = []
    for c in df.columns:
        if ('pass' in c):
            truth_columns.append(c)
            prediction_columns.append(c.replace('pass', 'pred'))
    for c in prediction_columns:
        df[c] = -1
    if len(truth_columns)==1:
        return df, truth_columns[0]
    elif len(truth_columns)<1:
        return df, None

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
        if plot_loss:
            plot_loss(history, label, OUT_PATH)
        prediction = pd.DataFrame(dnn.predict(x_eval))
        df.loc[eval_filter, prediction_columns] = prediction.values
    print(prediction_columns)
    df['best_guess_label'] = df[prediction_columns].idxmax(axis=1)
    df['best_guess_label'] = df.best_guess_label.str.replace('pred: ', 'pass: ')
    pred_label = f'DNN recommendation {opt_label}'
    df[pred_label] = df.lookup(df.index, df.best_guess_label)
    return df, pred_label

        
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
            memory_limit='2GB'
        )
        rets = client.gather(
            client.map(build_dataset, datasets_)
        )
    else:
        rets = []
        for ds in datasets_:
            rets.append(build_dataset(ds, progress_bar=True))
    
    df = preprocess(rets, only_overlap_events=True)
    print(df)
    pred_col_names = None
    if args.train:
        pars = {
            'save': True,
            'model_path': MODEL_PATH,
            'opt_label': label,
            'plot_loss': False
        }
        df, pred_col_name = train(df, **pars)
        if pred_col_name is not None:
            pred_col_names = [pred_col_name]
    
    pars = {
        'cols_to_plot': pred_col_names,
        'opt_label': label,
        'draw': False,
        'out_path': OUT_PATH,
        'ymin': 0.8,
        'prefix': 'DNN_5seeds_'
    }
    eff_plots = plot_efficiencies(df, **pars)
    
    if args.train and label!='default':
        plot_strategies(df, out_path=OUT_PATH)
    return eff_plots

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dask", action='store_true')
    parser.add_argument("-t", "--train", action='store_true')
    args = parser.parse_args()

    tick = time.time()
    
    #datasets_dict = {k:v for k,v in datasets_dict.items() if ((k!='2 seeds')and(k!='7 seeds'))}
    #datasets_dict = {k:v for k,v in datasets_dict.items() if ((k!='1 seed'))}

    #run_training(args=args, datasets=datasets)

    if len(datasets_dict)==0:
        print("Nothing to process")
        sys.exit()

    if True:
        eff_plots = {}
        for label, ds in datasets_dict.items():
            print(f"Training for {label}")
            eff_plots[label] = run_training(args=args, datasets=ds, label=label)

        x_opts = ['eta', 'pt', 'nVtx']
        eff_plots = transform_plots(eff_plots, x_opts)
        for x in x_opts:
            plot(
                eff_plots[x],
                out_name=eff_plots[x]['out_name'],
                ymin=0.7,
                ymax=1.01,
                ylabel='Efficiency',
                out_path=OUT_PATH,
                add_text=True
            )

    tock = time.time()
    print(f'Completed in {tock-tick} s.')


