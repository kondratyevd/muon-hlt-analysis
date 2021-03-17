import os, sys
import argparse
import time

import dask
from dask.distributed import Client
import logging
logger = logging.getLogger("distributed.utils_perf")
logger.setLevel(logging.ERROR)

from datasets.datasets_phase2_DY_L2vsL1TkMu import datasets

from tools import build_dataset, preprocess, plot

from utils import *

OUT_PATH = '/home/dkondra/muon-hlt-analysis/plots/'


def plot_efficiencies(df, only_default_and_best=False):
    #x_opts = ['eta', 'pt']
    x_opts = ['eta', 'pt', 'nVtx']#, 'validHits']
    bin_opts = {
        'eta': regular(20, -2.4, 2.4),
        'pt': np.array([5, 7, 9, 12, 16, 20, 24, 27, 30, 35, 40, 45, 50, 60, 70, 90, 150]),
        'nVtx': regular(50, 0, 200),
        'validHits': regular(30, 0, 60),
    }
    strategies = []
    for c in df.columns:
        if ('pass' in c):
            strategies.append(c)
    for x in x_opts:
        out_name = f'eff_vs_{x}'
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
                cut = (df[x] >= bin_min) & (df[x] < bin_max) & (df[s] >= 0)
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
        }
        plot(
            data,
            out_name=out_name,
            ymin=0.,
            ymax=1.01,
            ylabel='Efficiency',
            out_path=OUT_PATH
        )


def plot_distributions(df, variables, prefix=''):
    for v in variables:
        if 'eta' in v:
            bins = regular(100, df[v].min(), df[v].max())
            values, bins = np.histogram(df[v], bins)
        else:
            bins = regular(100, df.eta.min(), df.eta.max())
            values = []
            for ibin in range(len(bins)-1):
                    bin_min = bins[ibin]
                    bin_max = bins[ibin+1]
                    cut = (df.eta >= bin_min) & (df.eta < bin_max)
                    value = df.loc[cut, v].mean()
                    values = np.append(values, value)
        

        data = {
            'xlabel': 'eta',
            'edges': bins,
            'values': {'': values},
        }
        plot(
            data,
            out_name=f'{prefix}_{v}',
            histtype='step',
            title=f'Avg. {v}',
            out_path=OUT_PATH
        )


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dask", action='store_true')
    args = parser.parse_args()

    tick = time.time()

    # Get data for efficiency plots
    if args.dask:
        n = 8
        print(f'Using Dask with {n} local workers')
        client = dask.distributed.Client(
            processes=True,
            n_workers=n,
            threads_per_worker=1,
            memory_limit='8GB'
        )
        rets = client.gather(
            client.map(build_dataset, datasets)
        )
    else:
        rets = []
        for ds in datasets:
            rets.append(build_dataset(ds, progress_bar=True))
    
    df = preprocess(rets, overlap_events=False)
    print(df)

    #plot_distributions(df, ['eta'], prefix='seed')

    plot_efficiencies(df, only_default_and_best=False)

    tock = time.time()
    print(f'Completed in {tock-tick} s.')