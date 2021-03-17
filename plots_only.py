import os, sys
import argparse
import time

import dask
from dask.distributed import Client
import logging
logger = logging.getLogger("distributed.utils_perf")
logger.setLevel(logging.ERROR)

from datasets.datasets_phase2_DY_L2vsL1TkMu import datasets

from tools import *
from utils import *

OUT_PATH = '/home/dkondra/muon-hlt-analysis/plots/'


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
    
    df = preprocess(rets, only_overlap_events=False)
    print(df)

    #plot_distributions(df, ['eta'], prefix='seed')

    plot_efficiencies(df, out_path=OUT_PATH, ymin=0.4)

    tock = time.time()
    print(f'Completed in {tock-tick} s.')