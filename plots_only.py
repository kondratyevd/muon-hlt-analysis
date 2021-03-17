import os, sys
import argparse
import time

import dask
from dask.distributed import Client
import logging
logger = logging.getLogger("distributed.utils_perf")
logger.setLevel(logging.ERROR)


from datasets.datasets_phase2_pu140_vh import datasets
from tools import build_dataset, preprocess, plot

from utils import *


def plot_efficiencies(df, only_default_and_best=False):
    #x_opts = ['eta', 'pt']
    x_opts = ['eta', 'pt', 'nVtx']#, 'validHits']
    bin_opts = {
        'eta': regular(20, -2.4, 2.4),
        'pt': np.array([5, 7, 9, 12, 16, 20, 24, 27, 30, 35, 40, 45, 50, 60, 70, 90, 150]),#regular(50, 0, 100),
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
            ylabel='Efficiency'
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
        )


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dask", action='store_true')
    args = parser.parse_args()

    tick = time.time()
    datasets_ = [
        {
            'name':'1HB',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_noPU_L1TkMu_fromArnab_1HB/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_noPU_L1TkMu_fromArnab_1HB/210310_154141/0000/muon*root'
        },
        {
            'name':'1HL',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_noPU_L1TkMu_fromArnab_1HL/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_noPU_L1TkMu_fromArnab_1HL/210310_153340/0000/muon*root'
        },
        {
            'name':'2HL',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_noPU_L1TkMu_fromArnab_2HL/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_noPU_L1TkMu_fromArnab_2HL/210310_153434/0000/muon*root'
        },
        {
            'name':'3HL',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_noPU_L1TkMu_fromArnab_3HL/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_noPU_L1TkMu_fromArnab_3HL/210310_153515/0000/muon*root'
        },
        {
            'name':'4HL',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_noPU_L1TkMu_fromArnab_4HL/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_noPU_L1TkMu_fromArnab_4HL/210310_153557/0000/muon*root'
        },
        {
            'name':'5HL',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_noPU_L1TkMu_fromArnab_5HL/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_noPU_L1TkMu_fromArnab_5HL/210310_153651/0000/muon*root'
        },

        {
            'name':'default',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_noPU_L1TkMu_fromArnab_default/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_noPU_L1TkMu_fromArnab_default/210310_153743/0000/muon*root'
        },
    ]
    
    datasets = [
        #{
        #    'name':'PU 140: OI from L2',
        #    'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_OIFromL2_default/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_OIFromL2_default/210317_010510/0000/muon*root',
        #},
        #{
        #    'name':'PU 140: OI from L1TkMu',
        #    'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_OIFromL1TkMu_default/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_OIFromL1TkMu_default/210317_010703/0000/muon*root',
        #    'collection': 'L1Tkmuons'
        #},
        #{
        #    'name':'PU 140: OI from L2 - VectorHits',
        #    'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_OIFromL2_default_VHenabled/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_OIFromL2_default_VHenabled/210317_015040/0000/muon*root'
        #},
        {
            'name':'PU 140: OI from L1TkMu - VectorHits',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_OIFromL1TkMu_default_VHenabled/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_OIFromL1TkMu_default_VHenabled/210317_052649//0000/muon*root',
            'collection': 'L1Tkmuons'
        },

        #{
        #    'name':'no PU: OI from L2',
        #    'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_noPU_OIFromL2_default/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_noPU_OIFromL2_default/210317_033103/0000/muon*root',
        #},
        #{
        #    'name':'no PU: OI from L1TkMu',
        #    'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_noPU_OIFromL1TkMu_default/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_noPU_OIFromL1TkMu_default/210317_050826/0000/muon*root',
        #    'collection': 'L1Tkmuons'
        #},
        #{
        #    'name':'no PU: OI from L2 - VectorHits',
        #    'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_noPU_OIFromL2_default_VHenabled/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_noPU_OIFromL2_default_VHenabled/210317_033132/0000/muon*root'
        #},
        {
            'name':'no PU: OI from L1TkMu - VectorHits',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_noPU_OIFromL1TkMu_default_VHenabled/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_noPU_OIFromL1TkMu_default_VHenabled/210317_050855/0000/muon*root',
            'collection': 'L1Tkmuons'
        },

    ]
    
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