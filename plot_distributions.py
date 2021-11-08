import os, sys
import argparse
import time

import dask
from dask.distributed import Client
import logging
logger = logging.getLogger("distributed.utils_perf")
logger.setLevel(logging.ERROR)

#from datasets.datasets_run2_DY_may5 import datasets, datasets_dict
#from datasets.datasets_run3_final_tests import datasets, datasets_dict
#from datasets.datasets_test_jun24 import datasets, datasets_dict
#from datasets.datasets_layers import datasets, datasets_dict
#from datasets.datasets_run3_may18 import datasets, datasets_dict
#from datasets.datasets_run2_DY_1to7seeds import datasets, datasets_dict
#from datasets.datasets_phase2_DY_apr14 import datasets
#from datasets.datasets_phase2_DY_apr14_forDNN import datasets
#from datasets.datasets_phase2_DY_L2vsL1TkMu import datasets
#from datasets.datasets_run2_VariousMC_5seeds import datasets
#from datasets.datasets_phase2_DYPU140_1seed_OIFromL2_VH import datasets

from tools import *
from utils import *

OUT_PATH = '/home/dkondra/muon-hlt-analysis/plots/run3_aug21/'


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dask", action='store_true')
    args = parser.parse_args()

    tick = time.time()

    datasets = [
        {
            "name": "muonHLTtest, Run2, DYJets, 0HB(d), 1HL(IP), 0HL(MuS)", 
            "path": "/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_DYJets_0HBd_1HLIP_0HLMuS/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/muonHLTtest_Run2_DYJets_0HBd_1HLIP_0HLMuS//210216_204601/0000/muonNtuple_test_MC_*.root"
        },
        {
            "name": "muonHLTtest, Run2, DYJets, 0HB(d), 0HL(IP), 1HL(MuS)", 
            "path": "/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_DYJets_0HBd_0HLIP_1HLMuS/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/muonHLTtest_Run2_DYJets_0HBd_0HLIP_1HLMuS//210216_204549/0000/muonNtuple_test_MC_*.root"
        },
        {
            "name": "muonHLTtest, Run2, DYJets, 1HB(d), 0HL(IP), 0HL(MuS)", 
            "path": "/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_DYJets_1HBd_0HLIP_0HLMuS/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/muonHLTtest_Run2_DYJets_1HBd_0HLIP_0HLMuS//210216_204613/0000/muonNtuple_test_MC_*.root"
        }
    ]
    datasets_dict = {}
    # Get data for efficiency plots
    if args.dask:
        n = min(23, len(datasets))
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
    passHLIP = 'pass: muonHLTtest, Run2, DYJets, 0HB(d), 1HL(IP), 0HL(MuS)'
    passHLMuS = 'pass: muonHLTtest, Run2, DYJets, 0HB(d), 0HL(IP), 1HL(MuS)'
    passHBd = 'pass: muonHLTtest, Run2, DYJets, 1HB(d), 0HL(IP), 0HL(MuS)'

    cuts = {
        'pass 1HL(IP)': df[passHLIP]==1,
        #'fail 1HL(IP)': df[passL3]==0,
        'pass 1HL(MuS)': df[passHLMuS]==1,
        'pass 1HB(d)': df[passHBd]==1,
    }
    plot_distributions(df, cuts, l2_branches, prefix='distribution', out_path=OUT_PATH)
    sys.exit()

    
    tock = time.time()
    print(f'Completed in {tock-tick} s.')
