import os, sys
import argparse
import time

import dask
from dask.distributed import Client
import logging
logger = logging.getLogger("distributed.utils_perf")
logger.setLevel(logging.ERROR)

from datasets.datasets_run2_DY_may5 import datasets, datasets_dict
#from datasets.datasets_run3_may18 import datasets, datasets_dict
#from datasets.datasets_run2_DY_1to7seeds import datasets, datasets_dict
#from datasets.datasets_phase2_DY_apr14 import datasets
#from datasets.datasets_phase2_DY_apr14_forDNN import datasets
#from datasets.datasets_phase2_DY_L2vsL1TkMu import datasets
#from datasets.datasets_run2_VariousMC_5seeds import datasets
#from datasets.datasets_phase2_DYPU140_1seed_OIFromL2_VH import datasets

from tools import *
from utils import *

OUT_PATH = '/home/dkondra/muon-hlt-analysis/plots/run3_may24/'


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dask", action='store_true')
    args = parser.parse_args()

    tick = time.time()

    """
    datasets = [
        #{
        #'name': 'default Run2 DY - pixel OFF',
        #'path': '/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_DYJets_default/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/muonHLTtest_Run2_DYJets_default/210325_210356/0000/m*root'
        #'path': '/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_DYJets_default_pixelOFF/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/muonHLTtest_Run2_DYJets_default_pixelOFF/210402_174842/0000/m*root',
        #'reference': 'hltTrackOI'
        #},
        {
        'name': 'default Run2 DY - pixel ON',
        'path': '/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_DYJets_default_pixelON/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/muonHLTtest_Run2_DYJets_default_pixelON/210405_204336/0000/m*root',
        'reference': 'hltTrackOI'
        },
    ]
    """
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
    #covMat = [c for c in df.columns if 'covMat' in c]
    #print(df[covMat].mean())
    """
    cuts = {
        'total': True,
        'barrel low pT': (df.pt<10)&(abs(df.eta)<0.6),
        'barrel medium pT': (df.pt>10)&(df.pt<20)&(abs(df.eta)<0.6),
        'barrel high pT': (df.pt>20)&(abs(df.eta)<0.6),
        'overlap low pT': (df.pt<10)&(abs(df.eta)>0.6)&(abs(df.eta)<1.6),
        'overlap medium pT': (df.pt>10)&(df.pt<20)&(abs(df.eta)>0.6)&(abs(df.eta)<1.6),
        'overlap high pT': (df.pt>20)&(abs(df.eta)>0.6)&(abs(df.eta)<1.6),
        
        'endcap1 low pT': (df.pt<10)&(abs(df.eta)>1.6)&(abs(df.eta)<2.2),
        'endcap1 medium pT': (df.pt>10)&(df.pt<20)&(abs(df.eta)>1.6)&(abs(df.eta)<2.2),
        'endcap1 high pT': (df.pt>20)&(abs(df.eta)>1.6)&(abs(df.eta)<2.2),
        
        'endcap2 low pT': (df.pt<10)&(abs(df.eta)>2.2),
        'endcap2 medium pT': (df.pt>10)&(df.pt<20)&(abs(df.eta)>2.2),
        'endcap2 high pT': (df.pt>20)&(abs(df.eta)>2.2),
    }
    for cname, cut in cuts.items():
        #print(cname)
        val_33 = df.loc[cut&(df['covMat_33']>-999), 'covMat_33'].median()
        val_34 = df.loc[cut&(df['covMat_34']>-999), 'covMat_34'].median()
        val_44 = df.loc[cut&(df['covMat_44']>-999), 'covMat_44'].median()
        #print(f"  x error2: {val_33:.3e}  y error2: {val_44:.3e}  x-y cov: {val_34:.3e}")
        #print()
        #for j in range(5):
        #    line = ''
        #    for i in range(5):
        #        name = f'covMat_{i}{j}'
        #        val = df.loc[(df.pt>20)&(df.pt<50)&cut&(df[name]>-999), name].mean()
        #        line += f'{val:.2e}   '
        #    print(line)
    plot_distributions(df[abs(df.eta)<2.4], ['covMat_33', 'covMat_44'], prefix='cov', out_path=OUT_PATH)
    #plot_distributions(df[abs(df.eta)<2.4], ['covMat_34'], prefix='cov', out_path=OUT_PATH)
    sys.exit()
    """
    
    #plot_distributions(df, ['eta'], prefix='seed')

    plot_efficiencies(df, out_path=OUT_PATH, ymin=0.85)

    """
    eff1hb = np.array([ 0.8292512246326103, 0.827300930713547, 0.8068910256410257, 0.811831789023521, 0.8092732653732325, 0.7976841428111933, 0.7857142857142857, 0.7869803416048985, 0.7767118756202448, 0.7704799711295561, 0.7685554668794893, 0.7693014705882353, 0.7543859649122807, 0.7703045685279187, 0.7520976353928299, 0.7746341463414634, 0.7366412213740458, 0.7689463955637708, 0.7458823529411764, 0.7412140575079872 ])
    eff1hbd = np.array([ 0.9223233030090973, 0.9177869700103413, 0.9058493589743589, 0.8977191732002852, 0.898059848733969, 0.8906400771952396, 0.8888532991672005, 0.8884950048340315, 0.8746278531260337, 0.8657524359437027, 0.8723064644852354, 0.8639705882352942, 0.8745348219032429, 0.8616751269035533, 0.8688024408848207, 0.8829268292682927, 0.851145038167939, 0.878003696857671, 0.84, 0.8690095846645367 ])
    ratio = eff1hbd/eff1hb
    data = {
        'values': {'1 HB doublet / 1 HB': ratio},
        'edges': regular(20, 20, 60),
        'xlabel': 'nVtx'
    }
    plot(
         data,
         out_name='PUratio',
         ymin=1,
         ymax=1.4,
         ylabel='Efficiency ratio',
         out_path=OUT_PATH,
         add_text=True
    )
    """
    
    tock = time.time()
    print(f'Completed in {tock-tick} s.')