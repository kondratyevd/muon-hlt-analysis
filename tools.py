from tqdm import tqdm
import glob
import uproot
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplhep as hep


from utils import *

l2_branches = [
    'pt',
    'eta',
    'phi',
#    'chi2',
    'validHits',
    'tsos_IP_eta',
    'tsos_IP_phi',
    'tsos_IP_pt',
    'tsos_IP_pt_eta',
    'tsos_IP_pt_phi',
    'err0_IP',
    'err1_IP',
    'err2_IP',
    'err3_IP',
    'err4_IP',
    'tsos_MuS_eta',
    'tsos_MuS_phi',
    'tsos_MuS_pt',
    'tsos_MuS_pt_eta',
    'tsos_MuS_pt_phi',
    'err0_MuS',
    'err1_MuS',
    'err2_MuS',
    'err3_MuS',
    'err4_MuS',
    'tsos_IP_valid',
    'tsos_MuS_valid',
]
l2_branches = [
    'pt',
    'eta',
    'phi',]
l3_branches = gen_branches = ['pt', 'eta', 'phi']

seed_branches = [
    #'pt', 'eta', 'phi', 'hitBased', 'eta_pos', 'layerNum',
    #'tsos_q_p_err', 'tsos_lambda_err', 'tsos_phi_err','tsos_xT_err','tsos_yT_err', 
    #'dR_pos', 'dR_mom'
]

def build_dataset(dataset, progress_bar=False, study_seeds=False):
    label = dataset['name']
    path = dataset['path']
    l2_name = dataset.pop('collection', 'L2muons')
    print(f'Processing {label}')
    all_l2s = MuCollection()
    all_l3s = MuCollection()
    all_seeds = MuCollection()
    loop = tqdm(glob.glob(path)) if progress_bar else glob.glob(path)
    i = 0
    for fname in loop:
        i+=1
        #if i==2:
        #    break
        tree = uproot.open(fname)['muonNtuples']['muonTree']['event']
        nVtx = tree['nVtx'].array()
        event = tree['eventNumber'].array()
        if len(event)==0:
            continue
        #l2_name = 'L2muons'
        #l2_name = 'L1Tkmuons'
        l3_name = 'hltOImuons'
        gen_name = 'genParticles'

        l2_muons = {}
        l3_muons = {}
        gen_muons = {}
        seeds = {}

        #l2_cut = (tree[f'{l2_name}.validHits'].array()>20)&(tree[f'{l2_name}.pt'].array()>10)
        #l2_cut = (tree[f'{l2_name}.pt'].array()>24)
        l2_cut = (tree[f'{l2_name}.pt'].array()>0)

        
        for branch in l2_branches:
            l2_muons[branch] = tree[f'{l2_name}.{branch}'].array()[l2_cut]
        for branch in l3_branches:
            l3_muons[branch] = tree[f'{l3_name}.{branch}'].array()
        for branch in gen_branches:
            gen_muons[branch] = tree[f'{gen_name}.{branch}'].array()

        l2_muons.update({
            'nVtx': ak.broadcast_arrays(nVtx, tree[f'{l2_name}.pt'].array()[l2_cut])[0],
            'event': ak.broadcast_arrays(event, tree[f'{l2_name}.pt'].array()[l2_cut])[0],
        })
        l3_muons.update({
            'nVtx': ak.broadcast_arrays(nVtx, tree[f'{l3_name}.pt'].array())[0],
            'event': ak.broadcast_arrays(event, tree[f'{l3_name}.pt'].array())[0],
        })
        gen_muons.update({
            'nVtx': ak.broadcast_arrays(nVtx, tree[f'{gen_name}.pt'].array())[0],
            'event': ak.broadcast_arrays(event, tree[f'{gen_name}.pt'].array())[0],
        })

        l2s = MuCollection(**l2_muons)
        l3s = MuCollection(**l3_muons)
        has_matched_l3 = match(l2s, l3s, dR_cutoff=0.3)
        l2_muons[f'pass: {label}'] = has_matched_l3
        all_l2s += MuCollection(**l2_muons)
        all_l3s += MuCollection(**l3_muons)
        
        if study_seeds:
            seed_cut = (tree['seeds.hitBased'].array()==0)&(tree['seeds.layerNum'].array()==0)
            for branch in seed_branches:
                seeds[branch] = tree[f'seeds.{branch}'].array()[seed_cut]
            seeds.update({
                'nVtx': ak.broadcast_arrays(nVtx, tree[f'seeds.pt'].array()[seed_cut])[0],
                'event': ak.broadcast_arrays(event, tree[f'seeds.pt'].array()[seed_cut])[0],
            })
        all_seeds += MuCollection(**seeds)

    print(f'Done: {label}')
    return {
        label: all_l2s,
        #f'{label}_seeds': all_seeds
        #f'{label}_l3s': all_l3s
    }


def preprocess(rets, overlap_events=True):
    df = pd.DataFrame()
    for ret in rets:
        for label, data in ret.items():
            attrs = [a for a in dir(data) if not a.startswith('__') and not callable(getattr(data, a))]
            if len(df)==0:
                df = ak.to_pandas(getattr(data, 'event'))
                df['event'] = ak.to_pandas(getattr(data, 'event'))
                df.reset_index(inplace=True)
                df.set_index(['event','subentry'], inplace=True)
            for a in attrs:
                if (a not in df.columns) and (a!='event'):
                    df_ = ak.to_pandas(getattr(data, 'event'))
                    df_['event'] = ak.to_pandas(getattr(data, 'event'))
                    df_[a] = ak.to_pandas(getattr(data, a))
                    df_.reset_index(inplace=True)
                    df_.set_index(['event','subentry'], inplace=True)
                    df[a] = df_[a]

    df = df.drop_duplicates(subset=['pt','eta','phi'])
    if overlap_events:
        df = df.dropna()
    df['event'] = df['values']
    for c in df.columns:
        if 'pass' in c:
            df[c] = df[c].fillna(-1).astype(int)
    return df


def plot(data, **kwargs):
    out_name = kwargs.pop('out_name', 'test')
    ymin = kwargs.pop('ymin', None)
    ymax = kwargs.pop('ymax', None)
    ylabel = kwargs.pop('ylabel', None)
    title = kwargs.pop('title', None)
    histtype = kwargs.pop('histtype', 'errorbar')
    out_path = '/home/dkondra/hlt-plotting/plots/'
    # Prepare canvas
    fig = plt.figure()
    plt.rcParams.update({'font.size': 12})
    if histtype=='errorbar':
        data_opts = {'marker': '.', 'markersize': 8}
    else:
        data_opts = {}
    fig.clf()
    plotsize = 6
    fig.set_size_inches(plotsize, plotsize)
    plt1 = fig.add_subplot(1, 1, 1)
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(data['values'].keys())))    
    
    bins = np.array(data['edges'])
    xerr = (bins[1:] - bins[:-1]) / 2.

    for (label, values), color in zip(data['values'].items(), colors):
        data_opts['color'] = tuple(color)
        if ('errors_lo' in data.keys()) and ('errors_hi' in data.keys()):
            yerr = [data['errors_lo'][label], data['errors_hi'][label]]
        else:
            yerr = None
        # Draw efficiency plot(s)
        ax = hep.histplot(
            values,
            bins,
            histtype=histtype,
            xerr=[xerr],
            yerr=yerr,
            label=label.replace('pass: ', ''),
            **data_opts
        )

    # Draw line at 1.0
    plt1.plot([bins[0], bins[-1]], [1., 1.], 'b--', linewidth=0.5, zorder=-1)

    # Styling
    lbl = hep.cms.label(ax=plt1, data=False, paper=False, year='')
    if (ymin is not None) and (ymax is not None):
        plt1.set_ylim(ymin, ymax)
    xlabel = data['xlabel']
    if xlabel=='pt':
        xlabel = 'pT, GeV'
    plt1.set_xlabel(xlabel)
    plt1.set_ylabel(ylabel)
    plt1.set_title(title)
    plt1.legend(prop={'size': 'xx-small'})

    try:
        os.mkdir(out_path)
    except Exception:
        pass

    out = f'{out_path}/{out_name}.png'
    fig.savefig(out)
    print(f'Saved: {out}')