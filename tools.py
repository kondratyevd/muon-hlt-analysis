from tqdm import tqdm
import glob
import uproot
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplhep as hep


from utils import *

l1_branches = ['pt', 'eta', 'phi']

l2_branches = [
    'pt','eta', 'phi',
    'validHits',
    'tsos_IP_eta', 'tsos_IP_phi', 'tsos_IP_pt',
    'tsos_IP_pt_eta', 'tsos_IP_pt_phi',
    'err0_IP', 'err1_IP', 'err2_IP', 'err3_IP', 'err4_IP',
    'tsos_MuS_eta', 'tsos_MuS_phi', 'tsos_MuS_pt',
    'tsos_MuS_pt_eta', 'tsos_MuS_pt_phi',
    'err0_MuS', 'err1_MuS', 'err2_MuS', 'err3_MuS', 'err4_MuS',
    'tsos_IP_valid', 'tsos_MuS_valid',
]

l3_branches = gen_branches = ['pt', 'eta', 'phi']

seed_branches = [
    'pt', 'eta', 'phi', 'hitBased', 'eta_pos', 'layerNum',
    'tsos_q_p_err', 'tsos_lambda_err', 'tsos_phi_err','tsos_xT_err','tsos_yT_err', 
    'dR_pos', 'dR_mom'
]

def build_dataset(dataset, progress_bar=False, study_seeds=False):
    label = dataset['name']
    path = dataset['path']
    ref_name = dataset.pop('reference', 'L2muons')
    trg_name = dataset.pop('target', 'hltOImuons')
    gen_name = 'genParticles'

    all_ref = MuCollection()
    all_trg = MuCollection()
    all_seeds = MuCollection()
    
    print(f'Processing {label}')

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

        ref_muons = {}
        trg_muons = {}
        gen_muons = {}
        seeds = {}

        #ref_cut = (tree[f'{ref_name}.validHits'].array()>20)&(tree[f'{ref_name}.pt'].array()>10)
        #ref_cut = (tree[f'{ref_name}.pt'].array()>24)
        ref_cut = (tree[f'{ref_name}.pt'].array()>0)

        if 'L2' in ref_name:
            ref_branches = l2_branches
        elif 'L1' in ref_name:
            ref_branches = l1_branches
        else:
            ref_branches = []

        trg_branches = l3_branches
        
        for branch in ref_branches:
            ref_muons[branch] = tree[f'{ref_name}.{branch}'].array()[ref_cut]
        for branch in trg_branches:
            trg_muons[branch] = tree[f'{trg_name}.{branch}'].array()
        for branch in gen_branches:
            gen_muons[branch] = tree[f'{gen_name}.{branch}'].array()

        ref_muons.update({
            'nVtx': ak.broadcast_arrays(nVtx, tree[f'{ref_name}.pt'].array()[ref_cut])[0],
            'event': ak.broadcast_arrays(event, tree[f'{ref_name}.pt'].array()[ref_cut])[0],
        })
        trg_muons.update({
            'nVtx': ak.broadcast_arrays(nVtx, tree[f'{trg_name}.pt'].array())[0],
            'event': ak.broadcast_arrays(event, tree[f'{trg_name}.pt'].array())[0],
        })
        gen_muons.update({
            'nVtx': ak.broadcast_arrays(nVtx, tree[f'{gen_name}.pt'].array())[0],
            'event': ak.broadcast_arrays(event, tree[f'{gen_name}.pt'].array())[0],
        })

        refs = MuCollection(**ref_muons)
        trgs = MuCollection(**trg_muons)
        has_matched = match(refs, trgs, dR_cutoff=0.3)
        ref_muons[f'pass: {label}'] = has_matched
        all_ref += MuCollection(**ref_muons)
        all_trg += MuCollection(**trg_muons)
        
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
        label: all_ref,
        #f'{label}_seeds': all_seeds
        #f'{label}_l3s': all_trg
    }


def preprocess(rets, only_overlap_events=True):
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
    if only_overlap_events:
        df = df.dropna()
    df['event'] = df['values']
    for c in df.columns:
        if 'pass' in c:
            df[c] = df[c].fillna(-1).astype(int)
    return df


def plot_efficiencies(df, **kwargs):
    cols_to_plot = kwargs.pop('cols_to_plot', None)
    opt_label = kwargs.pop('opt_label', '')
    draw = kwargs.pop('draw', True)
    prefix = kwargs.pop('prefix', '')
    out_path = kwargs.pop('out_path', './')
    ymin = kwargs.pop('ymin', 0.0)

    x_opts = ['eta', 'pt', 'nVtx']
    bin_opts = {
        'eta': regular(20, -2.4, 2.4),
        'pt': np.array([5, 7, 9, 12, 16, 20, 24, 27, 30, 35, 40, 45, 50, 60, 70, 90, 150]),
        'nVtx': regular(50, 0, 200),
    }

    strategies = cols_to_plot
    if strategies==None:
        strategies = []
        for c in df.columns:
            if ('pass' in c):
                strategies.append(c)

    plotting_data = {}
    for x in x_opts:
        plotting_data[x] = {}
        out_name = f'{prefix}eff_vs_{x}'
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
            'out_name': out_name,
        }
        plotting_data[x] = data
        if draw:
            plot(
                data,
                out_name=out_name,
                ymin=ymin,
                ymax=1.01,
                ylabel='Efficiency',
                out_path=out_path
            )
    return plotting_data


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


def plot(data, **kwargs):
    out_name = kwargs.pop('out_name', 'test')
    ymin = kwargs.pop('ymin', None)
    ymax = kwargs.pop('ymax', None)
    ylabel = kwargs.pop('ylabel', None)
    title = kwargs.pop('title', None)
    histtype = kwargs.pop('histtype', 'errorbar')
    out_path = kwargs.pop('out_path', './')

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
    
    
def plot_loss(history, label, out_path):
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
    out = f'{out_path}/loss_{label}'
    fig.savefig(out)
    print(f'Saved loss plot: {out}')
