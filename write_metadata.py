import os, sys
import pandas as pd
from array import array
import numpy as np
import ROOT

from model_descriptions import *

path = "/home/dkondra/muon-hlt-analysis/metadata/"

inputs = ["pt","eta", "phi", "validHits",
        "tsos_IP_eta", "tsos_IP_phi", "tsos_IP_pt", "tsos_IP_pt_eta", "tsos_IP_pt_phi",
        "err0_IP", "err1_IP", "err2_IP", "err3_IP", "err4_IP",
        "tsos_MuS_eta", "tsos_MuS_phi", "tsos_MuS_pt", "tsos_MuS_pt_eta", "tsos_MuS_pt_phi",
        "err0_MuS", "err1_MuS", "err2_MuS", "err3_MuS", "err4_MuS",
        "tsos_IP_valid", "tsos_MuS_valid"]
columns = ['nHBd', 'nHLIP', 'nHLMuS']



def write_decoder(name, desc):
    scheme = pd.DataFrame(data=desc['scheme'])
    xbins = scheme.columns.values
    ybins = scheme.index.values
    xbins = np.append(xbins, [max(xbins)+1])
    ybins = np.append(ybins, [max(ybins)+1])

    decoder = ROOT.TH2D(
        'scheme', 'scheme',
        scheme.shape[1], array('d', xbins),
        scheme.shape[0], array('d', ybins)
    )

    for strategy in scheme.index.values:
        for parameter in scheme.columns.values:
            decoder.SetBinContent(
                int(parameter)+1, int(strategy)+1,
                scheme.loc[strategy, parameter]
            )
            decoder.GetXaxis().SetBinLabel(int(parameter)+1, columns[parameter])

    ########################################

    bins = [i for i in range(len(inputs)+1)]
    input_order = ROOT.TH1D(
        'input_order', 'input_order',
        len(inputs), array('d', bins),
    )

    for i, inp_name in enumerate(inputs):
        #input_order.SetBinContent(i+1, i)
        input_order.GetXaxis().SetBinLabel(i+1, inp_name)

    ########################################

    layer_names = ROOT.TH1D(
        'layer_names', 'layer_names', 2, array('d', [0, 1, 2]),
    )

    for i, key in enumerate(['input_layer', 'output_layer']):
        #layer_names.SetBinContent(i+1, i)
        layer_names.GetXaxis().SetBinLabel(i+1, desc[key])   

    ########################################

    f_out = ROOT.TFile.Open(f"{path}/metadata_{name}.root", "RECREATE")
    decoder.Write()
    input_order.Write()
    layer_names.Write()
    f_out.Close()

descriptions = {
    '5_seeds': model_5_seeds,
    '7_seeds': model_7_seeds,
}

for name, desc in descriptions.items():
    write_decoder(name, desc)

