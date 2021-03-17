from datasets.datasets_run2_DY_1to7seeds import datasets_dict as ds_dy
ds_dy = ds_dy['5 seeds']
from datasets.datasets_run2_TT2L2Nu_5seeds import datasets as ds_tt2l2nu
from datasets.datasets_run2_TTSemiLept_5seeds import datasets as ds_ttsemilept
from datasets.datasets_run2_QCD_5seeds import datasets as ds_qcd

default_dy = {
"name": "default Drell-Yan", 
"path": "/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_DYJets_default/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/muonHLTtest_Run2_DYJets_default//210204_052450/0000/muonNtuple_test_MC_*.root"
} 

default_tt2l2nu = {
"name": "default TTTo2L2Nu", 
"path": "/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_TTTo2L2Nu_default/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/muonHLTtest_Run2_TTTo2L2Nu_default/210220_192158/0000/muonNtuple_test_MC_*.root"
} 

default_ttsemilept = {
"name": "default TTToSemiLeptonic", 
"path": "/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_TTToSemiLeptonic_default/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/muonHLTtest_Run2_TTToSemiLeptonic_default/210221_163329/0000/muonNtuple_test_MC_*.root"
} 

default_qcd = {
"name": "default QCD", 
"path": "/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_QCD_default/QCD_Pt-20toInf_MuEnrichedPt5_TuneCP5_13TeV_pythia8/muonHLTtest_Run2_QCD_default/210222_142439/0000/muonNtuple_test_MC_*.root"
} 

datasets = [
    default_dy,
    default_tt2l2nu,
    default_ttsemilept,
    default_qcd
]

datasets_dict = {
#    'default DY': [default_dy],
#    'default TTTo2L2Nu': [default_tt2l2nu],
#    'default TTToSemiLeptonic': [default_ttsemilept],
#    'default QCD': [default_qcd],
    'DNN 5 seeds DY': ds_dy,
    'DNN 5 seeds TTTo2L2Nu': ds_tt2l2nu,
    'DNN 5 seeds TTToSemiLeptonic': ds_ttsemilept,
    'DNN 5 seeds QCD': ds_qcd
}

