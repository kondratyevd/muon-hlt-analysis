run2_DY_default = {
    'name':'Run2 default',
    'path':'/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_DYJets_default/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/muonHLTtest_Run2_DYJets_default/210504_194338/0000/muonNtuple_test_MC_*root',
    'target': 'hltOImuons'
}

run2_DY_1HB = {
    'name':'1 HB',
    'path':'/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_DYJets_1HB/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/muonHLTtest_Run2_DYJets_1HB/210504_194549/0000/muonNtuple_test_MC_*root',
    'target': 'hltOImuons'
}

run2_DY_2HB = {
    'name':'2 HB',
    'path':'/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_DYJets_2HB/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/muonHLTtest_Run2_DYJets_2HB//0000/muonNtuple_test_MC_*root',
    'target': 'hltOImuons'
}

run2_DY_1HBd = {
    'name':'1 HB doublet',
    'path':'/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_DYJets_1HBd/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/muonHLTtest_Run2_DYJets_1HBd/210504_194650/0000/muonNtuple_test_MC_*root',
    'target': 'hltOImuons'
}

run2_DY_OI = {
    'name':'Outside-in',
    'path':'/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_DYJets_IOorOI/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/muonHLTtest_Run2_DYJets_IOorOI/210505_153725/0000/muonNtuple_test_MC_*root',
    'target': 'hltOImuons'
}

run2_DY_IO = {
    'name':'Inside-out',
    'path':'/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run2_DYJets_IOorOI/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/muonHLTtest_Run2_DYJets_IOorOI/210505_153725/0000/muonNtuple_test_MC_*root',
    'target': 'hltIOmuons'
}

run3_DY_default = {
    'name':'Run3 default',
    'path':'/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run3_DYJets_default/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLTtest_Run3_DYJets_default/210518_150222/0000/muonNtuple*root',
    'target': 'hltOImuons'
}

run3_DY_1HB = {
    'name':'1 HB',
    'path':'/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run3_DYJets_1HB/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLTtest_Run3_DYJets_1HB/210521_185056/0000/muonNtuple*root',
    'target': 'hltOImuons'
}

run3_DY_2HB = {
    'name':'2 HB',
    'path':'/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run3_DYJets_2HB/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLTtest_Run3_DYJets_2HB/210521_185222/0000/muonNtuple*root',
    'target': 'hltOImuons'
}

run3_DY_1HBd = {
    'name':'1 HB doublet',
    'path':'/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run3_DYJets_1HBd/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLTtest_Run3_DYJets_1HBd/210521_185336/0000/muonNtuple*root',
    'target': 'hltOImuons'
}

run_muonHLTtest_Run3_DYJets_DNN_5seeds = {
    "name": "Run3 DY DNN 5 seeds", 
    "path": "/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run3_DYJets_DNN_5seeds/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLTtest_Run3_DYJets_DNN_5seeds/210520_233658/0000/muonNtuple*.root"
}
run_muonHLTtest_Run3_DYJets_DNN_7seeds = {
    "name": "Run3 DY DNN 7 seeds", 
    "path": "/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run3_DYJets_DNN_7seeds/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLTtest_Run3_DYJets_DNN_7seeds/210521_012558/0000/muonNtuple*.root"
}
run_muonHLTtest_Run3_DYJets_DNN_b5seeds_e7seeds = {
    "name": "Run3 DY DNN 5-barrel 7-endcap", 
    "path": '/mnt/hadoop/store/user/dkondrat/muonHLTtest_Run3_DYJets_DNN_b5seeds_e7seeds/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLTtest_Run3_DYJets_DNN_b5seeds_e7seeds/210524_034728/0000/muonNtuple*.root'
}

datasets = [
    #run2_DY_default,
    run3_DY_default,
    #run2_DY_1HB,
    #run2_DY_1HBd,
    #run3_DY_1HB,
    #run3_DY_2HB,
    #run3_DY_1HBd,
    #run2_DY_OI,
    #run2_DY_IO,
    run_muonHLTtest_Run3_DYJets_DNN_5seeds,
    run_muonHLTtest_Run3_DYJets_DNN_7seeds,
    run_muonHLTtest_Run3_DYJets_DNN_b5seeds_e7seeds
]

datasets_dict = {

}
