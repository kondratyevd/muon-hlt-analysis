phase2_default = {
    'name': 'Run2 strategy (PU 140)',
    'path': '/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_default/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_default/210217_003837/0000/muonNtuple_phase2_MC_*.root'
}

phase2_default_VH = {
    'name': 'Run2 strategy but with VectorHits (PU 140)',
    'path': '/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_default_VH/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_default_VH/210217_161545/0000/muonNtuple_phase2_MC_*.root'
}

phase2_1HB = {
    'name': 'Phase2 DY PU140: 1HB - VH OFF',
    'path': '/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_1HB/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_1HB/210217_004357/0000/muonNtuple_phase2_MC_*.root'
}

phase2_1VHB = {
    'name': 'Phase2 DY PU140: 1HB - VH ON',
    'path': '/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_1VHB/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_1VHB/210217_161708/0000/muonNtuple_phase2_MC_*.root'
}

phase2_1HLIP = {
    'name': 'Phase2 DY PU140: 1HL(IP) - VH OFF',
    'path': '/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_1HLIP/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_1HLIP/210217_004824/0000/muonNtuple_phase2_MC_*.root'
}

phase2_1HLMuS = {
    'name': 'Phase2 DY PU140: 1HL(MuS) - VH OFF',
    'path': '/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_1HLMuS/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_1HLMuS/210217_004936/0000/muonNtuple_phase2_MC_*.root'
}

phase2_1HLIP_vh = {
    'name': 'Phase2 DY PU140: 1HL(IP) - VH ON',
    'path': '/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_1HLIP_VHenabled/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_1HLIP_VHenabled/210220_190025/0000/muonNtuple_phase2_MC_*.root'
}

phase2_1HLMuS_vh = {
    'name': 'Phase2 DY PU140: 1HL(MuS) - VH ON',
    'path': '/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_1HLMuS_VHenabled/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_1HLMuS_VHenabled/210220_190119/0000/muonNtuple_phase2_MC_*.root'
}


datasets = [
    #phase2_default,
    #phase2_default_VH,
    phase2_1HB,
    phase2_1HLIP,
    #phase2_1HLMuS,
    phase2_1VHB,
    phase2_1HLIP_vh,
    #phase2_1HLMuS_vh
]
