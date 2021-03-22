phase2_pu140_OIFromL2 = {
            'name':'Phase 2 DY, PU 140: OI from L2 - VectorHits OFF',
            #'name':'Phase 2 DY, PU 140: OI from L2',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_OIFromL2_default/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_OIFromL2_default/210317_010510/0000/muon*root',
        }

phase2_pu140_OIFromL1TkMu = {
            'name':'Phase 2 DY, PU 140: OI from L1TkMu',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_OIFromL1TkMu_default/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_OIFromL1TkMu_default/210317_073058/0000/muon*root',
            'reference': 'L1Tkmuons'
        }

phase2_pu140_OIFromL2_VH = {
            'name':'Phase 2 DY, PU 140: OI from L2 - VectorHits ON',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_OIFromL2_default_VHenabled/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_OIFromL2_default_VHenabled/210317_015040/0000/muon*root'
        }

phase2_pu140_OIFromL1TkMu_VH = {
            'name':'Phase 2 DY, PU 140: OI from L1TkMu - VectorHits',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_PU140_OIFromL1TkMu_default_VHenabled/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_PU140_OIFromL1TkMu_default_VHenabled/210317_073126/0000/muon*root',
            'reference': 'L1Tkmuons'
        }


phase2_noPU_OIFromL2 = {
            'name':'Phase 2 DY, no PU: OI from L2',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_noPU_OIFromL2_default/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_noPU_OIFromL2_default/210317_033103/0000/muon*root',
        }

phase2_noPU_OIFromL1TkMu = {
            'name':'Phase 2 DY, no PU: OI from L1TkMu',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_noPU_OIFromL1TkMu_default/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_noPU_OIFromL1TkMu_default/210317_062447/0000/muon*root',
            'reference': 'L1Tkmuons'
        }

phase2_noPU_OIFromL2_VH = {
            'name':'Phase 2 DY, no PU: OI from L2 - VectorHits',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_noPU_OIFromL2_default_VHenabled/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_noPU_OIFromL2_default_VHenabled/210317_033132/0000/muon*root'
        }
phase2_noPU_OIFromL1TkMu_VH = {
            'name':'Phase 2 DY, no PU: OI from L1TkMu - VectorHits',
            'path':'/mnt/hadoop/store/user/dkondrat/muonHLT_phase2_DYToLL_noPU_OIFromL1TkMu_default_VHenabled/DYToLL_M-50_TuneCP5_14TeV-pythia8/muonHLT_phase2_DYToLL_noPU_OIFromL1TkMu_default_VHenabled/210317_062517/0000/muon*root',
            'reference': 'L1Tkmuons'
        }

datasets = [
    #phase2_noPU_OIFromL2,
    #phase2_noPU_OIFromL2_VH,
    #phase2_noPU_OIFromL1TkMu,
    #phase2_noPU_OIFromL1TkMu_VH,
    phase2_pu140_OIFromL2,
    phase2_pu140_OIFromL2_VH,
    #phase2_pu140_OIFromL1TkMu,
    #phase2_pu140_OIFromL1TkMu_VH,
]