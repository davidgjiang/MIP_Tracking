import uproot
import numpy as np
import awkward
import concurrent.futures
import glob 
executor = concurrent.futures.ThreadPoolExecutor(8)

files = glob.glob('/home/dgj1118/MIP_Tracking/kaons/v12_trigger/*.root')

# electron radius of containment based on angle and momentum

# 1. angle < 10 degrees, momentum < 500 MeV
radius_beam_68_thetalt10_plt500 = np.array([
    4.045666158618167,  4.086393662224346,  4.359141107602775,
    4.666549994726691,  5.8569181911416015, 6.559716356124256,
    8.686967529043072,  10.063482736354674, 13.053528344041274,
    14.883496407943747, 18.246694748611368, 19.939799900443724,
    22.984795944506224, 25.14745829663406,  28.329169392203216,
    29.468032123356345, 34.03271241527079,  35.03747443690781,
    38.50748727211848,  39.41576583301171,  42.63622296033334,
    45.41123601592071,  48.618139095742876, 48.11801717451056,
    53.220539860213655, 58.87753380915155,  66.31550881539764,
    72.94685877928593,  85.95506228335348,  89.20607201266672,
    93.34370253818409,  96.59471226749734,  100.7323427930147,
    103.98335252232795])

# 2. angle < 10 degrees, momentum >= 500 MeV
radius_beam_68_thetalt10_pgt500 = np. array([
    4.081926458777424,  4.099431732299409,  4.262428482867968,
    4.362017581473145,  4.831341579961153,  4.998346041276382,
    6.2633736512415705, 6.588371889265881,  8.359969947444522,
    9.015085558044309,  11.262722588206483, 12.250305471269183,
    15.00547660437276,  16.187264014640103, 19.573764900578503,
    20.68072032434797,  24.13797140783321,  25.62942209291236,
    29.027596514735617, 30.215039667389316, 33.929540248019585,
    36.12911729771914,  39.184563500620946, 42.02062468386282,
    46.972125628650204, 47.78214816041894,  55.88428562462974,
    59.15520134927332,  63.31816666637158,  66.58908239101515,
    70.75204770811342,  74.022963432757,    78.18592874985525,
    81.45684447449884])

# 3. angle between 10 and 20 degrees
radius_beam_68_theta10to20 = np.array([
    4.0251896715647115, 4.071661598616328,  4.357690094817289,
    4.760224640141712,  6.002480766325418,  6.667318981016246,
    8.652513285172342,  9.72379373302137,   12.479492693251478,
    14.058548828317289, 17.544872909347912, 19.43616066939176,
    23.594162859513734, 25.197329065282954, 29.55995803074302,
    31.768946746958296, 35.79247330197688,  37.27810357669942,
    41.657281051476545, 42.628141392692626, 47.94208483539388,
    49.9289473559796,   54.604030254423975, 53.958762417361655,
    53.03339560920388,  57.026277390001425, 62.10810455035879,
    66.10098633115634,  71.1828134915137,   75.17569527231124,
    80.25752243266861,  84.25040421346615,  89.33223137382352,
    93.32511315462106])

# 4. angle greater than 20 degrees
radius_beam_68_thetagt20 = np.array([
    4.0754238481177705, 4.193693485630508,  5.14209420056253,
    6.114996249971468,  7.7376807326481645, 8.551663213602291,
    11.129110612057813, 13.106293737495639, 17.186617323282082,
    19.970887612094604, 25.04088272634407,  28.853696411302344,
    34.72538105333071,  40.21218694947545,  46.07344239520299,
    50.074953583805346, 62.944045771758645, 61.145621459396814,
    69.86940198299047,  74.82378572939959,  89.4528387422834,
    93.18228303096758,  92.51751129204555,  98.80228884380018,
    111.17537347472128, 120.89712563907408, 133.27021026999518,
    142.99196243434795, 155.36504706526904, 165.08679922962185,
    177.45988386054293, 187.18163602489574, 199.55472065581682,
    209.2764728201696])

# distance between each ECal layer
layer_dz = np.array([7.850, 13.300, 26.400, 33.500, 47.950, 56.550, 72.250, 81.350, 97.050, 106.150,
            121.850, 130.950, 146.650, 155.750, 171.450, 180.550, 196.250, 205.350, 221.050,
            230.150, 245.850, 254.950, 270.650, 279.750, 298.950, 311.550, 330.750, 343.350,
            362.550, 375.150, 394.350, 406.950, 426.150, 438.750])

# z-position of each ECal layer
layer_z = 240.5 + layer_dz


# list of branches we want to store from the root file
branches = []
branch_suffix = 'v3_v12'

# branchnames
ecal_branch = 'EcalRecHits_{}/EcalRecHits_{}'.format(branch_suffix, branch_suffix)
ecalSP_branch = 'EcalScoringPlaneHits_{}/EcalScoringPlaneHits_{}'.format(branch_suffix, branch_suffix)
tSP_branch = 'TargetScoringPlaneHits_{}/TargetScoringPlaneHits_{}'.format(branch_suffix, branch_suffix)
ecalVeto_branch = 'EcalVeto_{}'.format(branch_suffix)
ecalSim_branch = 'EcalSimHits_{}/EcalSimHits_{}'.format(branch_suffix, branch_suffix)

# add EcalRecHits branches
for leaf in ['xpos_', 'ypos_', 'zpos_', 'energy_', 'amplitude_']:
    branches.append('{}.{}'.format(ecal_branch, leaf))

# add EcalVeto branches
for leaf in ['showerRMS_', 'epAng_', 'passesVeto_','discValue_', 'nStraightTracks_', 'nLinregTracks_','firstNearPhLayer_']:
    branches.append('{}/{}'.format(ecalVeto_branch, leaf))

# add EcalSPHits and TargetSPHits branches
# Use EcalSPHits for fiducial/non-fiducial and TargetSPHits for trajectories
for leaf in ['x_', 'y_', 'z_', 'px_', 'py_', 'pz_', 'pdgID_', 'trackID_']:
    branches.append('{}.{}'.format(ecalSP_branch, leaf))
    branches.append('{}.{}'.format(tSP_branch, leaf))

# add EcalSimHits branches
for leaf in ['x_', 'y_', 'z_', 'pdgCodeContribs_', 'edepContribs_', 'incidentIDContribs_']:
    branches.append('{}.{}'.format(ecalSim_branch, leaf))

# add the eventNumber leaf
branches.append('EventHeader/eventNumber_')

filenum = 0
for filename in files:    
    print("Processing file {}".format(filename))
    output_path = "/home/dgj1118/MIP_Tracking/kaons/v12_trigger_BDT_MIP_skimmed_RoC/v12_trigger_BDT_MIP_skimmed_{}.root".format(filenum)
    filenum += 1
    # use uproot and load all of the data into a dictionary: {'leaf1': [data], 'leaf2': [data], ...}
    t = uproot.open(filename)['LDMX_Events']
    table = t.arrays(expressions=branches, interpretation_executor=executor)

    # we store all of our data into the dictionary "tree"
    tree = {}
    for branch in branches:
        tree[branch] = table[branch]

    # find the number of events that pass through the vetoes
    # record the event numbers of the events that pass through the vetoes
    events = 0
    eventNumbers = []
    for event in range(len(tree['EventHeader/eventNumber_'])):
        events +=1
        eventNumbers.append(tree['EventHeader/eventNumber_'][event])

    passVetoes = []

    # keep events that pass ECal Vetoes and MIP Tracking
    for event in range(len(tree['EcalVeto_v3_v12/discValue_'])):
        if (tree['EcalVeto_v3_v12/discValue_'][event] > 0.99 
        and tree['EcalVeto_v3_v12/nStraightTracks_'][event] == 0 
        and tree['EcalVeto_v3_v12/nLinregTracks_'][event] == 0
        and tree['EcalVeto_v3_v12/firstNearPhLayer_'][event] >= 6):
            passVetoes.append(event)

    b1 = []
    b2 = []
    b3 = []
    b4 = []
    b5 = []

    # loop through all of the events
    for i in range(len(tree['EventHeader/eventNumber_'])):

        if i not in passVetoes:
            continue

        tSPHits = {}
        ecalSPHits = {}
        for x in ['x_', 'y_', 'z_', 'px_', 'py_', 'pz_', 'pdgID_', 'trackID_']:
            tSPHits[x] = (table['{}.{}'.format(tSP_branch, x)])[i]
            ecalSPHits[x] = (table['{}.{}'.format(ecalSP_branch, x)])[i]

        # find the max pz at the target scoring plane for the recoil electron
        max_pz = 0
        r = 0
        for j in range(len(tSPHits['z_'])):
            if tSPHits['pdgID_'][j] == 11 and tSPHits['z_'][j] > 4.4 and tSPHits['z_'][j] < 4.6 and tSPHits['pz_'][j] > max_pz and tSPHits['trackID_'][j] == 1:
                max_pz = tSPHits['pz_'][j]
                r = j

        # find the max pz at the ecal face for the recoil electron
        max_pz_e = 0
        r_e = 0
        for j in range(len(ecalSPHits['z_'])):
            if ecalSPHits['pdgID_'][j] == 11 and ecalSPHits['z_'][j] > 239 and ecalSPHits['z_'][j] < 242 and ecalSPHits['pz_'][j] > max_pz_e and ecalSPHits['trackID_'][j] == 1:
                max_pz_e = ecalSPHits['pz_'][j]
                r_e = j
        fiducial = max_pz_e != 0
        
        # ignore non-fiducial events (for now. maybe can work with them in the future)
        if fiducial:
            E_beam = 4000
            ecalSP = 240.5
            targetSP = 4.577700138092041

            # positions and trajectory vectors of recoil e- at Ecal SP
            enorm_sp = np.array((ecalSPHits['px_'][r_e]/ecalSPHits['pz_'][r_e], ecalSPHits['py_'][r_e]/ecalSPHits['pz_'][r_e], 1.0))
            etraj_sp = np.array((ecalSPHits['x_'][r_e], ecalSPHits['y_'][r_e], ecalSPHits['z_'][r_e]))

            # momentum vectors of the recoil e- at Ecal SP
            emom_sp = np.array((ecalSPHits['px_'][r_e],ecalSPHits['py_'][r_e],ecalSPHits['pz_'][r_e]))

            # positions and trajectory vectors of photon at Ecal SP
            pnorm_sp = np.array((-tSPHits['px_'][r]/(E_beam - tSPHits['pz_'][r]), -tSPHits['py_'][r]/(E_beam - tSPHits['pz_'][r]), 1.0))
            ptraj_sp = np.array((tSPHits['x_'][r] + (ecalSP-targetSP)*pnorm_sp[0], tSPHits['y_'][r] + (ecalSP-targetSP)*pnorm_sp[1], tSPHits['z_'][r] + (ecalSP-targetSP)))
            
            # store the trajectory information 
            b1.append(enorm_sp)
            b2.append(etraj_sp)
            b3.append(pnorm_sp)
            b4.append(ptraj_sp)
            b5.append(emom_sp)
        else:
            passVetoes.remove(i)

    trajectories = {}
    trajectories['eVec_EcalSP'] = b1
    trajectories['ePos_EcalSP'] = b2
    trajectories['pVec_EcalSP'] = b3
    trajectories['pPos_EcalSP'] = b4
    eMomentumSP = b5

    # Get the PDG IDs of the hits and their position
    pdgIDs = []
    hitPositions = []

    for event in range(len(tree['EventHeader/eventNumber_'])):
        if event not in passVetoes:
            continue

        # data from EcalSimHits
        sim_dict = {}
        for x in ['x_', 'y_', 'z_', 'pdgCodeContribs_', 'edepContribs_', 'incidentIDContribs_']:
            sim_dict[x] = table['{}.{}'.format(ecalSim_branch, x)][event]

        # data from EcalRecHits
        ecal_dict = {}
        for x in ['xpos_', 'ypos_', 'zpos_']:
            ecal_dict[x] = table['{}.{}'.format(ecal_branch, x)][event]

        rec_matched_ids = []
        rec_positions = []
        for j in range(len(ecal_dict['zpos_'])):
            # For each hit:  Find a contrib w/ the same position, then match:
            simIndex = None # EcalSimHit index
            contribIndex = None # index of contribution within EcalSimHit
            for k in range(len(sim_dict['x_'])):
                if round(sim_dict['x_'][k]) == round(ecal_dict['xpos_'][j]) and \
                    round(sim_dict['y_'][k]) == round(ecal_dict['ypos_'][j]) and \
                    round(sim_dict['z_'][k]) == round(ecal_dict['zpos_'][j]):
                    simIndex = k # we found a matching hit. 
                    # now, go through the contribs and find the pdgID of the contrib w/ max edep:
                    eDepMax = 0
                    for l in range(len(sim_dict['edepContribs_'][k])):
                        if sim_dict['edepContribs_'][k][l] > eDepMax:
                            eDepMax = sim_dict['edepContribs_'][k][l]
                            contribIndex = l
            if not simIndex:  # If no EcalSimHit found, presumably noise; record
                rec_matched_ids.append(-9999999) # use -9999999 as the pdgID for noise
                rec_positions.append([ecal_dict['xpos_'][j],ecal_dict['ypos_'][j],ecal_dict['zpos_'][j]])
            else:
                rec_matched_ids.append(sim_dict['pdgCodeContribs_'][simIndex][contribIndex])
                rec_positions.append([ecal_dict['xpos_'][j],ecal_dict['ypos_'][j],ecal_dict['zpos_'][j]])
        
        pdgIDs.append(rec_matched_ids)
        hitPositions.append(rec_positions)

    # Save all information
    hitXYZ = [] # [[x0 y0 z0] [x1 y1 z1] ... [xN yN zN]]
    pdgID = []  # [pdgID_0 pdgID_1 ... pdgID_N]
    eTraj = []  # [[x0 ... x33] [y0 ... y33] [z0 ... z33]]
    pTraj = []  # [[x0 ... x33] [y0 ... y33] [z0 ... z33]]

    passVetoEvents = np.arange(0,len(passVetoes),1) # re-number the events that pass all the vetoes to: 0, 1, 2, ... N

    for event in passVetoEvents:
        hitXYZPerEvent = [] # the hit positions per event
        pdgIDPerEvent = [] # pdg id of hits per event 

        # obtain the e- ECal layer intercepts 
        eLayerIntercepts = []

        eIntercept = (trajectories['ePos_EcalSP'][event][0] + layer_dz*trajectories['eVec_EcalSP'][event][0], 
                    trajectories['ePos_EcalSP'][event][1] + layer_dz*trajectories['eVec_EcalSP'][event][1], 
                    trajectories['ePos_EcalSP'][event][2] + layer_dz*trajectories['eVec_EcalSP'][event][2])
        
        eLayerIntercepts.append(eIntercept)
        eLayerIntercepts = np.concatenate(np.array(eLayerIntercepts)) # form: [[x0 ... x33] [y0 ... y33] [z0 ... z33]]
        eTraj.append(eLayerIntercepts)

        # obtain the photon ECal layer intercepts 
        pLayerIntercepts = []

        pIntercept = (trajectories['pPos_EcalSP'][event][0] + layer_dz*trajectories['pVec_EcalSP'][event][0], 
                    trajectories['pPos_EcalSP'][event][1] + layer_dz*trajectories['pVec_EcalSP'][event][1], 
                    trajectories['pPos_EcalSP'][event][2] + layer_dz*trajectories['pVec_EcalSP'][event][2])
        
        pLayerIntercepts.append(pIntercept)
        pLayerIntercepts = np.concatenate(np.array(pLayerIntercepts)) # form: [[x0 ... x33] [y0 ... y33] [z0 ... z33]]
        pTraj.append(pLayerIntercepts)

        # calculate the electron angle wrt Z-axis
        eVector = np.array(trajectories['eVec_EcalSP'][event])
        zVector = np.array([0,0,1.0])
        angle = np.arccos(np.dot(eVector,zVector)/(np.sqrt(eVector[0]**2 + eVector[1]**2 + eVector[2]**2)))*180.0/np.pi

        # calculate the electron momentum at Ecal SP
        eMomentum = np.sqrt(eMomentumSP[event][0]**2 + eMomentumSP[event][1]**2 + eMomentumSP[event][2]**2)

        # loop through each hit in this event
        for hit in range(len(hitPositions[event])):
            x = hitPositions[event][hit][0]
            y = hitPositions[event][hit][1]
            z = hitPositions[event][hit][2]
            pdgid = pdgIDs[event][hit]

            # choose the corresponding radius of containment to use based off electron trajectory angle and momentum
            eRoC = [] # choice of electron 68% radius of containment
            if (angle < 10.0 and eMomentum < 500.0):
                eRoC = radius_beam_68_thetalt10_plt500
            elif (angle < 10.0 and eMomentum >= 500.0):
                eRoC = radius_beam_68_thetalt10_pgt500
            elif (angle >= 10.0 and angle <= 20.0):
                eRoC = radius_beam_68_theta10to20
            elif (angle > 20.0):
                eRoC = radius_beam_68_thetagt20
            else:
                print("None satisfied...ERROR!")
            
            # loop through each layer of the ECal and check if its distance from the e- layer intercept is > corresponding layer's containment radius
            for layer in range(34):
                if z >= layer_z[layer] - .25 and z <= layer_z[layer] + .25:
                    dist = np.sqrt((eLayerIntercepts[0][layer] - x)**2 + (eLayerIntercepts[1][layer] - y)**2)
                    if dist >= eRoC[layer]:
                        hitXYZPerEvent.append([x,y,z])
                        pdgIDPerEvent.append(pdgid)
        
        hitXYZ.append(hitXYZPerEvent)
        pdgID.append(pdgIDPerEvent)

    import ROOT as r
    # Make the ROOT file
    outfile = r.TFile(output_path,"RECREATE")
    outfile.cd()
    tree = r.TTree("Events","Events")

    hitX = r.std.vector('float')()
    hitY = r.std.vector('float')()
    hitZ = r.std.vector('float')()
    pdg_IDs = r.std.vector('int')()
    photon_TrajectoryX = r.std.vector('float')()
    photon_TrajectoryY = r.std.vector('float')()
    photon_TrajectoryZ = r.std.vector('float')()
    electron_TrajectoryX = r.std.vector('float')()   
    electron_TrajectoryY = r.std.vector('float')()   
    electron_TrajectoryZ = r.std.vector('float')()   

    tree.Branch("hitX",hitX)
    tree.Branch("hitY",hitY)
    tree.Branch("hitZ",hitZ)
    tree.Branch("pdg_IDs",pdg_IDs)
    tree.Branch("photon_TrajectoryX",photon_TrajectoryX)
    tree.Branch("photon_TrajectoryY",photon_TrajectoryY)
    tree.Branch("photon_TrajectoryZ",photon_TrajectoryZ)
    tree.Branch("electron_TrajectoryX",electron_TrajectoryX)
    tree.Branch("electron_TrajectoryY",electron_TrajectoryY)
    tree.Branch("electron_TrajectoryZ",electron_TrajectoryZ)

    for event in range(len(passVetoEvents)):
        hitX.clear()
        hitY.clear()
        hitZ.clear()
        pdg_IDs.clear()
        photon_TrajectoryX.clear()
        photon_TrajectoryY.clear()
        photon_TrajectoryZ.clear()
        electron_TrajectoryX.clear()
        electron_TrajectoryY.clear()
        electron_TrajectoryZ.clear()

        for layer in range(34):
            photon_TrajectoryX.push_back(pTraj[event][0][layer])
            photon_TrajectoryY.push_back(pTraj[event][1][layer])
            photon_TrajectoryZ.push_back(pTraj[event][2][layer])
            electron_TrajectoryX.push_back(eTraj[event][0][layer])
            electron_TrajectoryY.push_back(eTraj[event][1][layer])
            electron_TrajectoryZ.push_back(eTraj[event][2][layer])

        for hit in range(len(hitXYZ[event])):
            hitX.push_back(hitXYZ[event][hit][0])
            hitY.push_back(hitXYZ[event][hit][1])
            hitZ.push_back(hitXYZ[event][hit][2])
            pdg_IDs.push_back(pdgID[event][hit])

        tree.Fill()

    outfile.cd()
    tree.Write()
    outfile.Close()