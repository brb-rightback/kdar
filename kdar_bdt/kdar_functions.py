import matplotlib as mpl
import matplotlib.pyplot as plt
import uproot3 as uproot
import uproot4 as uprootnew
import pandas as pd
import numpy as np
import math
from tqdm import tqdm

import general_functions as utils

####### BDT Training related ################## 
def label_train(df,file_name):

    if file_name=="":
        df["is_train"] = [0 for i in range(df.shape[0])]
        return df
    
    train_run = []
    train_subrun = []
    train_event = []
    train_ssm_kine_energy = []
    train_ssm_x_dir = []
    
    
    with open(file_name) as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            train_run.append(int(row[0]))
            train_subrun.append(int(row[1]))
            train_event.append(int(row[2]))
            train_ssm_kine_energy.append(round(float(row[9]),5))
            train_ssm_x_dir.append(round(float(row[10]),5))
        
    run = df["run"].to_numpy()
    subrun = df["subrun"].to_numpy()
    event = df["event"].to_numpy()
    ssm_kine_energy = df["ssm_kine_energy"].to_numpy()
    ssm_x_dir = df["ssm_x_dir"].to_numpy()
    
    is_train = []
    
    for i in tqdm(range(len(run))):
        this_is_train = 0
        if run[i] in train_run: 
            this_index = train_run.index(run[i])
            dct = [e for e, x in enumerate(train_run) if x == run[i]]
            for this_index in dct:
                if subrun[i] == train_subrun[this_index] and event[i] == train_event[this_index] and ssm_kine_energy[i]>train_ssm_kine_energy[this_index]-0.001 and ssm_kine_energy[i]<train_ssm_kine_energy[this_index]+0.001: 
                    this_is_train = 1
                    break
        is_train.append(this_is_train)
    
    df["is_train"] = is_train
    return df


####### Truth variable calculations ################## 
def add_ntrue_nu_angle(df):
    #Truth info for the nu from the pfeval tree
    nu0 = df["truth_nu_momentum[0]"].to_numpy()
    nu1 = df["truth_nu_momentum[1]"].to_numpy()
    nu2 = df["truth_nu_momentum[2]"].to_numpy()
    nu3 = df["truth_nu_momentum[3]"].to_numpy()
    
    nu_thetas = []
    nu_phis = []
    nu_costhetas = []
    nu_cosphis = []
    
    nu_numi_thetas = []
    nu_numi_phis = []
    nu_numi_costhetas = []
    nu_numi_cosphis = []
    
    Enu = []
    
    for i in tqdm(range(len(nu3))):
        
        nu_theta = np.nan
        nu_phi = np.nan
        
        nu_numi_theta = np.nan
        nu_numi_phi = np.nan
        
        if(nu3[i]<=0): 
            
            nu_thetas.append(nu_theta)
            nu_phis.append(nu_phi)
            nu_costhetas.append(nu_theta)
            nu_cosphis.append(nu_phi)
            
            nu_numi_thetas.append(nu_numi_theta)
            nu_numi_phis.append(nu_numi_phi)
            nu_numi_costhetas.append(nu_numi_theta)
            nu_numi_cosphis.append(nu_numi_phi)
            
        else: 
            
            nu_theta, nu_phi = utils. get_angle(nu0[i], nu1[i], nu2[i])
            nu_thetas.append(nu_theta)
            nu_phis.append(nu_phi)
            nu_costhetas.append(np.cos(nu_theta))
            nu_cosphis.append(np.cos(nu_phi))
            
            nu_numi_theta, nu_numi_phi = utils. get_angle(nu0[i], nu1[i], nu2[i],to_numi=True)
            nu_numi_thetas.append(nu_numi_theta)
            nu_numi_phis.append(nu_numi_phi)
            nu_numi_costhetas.append(np.cos(nu_numi_theta))
            nu_numi_cosphis.append(np.cos(nu_numi_phi))
            
        Enu.append(nu3[i]*1000)
        
    df["truth_nu_theta"] = nu_thetas
    df["truth_nu_theta_deg"] = np.array(nu_thetas)*180/3.14
    df["truth_nu_phi"] = nu_phis
    df["truth_nu_phi_deg"] = np.array(nu_phis)*180/3.14
    df["truth_cos_nu_theta"] = nu_costhetas
    df["truth_cos_nu_phi"] = nu_cosphis 
    
    df["truth_nu_numi_theta"] = np.array(nu_numi_thetas)*180/3.14
    df["truth_nu_numi_theta_deg"] = nu_numi_thetas
    df["truth_nu_numi_phi_deg"] = np.array(nu_numi_phis)*180/3.14
    df["truth_nu_numi_phi"] = nu_numi_phis
    df["truth_cos_nu_numi_theta"] = nu_numi_costhetas
    df["truth_cos_nu_numi_phi"] = nu_numi_cosphis 
    
    
    df["truth_Enu"] = Enu

    return df


def add_truth_muon_info(df):
    #Truth info for the muon from the pfeval tree
    mu0 = df["truth_muonMomentum[0]"].to_numpy()
    mu1 = df["truth_muonMomentum[1]"].to_numpy()
    mu2 = df["truth_muonMomentum[2]"].to_numpy()
    mu3 = df["truth_muonMomentum[3]"].to_numpy()
    mu_thetas = []
    mu_phis = []
    mu_costhetas = []
    mu_cosphis = []
    mu_thetas_numi = []
    mu_phis_numi = []
    mu_costhetas_numi = []
    mu_cosphis_numi = []
    Emuon = []
    for i in tqdm(range(len(mu3))):
        mu_theta = np.nan
        mu_phi = np.nan
        mu_theta_numi = np.nan
        mu_phi_numi = np.nan
        
        if(mu3[i]<=0): 
            mu_theta = np.nan
            mu_phi = np.nan
            mu_theta_numi = np.nan
            mu_phi_numi = np.nan
        else: 
            mu_theta, mu_phi = utils. get_angle(mu0[i], mu1[i], mu2[i])
            mu_theta_numi, mu_phi_numi = utils. get_angle(mu0[i], mu1[i], mu2[i], to_numi=True)
            
        mu_thetas.append(mu_theta)
        mu_phis.append(mu_phi)
        mu_costhetas.append(np.cos(mu_theta))
        mu_cosphis.append(np.cos(mu_phi))
        
        mu_thetas_numi.append(mu_theta_numi)
        mu_phis_numi.append(mu_phi_numi)
        mu_costhetas_numi.append(np.cos(mu_theta_numi))
        mu_cosphis_numi.append(np.cos(mu_phi_numi))
        
        Emuon.append(mu3[i]*1000-105.7)
        
    df["truth_muon_theta"] = mu_thetas
    df["truth_muon_phi"] = mu_phis
    df["truth_muon_theta_deg"] = np.array(mu_thetas)*180/3.14
    df["truth_muon_phi_deg"] = np.array(mu_phis)*180/3.14
    df["truth_muon_costheta"] = mu_costhetas
    
    df["truth_muon_cosphi_numi"] = mu_cosphis_numi 
    df["truth_muon_theta_numi"] = mu_thetas_numi
    df["truth_muon_phi_numi"] = mu_phis_numi
    df["truth_muon_theta_numi_deg"] = np.array(mu_thetas_numi)*180/3.14
    df["truth_muon_phi_numi_deg"] = np.array(mu_phis_numi)*180/3.14
    df["truth_muon_costheta_numi"] = mu_costhetas_numi
    df["truth_muon_cosphi_numi"] = mu_cosphis_numi 
    
    df["truth_Emuon"] = Emuon

    return df

def add_truth_proton_info(df):

    num_prim_protons = []
    num_prim_protons_th35 = []
    
    photon_thetas = []
    photon_phis = []
    photon_energies = []
    
    Enu = df["truth_nuEnergy"].to_numpy()
    ext = df["isEXT"].to_numpy()
    truth_nuPdgs = df["truth_nuPdg"].to_numpy()
    truth_pdgs = df["truth_pdg"].to_numpy()
    truth_mothers = df["truth_mother"].to_numpy()
    truth_startMomentums = df["truth_startMomentum"].to_numpy()
    
    num_protons_all = []
    num_protons_all_th35 = []
    
    proton_thetas = []
    proton_phis = []
    prim_proton_thetas = []
    prim_proton_phis = []
    
    proton_costhetas = []
    proton_cosphis = []
    prim_proton_costhetas = []
    prim_proton_cosphis = []
    
    max_p_energy = []
    sub_p_energy = []
    subsub_p_energy = []
    min_p_energy = []
    max_prim_p_energy = []
    sub_prim_p_energy = []
    subsub_prim_p_energy = []
    min_prim_p_energy = []
    
    sum_prim_p_energy = []
    
    max_prim_p_Px = []
    max_prim_p_Py = []
    max_prim_p_Pz = []
    
    for i in tqdm(range(len(Enu))):
            
        num_prim_proton = 0
        num_prim_proton_th35 = 0
        num_proton = 0
        num_proton_th35 = 0
    
        
        truth_pdg_list = truth_pdgs[i]
        truth_mother_list = truth_mothers[i]
        truth_startMomentum_list = truth_startMomentums[i]
        
        proton_theta = np.nan
        proton_phi = np.nan
        prim_proton_theta = np.nan
        prim_proton_phi = np.nan
        
        imax_p_energy = 0
        isub_p_energy = 0
        isubsub_p_energy = 0
        imin_p_energy = 0
        imax_prim_p_energy = 0
        isub_prim_p_energy = 0
        isubsub_prim_p_energy = 0
        imin_prim_p_energy = 0
    
        isum_prim_p_energy = 0
        
        imax_prim_p_Px = 0
        imax_prim_p_Py = 0
        imax_prim_p_Pz = 0
    

        for j in range(len(truth_startMomentum_list)):

            if abs(truth_pdg_list[j]) != 2212: continue

            if truth_mother_list[j] != 0: # non-primary

                if truth_startMomentum_list[j][3]-0.938272>imax_p_energy: 
                    isubsub_p_energy = isub_p_energy
                    isub_p_energy = imax_p_energy
                    imax_p_energy = truth_startMomentum_list[j][3]-0.938272
                    proton_theta, proton_phi = utils. get_angle(truth_startMomentum_list[j][0], truth_startMomentum_list[j][1], truth_startMomentum_list[j][2])
                elif truth_startMomentum_list[j][3]-0.938272>isub_p_energy: 
                    isubsub_p_energy = isub_p_energy
                    isub_p_energy = truth_startMomentum_list[j][3]-0.938272
                elif truth_startMomentum_list[j][3]-0.938272>isubsub_p_energy: 
                    isubsub_p_energy = truth_startMomentum_list[j][3]-0.938272 
                num_proton += 1
                if truth_startMomentum_list[j][3]-0.938272>0.035: num_proton_th35 += 1

            else:

                isum_prim_p_energy+=truth_startMomentum_list[j][3]-0.938272
                
                num_prim_proton += 1
                num_proton += 1
                if truth_startMomentum_list[j][3]-0.938272>0.035:
                    num_prim_proton_th35 += 1
                    num_proton_th35 += 1
                if truth_startMomentum_list[j][3]-0.938272>imax_p_energy: 
                    isubsub_p_energy = isub_p_energy
                    isub_p_energy = imax_p_energy
                    imax_p_energy = truth_startMomentum_list[j][3]-0.938272
                    proton_theta, proton_phi = utils. get_angle(truth_startMomentum_list[j][0], truth_startMomentum_list[j][1], truth_startMomentum_list[j][2])
                elif truth_startMomentum_list[j][3]-0.938272>isub_p_energy: 
                    isubsub_p_energy = isub_p_energy
                    isub_p_energy = truth_startMomentum_list[j][3]-0.938272
                elif truth_startMomentum_list[j][3]-0.938272>isubsub_p_energy: 
                    isubsub_p_energy = truth_startMomentum_list[j][3]-0.938272
                if truth_startMomentum_list[j][3]-0.938272>imax_prim_p_energy: 
                    isubsub_prim_p_energy = isub_prim_p_energy
                    isub_prim_p_energy = imax_prim_p_energy
                    imax_prim_p_energy = truth_startMomentum_list[j][3]-0.938272
                    prim_proton_theta, prim_proton_phi = utils. get_angle(truth_startMomentum_list[j][0], truth_startMomentum_list[j][1], truth_startMomentum_list[j][2])
                    imax_prim_p_Px = truth_startMomentum_list[j][0]
                    imax_prim_p_Py = truth_startMomentum_list[j][1]
                    imax_prim_p_Pz = truth_startMomentum_list[j][2]
                elif truth_startMomentum_list[j][3]-0.938272>isub_prim_p_energy: 
                    isubsub_prim_p_energy = isub_p_energy
                    isub_prim_p_energy = truth_startMomentum_list[j][3]-0.938272
                elif truth_startMomentum_list[j][3]-0.938272>isubsub_prim_p_energy: 
                    isubsub_prim_p_energy = truth_startMomentum_list[j][3]-0.938272

                if truth_startMomentum_list[j][3]-0.938272<imin_p_energy or imin_p_energy==0:
                    imin_p_energy = truth_startMomentum_list[j][3]-0.938272
                if truth_startMomentum_list[j][3]-0.938272<imin_prim_p_energy or imin_prim_p_energy==0:
                    imin_prim_p_energy = truth_startMomentum_list[j][3]-0.938272
                

    
        num_prim_protons.append(num_prim_proton)
        num_prim_protons_th35.append(num_prim_proton_th35)
        num_protons_all.append(num_proton)
        num_protons_all_th35.append(num_proton_th35)      
        
        proton_thetas.append(proton_theta)
        proton_phis.append(proton_phi)
        prim_proton_thetas.append(prim_proton_theta)
        prim_proton_phis.append(prim_proton_phi)
    
        proton_costhetas.append(np.cos(proton_theta))
        proton_cosphis.append(np.cos(proton_phi))
        prim_proton_costhetas.append(np.cos(prim_proton_theta))
        prim_proton_cosphis.append(np.cos(prim_proton_phi))
        
        max_p_energy.append(imax_p_energy*1000)
        sub_p_energy.append(isub_p_energy*1000)
        subsub_p_energy.append(isubsub_p_energy*1000)
        min_p_energy.append(imin_p_energy*1000)
        max_prim_p_energy.append(imax_prim_p_energy*1000)
        sub_prim_p_energy.append(isub_prim_p_energy*1000)
        subsub_prim_p_energy.append(isubsub_prim_p_energy*1000)
        min_prim_p_energy.append(imin_prim_p_energy*1000)

        sum_prim_p_energy.append(isum_prim_p_energy*1000)
        
        max_prim_p_Px.append(imax_prim_p_Px*1000)
        max_prim_p_Py.append(imax_prim_p_Py*1000)
        max_prim_p_Pz.append(imax_prim_p_Pz*1000)
    
            
    
    df["truth_num_prim_proton"] = num_prim_protons
    df["truth_num_prim_proton_th35"] = num_prim_protons_th35
    df["truth_num_proton"] = num_protons_all
    df["truth_num_proton_th35"] = num_protons_all_th35
    
    df["truth_p_energy"] = max_p_energy
    df["truth_sub_p_energy"] = sub_p_energy
    df["truth_subsub_p_energy"] = subsub_p_energy
    df["truth_min_p_energy"] = min_p_energy
    df["truth_prim_p_energy"] = max_prim_p_energy
    df["truth_sub_prim_p_energy"] = sub_prim_p_energy
    df["truth_subsub_prim_p_energy"] = subsub_prim_p_energy
    df["truth_min_prim_p_energy"] = min_prim_p_energy
    
    df["truth_sum_prim_p_energy"] = sum_prim_p_energy
    
    df["truth_proton_theta"] = proton_thetas
    df["truth_proton_phi"] = proton_phis
    df["truth_prim_proton_theta"] = prim_proton_thetas
    df["truth_prim_proton_phi"] = prim_proton_phis

    df["truth_proton_theta_deg"] = np.array(proton_thetas)*180/3.14
    df["truth_proton_phi_deg"] = np.array(proton_phis)*180/3.14
    df["truth_prim_proton_theta_deg"] = np.array(prim_proton_thetas)*180/3.14
    df["truth_prim_proton_phi_deg"] = np.array(prim_proton_phis)*180/3.14
    
    df["truth_proton_costheta"] = proton_costhetas
    df["truth_proton_cosphi"] = proton_cosphis
    df["truth_prim_proton_costheta"] = prim_proton_costhetas
    df["truth_prim_proton_cosphi"] = prim_proton_cosphis
    
    df["max_prim_p_Px"] = max_prim_p_Px
    df["max_prim_p_Py"] = max_prim_p_Py
    df["max_prim_p_Pz"] = max_prim_p_Pz

    return df



def add_truth_ssm_info(df):
    mu0 = df["truth_muonMomentum[0]"].to_numpy()*1000
    mu1 = df["truth_muonMomentum[1]"].to_numpy()*1000
    mu2 = df["truth_muonMomentum[2]"].to_numpy()*1000
    Emu = df["truth_muonMomentum[3]"].to_numpy()*1000
    
    p0 = df["max_prim_p_Px"].to_numpy()
    p1 = df["max_prim_p_Py"].to_numpy()
    p2 = df["max_prim_p_Pz"].to_numpy()
    Ep = df["truth_prim_p_energy"].to_numpy()
    Ep_tot = df["truth_sum_prim_p_energy"].to_numpy()
    
    k = np.sqrt(mu0*mu0+mu1*mu1+mu2*mu2)
    true_q = []
    true_Q2 = []
    true_sqrtQ2 = []
    true_nu = []
    true_pt = []
    true_pl = []
    true_cosMu = []
    true_angle = []
    true_W = []
    
    true_cosP = []
    true_angle_P = []
    true_cosMuP = []
    true_angle_MuP = []

    true_Em = []
    true_Eh = []
    true_KE = []
    true_KE_th35 = []

    absorber_dir = [0.33, 0.75, -0.59]

    for event in tqdm(range(len(k))):
        if Emu[event]>0:
            cos = absorber_dir[0]*mu0[event]+absorber_dir[1]*mu1[event]+absorber_dir[2]*mu2[event]
            cos = cos / np.sqrt(mu0[event]*mu0[event]+mu1[event]*mu1[event]+mu2[event]*mu2[event]) / np.sqrt(absorber_dir[0]*absorber_dir[0]+absorber_dir[1]*absorber_dir[1]+absorber_dir[2]*absorber_dir[2])
            angle = np.arccos(cos)
            true_cosMu.append(cos)
            true_angle.append(angle)
            true_q.append( np.sqrt(235.5**2+k[event]**2-2*235.5*k[event]*cos) )
            true_Q2.append( 2*235.5*(Emu[event]-k[event]*cos)-105.7*105.7 )
            true_sqrtQ2.append( np.sqrt(2*235.5*(Emu[event]-k[event]*cos)-105.7*105.7 ))
            true_nu.append(235.5-Emu[event])
            #true_pt.append(abs(np.sin(angle)*k[event]))
            #true_pl.append(abs(cos*k[event]))
            true_pt.append(np.sin(angle)*k[event])
            true_pl.append(cos*k[event])
            true_W.append( np.sqrt(938*938 + 2*938*true_nu[-1] - true_Q2[-1]) )
            Em = 235.5-Emu[event]-Ep_tot[event]
            true_Em.append(Em)
            true_Eh.append(Ep_tot[event])
            true_KE.append(Emu[event]+Ep[event]-105.7)
            if Ep[event]>35: true_KE_th35.append(Emu[event]+Ep[event]-105.7)
            else: true_KE_th35.append(Emu[event]-105.7)
        else: 
            true_q.append(-999)
            true_Q2.append(-999)
            true_sqrtQ2.append(-999)
            true_nu.append(-999)
            true_pt.append(-999)
            true_pl.append(-999)
            true_cosMu.append(-999)
            true_angle.append(-999)
            true_W.append(-999)
            true_Em.append(-999)
            true_Eh.append(-999)
            true_KE.append(-999)
            true_KE_th35.append(-999)
            
        if Ep[event]>0:
            cos = absorber_dir[0]*p0[event]+absorber_dir[1]*p1[event]+absorber_dir[2]*p2[event]
            cos = cos / np.sqrt(p0[event]*p0[event]+p1[event]*p1[event]+p2[event]*p2[event]) / np.sqrt(absorber_dir[0]*absorber_dir[0]+absorber_dir[1]*absorber_dir[1]+absorber_dir[2]*absorber_dir[2])
            angle = np.arccos(cos)
            true_cosP.append(cos)
            true_angle_P.append(angle)
            
            cos = mu0[event]*p0[event]+mu1[event]*p1[event]+mu2[event]*p2[event]
            cos = cos / np.sqrt(p0[event]*p0[event]+p1[event]*p1[event]+p2[event]*p2[event]) / np.sqrt(mu0[event]*mu0[event]+mu1[event]*mu1[event]+mu2[event]*mu2[event])
            angle = np.arccos(cos)
            true_cosMuP.append(cos)
            true_angle_MuP.append(angle)
        else:
            true_cosP.append(-999)
            true_angle_P.append(-999)
            true_cosMuP.append(-999)
            true_angle_MuP.append(-999)
        
    df["true_q"] = true_q
    df["true_Q2"] = true_Q2
    df["true_sqrtQ2"] = true_sqrtQ2
    df["true_nu"] = true_nu
    df["true_pl"] = true_pl
    df["true_pt"] = true_pt
    df["true_cosMu"] = true_cosMu
    df["true_angle"] = true_angle
    df["true_angle_deg"] = np.array(true_angle)*180/3.14
    df["true_W"] = true_W
    
    df["true_cosP"] = true_cosP
    df["true_angle_P"] = true_angle_P
    df["true_angle_P_deg"] = np.array(true_angle_P)*180/3.14
    df["true_cosMuP"] = true_cosMuP
    df["true_angle_MuP"] = true_angle_MuP
    df["true_angle_MuP_deg"] = np.array(true_angle_MuP)*180/3.14

    df["true_Em"] = true_Em
    df["true_Eh"] = true_Eh
    df["true_KE"] = true_KE
    df["true_KE_th35"] = true_KE_th35

    return df
    
            
def set_is_kdar(df):
    truth_nuEnergy = df["truth_nuEnergy"].to_numpy()
    truth_nu_theta = df["truth_nu_theta"].to_numpy()
    truth_nuPdg = df["truth_nuPdg"].to_numpy()
    truth_isCC = df["truth_isCC"].to_numpy()
    truth_vtxInside = df["truth_vtxInside"].to_numpy()
    
    is_KDAR = []
    
    for i in range(len(truth_isCC)):
        if truth_nuEnergy[i]<237 and truth_nuEnergy[i]>234 and truth_nu_theta[i]>105*3.14/180 and truth_nu_theta[i]<130*3.14/180 and truth_nuPdg[i]==14 and truth_isCC[i]==1 and truth_vtxInside[i]==1:
            is_KDAR.append(1)
        else: is_KDAR.append(0)
    
    
    df["is_KDAR"] = is_KDAR 

    return df



########## Reconstruction variable adding ###################
def add_ssm_kine_info(df):
    k = np.sqrt(df["ssm_kine_energy"].to_numpy()**2+2*df["ssm_kine_energy"].to_numpy()*105.7) 
    Emu = df["ssm_kine_energy"].to_numpy()+105.7
    cos = np.cos(df["ssm_angle_to_absorber"].to_numpy())
    angle = df["ssm_angle_to_absorber"].to_numpy()
    
    E_t_1 = df["ssm_prim_track1_kine_energy_range"].to_numpy()
    E_t_2 = df["ssm_prim_track2_kine_energy_range"].to_numpy()
    E_s_1 = df["ssm_prim_shw1_kine_energy_best"].to_numpy()
    E_s_2 = df["ssm_prim_shw2_kine_energy_best"].to_numpy()
    
    ssm_Em = []
    ssm_E = []
    ssm_E = []
    ssm_KE = []
    ssm_Eh = []

    ssm_q = []
    ssm_Q2 = []
    ssm_sqrtQ2 = []
    ssm_nu = []
    ssm_pt = []
    ssm_pl = []
    
    for event in tqdm(range(len(k))):
        E = 0
        if Emu[event]>0 and cos[event]>-1 and cos[event]<1:
            ssm_q.append( np.nan_to_num(np.sqrt(235.5**2+k[event]**2-2*235.5*k[event]*cos[event]),nan=-999) )
            ssm_Q2.append( 2*235.5*(Emu[event]-k[event]*cos[event])-105.7*105.7 )
            ssm_sqrtQ2.append( np.sqrt(2*235.5*(Emu[event]-k[event]*cos[event])-105.7*105.7 ))
            ssm_nu.append(235.5-Emu[event])
            #ssm_pt.append(abs(np.sin(angle[event])*k[event]))
            #ssm_pl.append(abs(cos[event]*k[event]))
            ssm_pt.append(np.sin(angle[event])*k[event])
            ssm_pl.append(cos[event]*k[event])
            E = Emu[event]
            if E_t_1[event]>0: E += E_t_1[event]
            ssm_KE.append(E-105.7)  
            if E_t_2[event]>0: E += E_t_2[event]
            if E_s_1[event]>0: E += E_s_1[event]
            if E_s_2[event]>0: E += E_s_2[event]
            ssm_Em.append(235.5-E)
            ssm_E.append(E)
            ssm_Eh.append(E-Emu[event])
            
        else: 
            ssm_q.append(-999)
            ssm_Q2.append(-999)
            ssm_sqrtQ2.append(-999)
            ssm_nu.append(-999)
            ssm_pt.append(-999)
            ssm_pl.append(-999)
            ssm_Em.append(-999)
            ssm_E.append(-999)
            ssm_Eh.append(-999)
            ssm_KE.append(-999)
        
    df["ssm_q"] = ssm_q
    df["ssm_Q2"] = ssm_Q2
    df["ssm_sqrtQ2"] = ssm_sqrtQ2
    df["ssm_nu"] = ssm_nu
    df["ssm_pl"] = ssm_pl
    df["ssm_pt"] = ssm_pt
    df["ssm_cosMu"] = cos
    df["ssm_angle_to_absorber_deg"] = df["ssm_angle_to_absorber"].to_numpy()*180/3.14
    df["ssm_Em"] = ssm_Em
    df["ssm_E"] = ssm_E
    df["ssm_Eh"] = ssm_Eh
    df["ssm_KE"] = ssm_KE

    return df


def add_ssm_reco_proton_info(df):
    m0 = df["ssm_x_dir"].to_numpy()
    m1 = df["ssm_y_dir"].to_numpy()
    m2 = df["ssm_z_dir"].to_numpy()
    
    
    p0 = df["ssm_prim_track1_x_dir"].to_numpy()
    p1 = df["ssm_prim_track1_y_dir"].to_numpy()
    p2 = df["ssm_prim_track1_z_dir"].to_numpy()
    Ep = df["ssm_prim_track1_kine_energy_range"].to_numpy()
    
    cosP = []
    angle_P = []
    
    cosMuP = []
    angle_MuP = []
    
    absorber_dir = [0.33, 0.75, -0.59]

    for event in tqdm(range(len(Ep))):
        if Ep[event]>0:
            cos = absorber_dir[0]*p0[event]+absorber_dir[1]*p1[event]+absorber_dir[2]*p2[event]
            cos = cos / np.sqrt(p0[event]*p0[event]+p1[event]*p1[event]+p2[event]*p2[event]) / np.sqrt(absorber_dir[0]*absorber_dir[0]+absorber_dir[1]*absorber_dir[1]+absorber_dir[2]*absorber_dir[2])
            angle = np.arccos(cos)
            cosP.append(cos)
            angle_P.append(angle)
            
            cos = m0[event]*p0[event]+m1[event]*p1[event]+m2[event]*p2[event]
            cos = cos / np.sqrt(p0[event]*p0[event]+p1[event]*p1[event]+p2[event]*p2[event]) / np.sqrt(m0[event]*m0[event]+m1[event]*m1[event]+m2[event]*m2[event])
            angle = np.arccos(cos)
            cosMuP.append(cos)
            angle_MuP.append(angle)
        else:
            cosP.append(-999)
            angle_P.append(-999)
            cosMuP.append(-999)
            angle_MuP.append(-999)
            
    df["ssm_cosP"] = cosP
    df["ssm_angle_P"] = angle_P
    df["ssm_angle_P_deg"] = np.array(angle_P)*180/3.14
    df["ssm_cosMuP"] = cosMuP
    df["ssm_angle_MuP"] = angle_MuP
    df["ssm_angle_MuP_deg"] = np.array(angle_MuP)*180/3.14
    
    return df


def add_reco_proton_muon(df):
    #get the muon and leading proton info out of the pfeval tree
    mu0 = df["reco_muonMomentum[0]"].to_numpy()
    mu1 = df["reco_muonMomentum[1]"].to_numpy()
    mu2 = df["reco_muonMomentum[2]"].to_numpy()
    mu3 = df["reco_muonMomentum[3]"].to_numpy()
    
    p0 = df["reco_protonMomentum[0]"].to_numpy()
    p1 = df["reco_protonMomentum[1]"].to_numpy()
    p2 = df["reco_protonMomentum[2]"].to_numpy()
    p3 = df["reco_protonMomentum[3]"].to_numpy()
    
    kine_reco_Enu = df["kine_reco_Enu"].to_numpy()
    
    mu_thetas = []
    mu_phis = []
    mu_costhetas = []
    mu_cosphis = []
    reco_Emuon = []
    
    p_thetas = []
    p_phis = []
    p_costhetas = []
    p_cosphis = []
    reco_Eproton = []
    
    Ehadron = []

    for i in tqdm(range(len(mu3))):
        if(mu3[i]<=0): 
            mu_theta = np.nan
            mu_phi = np.nan
            Ehadron.append(kine_reco_Enu[i])
        else: 
            mu_theta, mu_phi = utils. get_angle(mu0[i], mu1[i], mu2[i])
            Ehadron.append(kine_reco_Enu[i]-mu3[i]*1000)
        mu_thetas.append(mu_theta)
        mu_phis.append(mu_phi)
        mu_costhetas.append(np.cos(mu_theta))
        mu_cosphis.append(np.cos(mu_phi))
        reco_Emuon.append(mu3[i]*1000)
        
        if(p3[i]<=0): 
            p_theta = np.nan
            p_phi = np.nan
            reco_Eproton.append(0)
            
        else: 
            p_theta, p_phi = utils. get_angle(p0[i], p1[i], p2[i])
            reco_Eproton.append(p3[i]*1000-938.27)
        p_thetas.append(p_theta)
        p_phis.append(p_phi)
        p_costhetas.append(np.cos(p_theta))
        p_cosphis.append(np.cos(p_phi))

    
    df["reco_muon_theta"] = mu_thetas
    df["reco_muon_phi"] = mu_phis
    df["reco_muon_theta_deg"] = np.array(mu_thetas)*180/3.14
    df["reco_muon_phi_deg"] = np.array(mu_phis)*180/3.14
    df["reco_muon_costheta"] = mu_costhetas
    df["reco_muon_cosphi"] = mu_cosphis 
    df["reco_Emuon"] = reco_Emuon
    
    df["reco_proton_theta"] = p_thetas
    df["reco_proton_phi"] = p_phis
    df["reco_proton_theta"] = np.array(p_thetas)*180/3.14
    df["reco_proton_phi"] = np.array(p_phis)*180/3.14
    df["reco_proton_costheta"] = p_costhetas
    df["reco_proton_cosphi"] = p_cosphis 
    df["reco_Eproton"] = reco_Eproton
    
    df["Ehadron"] = Ehadron

    return df


def add_ssm_reco_dirt_vars(df):
    ssmsp_x = df["ssm_vtxX"].to_numpy()
    ssmsp_y = df["ssm_vtxY"].to_numpy()
    ssmsp_z = df["ssm_vtxZ"].to_numpy()
    ssm_x_dir = df["ssm_x_dir"].to_numpy()
    ssm_y_dir = df["ssm_y_dir"].to_numpy()
    ssm_z_dir = df["ssm_z_dir"].to_numpy()
    ssm_kine_energy = df["ssm_kine_energy"].to_numpy()
    
    tRV_list = []
    tfRV_list = []
    tbRV_list = []
    tRV_nox_list = []
    tfRV_nox_list = []
    tbRV_nox_list = []

    for event in tqdm(range(len(ssmsp_x))):
        if ssm_kine_energy[event]<0:
            tRV_list.append(-999999)
            tfRV_list.append(-999999)
            tbRV_list.append(-999999)
            tRV_nox_list.append(-999999)
            tfRV_nox_list.append(-999999)
            tbRV_nox_list.append(-999999)
            continue
        tRV = (ssm_x_dir[event]*(128.175-ssmsp_x[event])+ssm_y_dir[event]*(0-ssmsp_y[event])+ssm_z_dir[event]*(518.4-ssmsp_z[event]))    
        tRV_list.append(tRV)
        tfRV = (ssm_x_dir[event]*(128.175-ssmsp_x[event])+ssm_y_dir[event]*(0-ssmsp_y[event])+ssm_z_dir[event]*(0-ssmsp_z[event]))    
        tfRV_list.append(tfRV)
        tbRV = (ssm_x_dir[event]*(128.175-ssmsp_x[event])+ssm_y_dir[event]*(0-ssmsp_y[event])+ssm_z_dir[event]*(1036.8-ssmsp_z[event]))    
        tbRV_list.append(tbRV)
        
        tRV_nox = (ssm_y_dir[event]*(0-ssmsp_y[event])+ssm_z_dir[event]*(518.4-ssmsp_z[event]))    
        tRV_nox_list.append(tRV_nox)
        tfRV_nox = (ssm_y_dir[event]*(0-ssmsp_y[event])+ssm_z_dir[event]*(0-ssmsp_z[event]))    
        tfRV_nox_list.append(tfRV_nox)
        tbRV_nox = (ssm_y_dir[event]*(0-ssmsp_y[event])+ssm_z_dir[event]*(1036.8-ssmsp_z[event]))    
        tbRV_nox_list.append(tbRV_nox)    
        
    df["tRV"] = tRV_list
    df["tfRV"] = tfRV_list
    df["tbRV"] = tbRV_list
    df["tRV_nox"] = tRV_nox_list
    df["tfRV_nox"] = tfRV_nox_list
    df["tbRV_nox"] = tbRV_nox_list

    return df


def apply_goodruns(df):
    good_runs = [4952, 4953, 4954, 4955, 4957, 4958, 4961, 4962, 4966, 4967, 4968, 4969, 4971, 4974, 4975, 4977, 4978, 4979, 4981, 4982, 4983, 4986, 4987,
    4988, 4989, 4991, 4992, 4995, 4997, 4998, 4999, 5000, 5001, 5002, 5005, 5009, 5010, 5011, 5012, 5013, 5015, 5016, 5017, 5019, 5021, 5022, 5023, 5024,
    5025, 5027, 5029, 5031, 5034, 5036, 5037, 5038, 5039, 5041, 5042, 5044, 5046, 5047, 5048, 5049, 5051, 5055, 5056, 5059, 5060, 5061, 5062, 5064, 5065,
    5066, 5067, 5068, 5069, 5070, 5071, 5074, 5075, 5076, 5077, 5078, 5079, 5082, 5084, 5086, 5087, 5089, 5090, 5091, 5092, 5093, 5095, 5097, 5098, 5099,
    5100, 5102, 5103, 5104, 5106, 5108, 5109, 5110, 5114, 5121, 5122, 5124, 5125, 5127, 5128, 5130, 5133, 5134, 5135, 5136, 5137, 5138, 5139, 5142, 5143,
    5144, 5145, 5146, 5147, 5151, 5153, 5154, 5155, 5157, 5159, 5160, 5161, 5162, 5164, 5165, 5166, 5167, 5168, 5169, 5170, 5171, 5176, 5177, 5179, 5181,
    5182, 5183, 5184, 5185, 5187, 5189, 5190, 5191, 5192, 5194, 5195, 5197, 5198, 5201, 5203, 5204, 5205, 5207, 5208, 5211, 5212, 5213, 5214, 5215, 5216,
    5217, 5219, 5222, 5223, 5226, 5227, 5229, 5233, 5235, 5237, 5262, 5263, 5264, 5265, 5266, 5267, 5268, 5269, 5270, 5271, 5272, 5273, 5274, 5275, 5277,
    5278, 5279, 5280, 5281, 5315, 5320, 5321, 5322, 5326, 5328, 5329, 5330, 5331, 5332, 5333, 5334, 5337, 5338, 5339, 5340, 5341, 5343, 5344, 5345, 5347,
    5348, 5349, 5351, 5353, 5354, 5359, 5360, 5361, 5362, 5363, 5364, 5365, 5366, 5367, 5368, 5370, 5371, 5374, 5375, 5376, 5377, 5380, 5382, 5383, 5384,
    5385, 5386, 5387, 5388, 5389, 5390, 5391, 5392, 5393, 5394, 5395, 5396, 5397, 5399, 5401, 5403, 5404, 5408, 5409, 5411, 5412, 5413, 5415, 5417, 5418,
    5419, 5422, 5423, 5424, 5425, 5427, 5428, 5430, 5431, 5432, 5433, 5435, 5436, 5437, 5440, 5441, 5442, 5444, 5445, 5448, 5449, 5450, 5451, 5452, 5454,
    5455, 5456, 5457, 5458, 5459, 5460, 5462, 5463, 5464, 5465, 5466, 5470, 5474, 5476, 5478, 5480, 5482, 5484, 5485, 5487, 5488, 5489, 5490, 5491, 5492,
    5493, 5495, 5497, 5498, 5499, 5500, 5501, 5504, 5506, 5507, 5508, 5509, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522,
    5523, 5524, 5525, 5526, 5527, 5528, 5530, 5531, 5532, 5533, 5535, 5536, 5538, 5539, 5540, 5541, 5544, 5545, 5546, 5547, 5553, 5555, 5557, 5561, 5564,
    5565, 5566, 5567, 5568, 5569, 5570, 5573, 5574, 5575, 5576, 5577, 5578, 5579, 5581, 5582, 5583, 5584, 5585, 5586, 5587, 5588, 5589, 5593, 5597, 5598,
    5600, 5601, 5602, 5603, 5604, 5605, 5606, 5607, 5608, 5609, 5611, 5614, 5616, 5617, 5618, 5619, 5622, 5623, 5624, 5625, 5627, 5628, 5630, 5632, 5634,
    5635, 5636, 5637, 5638, 5639, 5643, 5646, 5647, 5650, 5652, 5653, 5654, 5656, 5657, 5659, 5661, 5680, 5684, 5685, 5686, 5691, 5693, 5694, 5695, 5697,
    5698, 5699, 5702, 5703, 5704, 5705, 5706, 5707, 5708, 5709, 5710, 5712, 5713, 5715, 5718, 5719, 5720, 5721, 5722, 5723, 5724, 5725, 5726, 5727, 5728,
    5729, 5730, 5731, 5733, 5735, 5739, 5740, 5741, 5743, 5745, 5746, 5748, 5749, 5752, 5753, 5754, 5755, 5756, 5758, 5760, 5761, 5762, 5765, 5766, 5767,
    5768, 5769, 5771, 5772, 5773, 5774, 5776, 5777, 5778, 5779, 5781, 5782, 5783, 5891, 5892, 5894, 5895, 5896, 5897, 5899, 5900, 5901, 5904, 5905, 5906,
    5908, 5909, 5910, 5911, 5912, 5914, 5915, 5916, 5918, 5919, 5920, 5921, 5922, 5923, 5924, 5925, 5926, 5929, 5930, 5931, 5932, 5933, 5934, 5935, 5936,
    5937, 5938, 5940, 5941, 5942, 5946, 5947, 5948, 5949, 5952, 5953, 5956, 5957, 5959, 5960, 5961, 5963, 5964, 5965, 5966, 5968, 5969, 5971, 5975, 5976,
    5977, 5979, 5982, 5983, 5984, 5985, 5986, 5987, 5988, 5989, 5990, 5993, 5994, 5996, 5998, 6000, 6001, 6002, 6003, 6004, 6007, 6011, 6012, 6021, 6022,
    6023, 6024, 6025, 6026, 6027, 6028, 6030, 6031, 6032, 6035, 6036, 6037, 6041, 6043, 6044, 6045, 6046, 6047, 6050, 6052, 6055, 6056, 6058, 6059, 6060,
    6063, 6064, 6065, 6070, 6072, 6073, 6074, 6075, 6076, 6078, 6079, 6080, 6081, 6082, 6083, 6084, 6085, 6086, 6089, 6090, 6091, 6092, 6093, 6094, 6095,
    6096, 6098, 6099, 6100, 6101, 6102, 6103, 6105, 6106, 6107, 6108, 6110, 6111, 6113, 6114, 6115, 6117, 6118, 6119, 6120, 6121, 6122, 6123, 6125, 6130,
    6134, 6138, 6139, 6140, 6141, 6143, 6144, 6145, 6146, 6147, 6148, 6149, 6151, 6153, 6154, 6155, 6156, 6157, 6158, 6159, 6160, 6161, 6162, 6163, 6164,
    6165, 6166, 6168, 6170, 6172, 6173, 6176, 6178, 6179, 6180, 6182, 6183, 6184, 6185, 6187, 6190, 6191, 6192, 6194, 6195, 6197, 6198, 6199, 6201, 6203,
    6205, 6206, 6207, 6210, 6211, 6213, 6214, 6216, 6217, 6218, 6219, 6220, 6221, 6223, 6224, 6226, 6227, 6228, 6229, 6230, 6231, 6233, 6234, 6235, 6236,
    6238, 6239, 6241, 6244, 6246, 6247, 6254, 6260, 6261, 6262, 6265, 6266, 6276, 6277, 6278, 6279, 6280, 6281, 6282, 6283, 6284, 6285, 6286, 6288, 6295,
    6296, 6297, 6298, 6299, 6300, 6301, 6302, 6303, 6304, 6305, 6307, 6308, 6309, 6310, 6311, 6312, 6313, 6314, 6315, 6318, 6319, 6320, 6321, 6322, 6323,
    6324, 6325, 6326, 6327, 6329, 6330, 6332, 6333, 6334, 6335, 6336, 6337, 6338, 6339, 6340, 6341, 6342, 6343, 6344, 6346, 6347, 6348, 6349, 6350, 6352,
    6353, 6354, 6355, 6356, 6358, 6359, 6360, 6361, 6362, 6363, 6365, 6366, 6367, 6369, 6370, 6371, 6374, 6375, 6376, 6377, 6378, 6379, 6380, 6381, 6382,
    6383, 6384, 6385, 6386, 6387, 6388, 6397, 6401, 6402, 6404, 6405, 6406, 6407, 6408, 6409, 6410, 6412, 6417, 6419, 6420, 6421, 6423, 6425, 6426, 6427,
    6428, 6429, 6430, 6431, 6432, 6433, 6436, 6437, 6438, 6439, 6440, 6441, 6442, 6443, 6444, 6445, 6446, 6448, 6450, 6451, 6453, 6454, 6455, 6457, 6460,
    6461, 6462, 6466, 6467, 6468, 6475, 6478, 6479, 6480, 6481, 6482, 6483, 6484, 6487, 6488, 6489, 6490, 6493, 6494, 6497, 6501, 6502, 6503, 6506, 6507,
    6508, 6510, 6517, 6520, 6521, 6525, 6527, 6528, 6529, 6530, 6533, 6534, 6535, 6540, 6542, 6544, 6546, 6547, 6548, 6549, 6551, 6552, 6553, 6559, 6560,
    6562, 6563, 6564, 6565, 6566, 6567, 6568, 6569, 6571, 6572, 6573, 6575, 6577, 6579, 6580, 6581, 6582, 6583, 6588, 6589, 6590, 6591, 6592, 6593, 6594,
    6596, 6598, 6599, 6600, 6602, 6603, 6605, 6606, 6608, 6609, 6611, 6614, 6615, 6616, 6617, 6619, 6620, 6621, 6622, 6623, 6625, 6626, 6628, 6629, 6630,
    6631, 6632, 6633, 6634, 6635, 6637, 6638, 6639, 6640, 6641, 6642, 6645, 6647, 6649, 6650, 6653, 6655, 6656, 6657, 6658, 6659, 6663, 6664, 6665, 6666,
    6667, 6668, 6669, 6670, 6671, 6672, 6674, 6675, 6676, 6680, 6681, 6683, 6684, 6685, 6686, 6687, 6688, 6689, 6690, 6691, 6692, 6693, 6694, 6695, 6696,
    6697, 6698, 6699, 6710, 6711, 6712, 6714, 6717, 6720, 6722, 6725, 6747, 6748, 6749, 6751, 6752, 6756, 6758, 6761, 6763, 6765, 6766, 6767, 6768, 6769,
    6772, 6773, 6775, 6776, 6777, 6778, 6779, 6780, 6782, 6783, 6784, 6785, 6786, 6787, 6788, 6789, 6790, 6792, 6793, 6794, 6795, 6797, 6798, 6799, 6800,
    6801, 6802, 6803, 6804, 6805, 6806, 6810, 6811, 6812, 6813, 6814, 6822, 6823, 6824, 6825, 6826, 6827, 6828, 6829, 6830, 6831, 6832, 6835, 6836, 6837,
    6840, 6841, 6850, 6851, 6853, 6854, 6855, 6856, 6857, 6859, 6861, 6863, 6865, 6866, 6867, 6869, 6870, 6871, 6873, 6874, 6875, 6876, 6879, 6880, 6881,
    6882, 6883, 6884, 6885, 6886, 6887, 6888, 6890, 6891, 6893, 6894, 6895, 6896, 6901, 6902, 6903, 6905, 6906, 6907, 6908, 6911, 6912, 6913, 6914, 6915,
    6916, 6917, 6918, 6919, 6921, 6922, 6923, 6925, 6926, 6928, 6930, 6932, 6933, 6934, 6935, 6936, 6937, 6938, 6939, 6940, 6942, 6944, 6945, 6946, 6947,
    6949, 6950, 6951, 6952, 6953, 6954, 6955, 6956, 6957, 6958, 6959, 6960, 6963, 6964, 6965, 6966, 6967, 6968, 6969, 6970, 6971, 6972, 6973, 6974, 6975,
    6976, 6977, 6978, 6979, 6980, 6981, 6982, 6983, 6984, 6985, 6986, 6987, 6989, 6990, 6991, 6992, 6993, 6995, 6997, 6998, 6999, 7001, 7003, 7004, 7006,
    7007, 7008, 7010, 7011, 7012, 7013, 7014, 7016, 7017, 7018, 7019, 7020, 7021, 7022, 7023, 7025, 7026, 7046, 7047, 7048, 7049, 7051, 7053, 7054, 7055,
    7057, 7058, 7059, 7078, 7079, 7080, 7082, 7084, 7085, 7086, 7090, 7093, 7095, 7096, 7097, 7098, 7099, 7100, 7103, 7105, 7107, 7108, 7110, 7112, 7113,
    7114, 7115, 7116, 7117, 7123, 7124, 7125, 7126, 7127, 7128, 7129, 7130, 7131, 7132, 7133, 7134, 7135, 7136, 7139, 7141, 7142, 7143, 7144, 7145, 7146,
    7165, 7166, 7167, 7279, 7280, 7282, 7283, 7285, 7287, 7288, 7289, 7290, 7291, 7292, 7293, 7294, 7295, 7297, 7298, 7299, 7301, 7302, 7303, 7305, 7306,
    7307, 7308, 7309, 7310, 7311, 7312, 7314, 7315, 7318, 7319, 7320, 7322, 7323, 7324, 7326, 7327, 7328, 7331, 7332, 7333, 7334, 7336, 7337, 7338, 7341,
    7343, 7344, 7345, 7346, 7349, 7350, 7351, 7352, 7353, 7354, 7355, 7356, 7357, 7359, 7360, 7361, 7362, 7365, 7368, 7369, 7371, 7372, 7373, 7375, 7376,
    7377, 7378, 7380, 7381, 7382, 7383, 7384, 7385, 7386, 7387, 7388, 7389, 7390, 7391, 7392, 7393, 7394, 7397, 7398, 7400, 7401, 7403, 7404, 7406, 7407,
    7408, 7409, 7411, 7412, 7413, 7414, 7415, 7416, 7417, 7418, 7419, 7420, 7421, 7422, 7426, 7427, 7429, 7430, 7431, 7432, 7434, 7435, 7436, 7437, 7438,
    7439, 7440, 7441, 7442, 7443, 7444, 7445, 7455, 7456, 7457, 7458, 7459, 7461, 7462, 7463, 7464, 7465, 7466, 7468, 7469, 7470, 7471, 7473, 7474, 7476,
    7477, 7478, 7479, 7480, 7481, 7482, 7483, 7484, 7485, 7486, 7487, 7488, 7490, 7491, 7492, 7493, 7494, 7495, 7496, 7497, 7498, 7499, 7500, 7501, 7502,
    7503, 7504, 7505, 7506, 7507, 7508, 7509, 7510, 7512, 7513, 7514, 7515, 7516, 7517, 7518, 7519, 7520, 7521, 7522, 7523, 7524, 7525, 7526, 7527, 7528,
    7529, 7531, 7532, 7534, 7535, 7537, 7538, 7539, 7540, 7542, 7543, 7544, 7546, 7547, 7556, 7557, 7558, 7559, 7560, 7561, 7562, 7563, 7564, 7565, 7567,
    7568, 7571, 7573, 7574, 7575, 7576, 7577, 7578, 7580, 7583, 7584, 7586, 7587, 7588, 7611, 7623, 7625, 7626, 7627, 7628, 7631, 7633, 7634, 7635, 7636,
    7637, 7638, 7639, 7640, 7641, 7642, 7643, 7644, 7645, 7648, 7649, 7650, 7652, 7653, 7654, 7655, 7656, 7657, 7660, 7662, 7663, 7665, 7666, 7667, 7668,
    7669, 7670, 7671, 7673, 7674, 7675, 7676, 7677, 7678, 7680, 7681, 7682, 7683, 7685, 7687, 7688, 7689, 7690, 7691, 7693, 7694, 7695, 7696, 7699, 7700,
    7704, 7705, 7707, 7708, 7709, 7710, 7711, 7712, 7713, 7714, 7715, 7716, 7717, 7718, 7719, 7720, 7721, 7722, 7723, 7724, 7725, 7726, 7727, 7728, 7729,
    7730, 7731, 7736, 7738, 7739, 7740, 7741, 7742, 7744, 7745, 7746, 7747, 7748, 7749, 7750, 7751, 7753, 7754, 7755, 7756, 7757, 7759, 7760, 7761, 7762,
           7764, 7765, 7766, 7767, 7768, 7769, 7770, 
           13697, 13698, 13699, 13700, 13701, 13702, 13705, 13707, 13708, 13709, 13712, 13713, 13714, 13715, 13717, 13719, 13720, 13721, 13723, 13724,
    13725, 13728, 13729, 13730, 13731, 13732, 13733, 13736, 13737, 13738, 13739, 13740, 13741, 13742, 13743, 13744, 13745, 13746, 13755, 13756, 13757,
    13762, 13763, 13764, 13765, 13766, 13767, 13769, 13770, 13771, 13772, 13773, 13774, 13775, 13776, 13777, 13779, 13780, 13782, 13787, 13789, 13790,
    13791, 13792, 13793, 13794, 13797, 13799, 13800, 13801, 13802, 13803, 13805, 13806, 13807, 13809, 13810, 13811, 13812, 13860, 13861, 13862, 13864,
    13865, 13867, 13868, 13869, 13871, 13872, 13873, 13874, 13875, 13876, 13877, 13878, 13879, 13883, 13884, 13888, 13889, 13894, 13895, 13896, 13897,
    13898, 13899, 13900, 13902, 13904, 13905, 13906, 13907, 13908, 13909, 13910, 13911, 13912, 13913, 13917, 13918, 13919, 13921, 13922, 13923, 13926,
    13928, 13929, 13930, 13935, 13936, 13937, 13938, 13939, 13940, 13942, 13943, 13944, 13945, 13947, 13948, 13950, 13951, 13954, 13955, 13956, 13957,
    13959, 13960, 13962, 13963, 13964, 13965, 13969, 13970, 13971, 13972, 13977, 13978, 13979, 13982, 13984, 13985, 13986, 13988, 13989, 13990, 13991,
    13992, 13993, 13996, 13997, 13998, 14000, 14001, 14002, 14004, 14006, 14007, 14008, 14009, 14010, 14011, 14012, 14014, 14015, 14016, 14017, 14020,
    14021, 14022, 14023, 14024, 14025, 14026, 14028, 14030, 14031, 14032, 14034, 14035, 14036, 14038, 14039, 14040, 14042, 14043, 14044, 14045, 14046,
    14047, 14049, 14053, 14055, 14056, 14058, 14059, 14060, 14062, 14063, 14064, 14065, 14066, 14067, 14068, 14069, 14073, 14074, 14075, 14076, 14077,
    14079, 14080, 14082, 14083, 14084, 14087, 14088, 14089, 14090, 14091, 14093, 14095, 14096, 14097, 14098, 14099, 14101, 14102, 14103, 14104, 14105,
    14107, 14108, 14109, 14110, 14111, 14112, 14114, 14115, 14116, 14117, 14121, 14122, 14123, 14124, 14125, 14127, 14128, 14129, 14131, 14132, 14133,
    14134, 14135, 14136, 14138, 14139, 14140, 14141, 14142, 14143, 14146, 14147, 14148, 14149, 14150, 14151, 14152, 14153, 14154, 14156, 14157, 14158,
    14159, 14161, 14162, 14163, 14164, 14165, 14171, 14172, 14173, 14174, 14175, 14177, 14181, 14182, 14183, 14184, 14185, 14186, 14187, 14188, 14189,
    14190, 14191, 14192, 14193, 14194, 14195, 14196, 14201, 14202, 14203, 14205, 14206, 14207, 14209, 14211, 14212, 14213, 14214, 14216, 14217, 14218,
    14219, 14220, 14222, 14223, 14224, 14225, 14226, 14227, 14228, 14229, 14231, 14232, 14233, 14234, 14235, 14238, 14239, 14240, 14241, 14242, 14243,
    14244, 14245, 14246, 14248, 14250, 14251, 14253, 14256, 14257, 14259, 14260, 14261, 14262, 14263, 14264, 14265, 14266, 14268, 14269, 14270, 14271,
    14272, 14274, 14275, 14276, 14277, 14278, 14279, 14280, 14281, 14282, 14283, 14284, 14285, 14286, 14287, 14290, 14291, 14292, 14293, 14294, 14295,
    14297, 14298, 14299, 14300, 14301, 14302, 14306, 14307, 14308, 14310, 14313, 14314, 14315, 14316, 14317, 14319, 14320, 14321, 14323, 14324, 14325,
    14327, 14328, 14329, 14332, 14333, 14334, 14335, 14336, 14337, 14338, 14340, 14341, 14342, 14343, 14344, 14347, 14349, 14350, 14351, 14355, 14356,
    14357, 14358, 14359, 14360, 14361, 14362, 14363, 14366, 14367, 14368, 14369, 14370, 14372, 14375, 14376, 14377, 14379, 14381, 14382, 14383, 14384,
    14385, 14388, 14390, 14391, 14392, 14393, 14394, 14396, 14399, 14401, 14403, 14404, 14406, 14407, 14408, 14409, 14410, 14411, 14412, 14413, 14414,
    14417, 14418, 14420, 14421, 14423, 14425, 14427, 14428, 14430, 14431, 14432, 14436, 14438, 14439, 14441, 14443, 14445, 14446, 14450, 14451, 14452,
    14453, 14454, 14455, 14456, 14458, 14459, 14460, 14461, 14463, 14464, 14467, 14468, 14469, 14470, 14473, 14475, 14478, 14480, 14481, 14488, 14490,
    14497, 14500, 14505, 14509, 14510, 14512, 14516, 14517, 14521, 14524, 14526, 14527, 14529, 14530, 14531, 14534, 14535, 14536, 14537, 14538, 14539,
    14541, 14542, 14543, 14544, 14545, 14546, 14548, 14549, 14550, 14551, 14552, 14553, 14554, 14555, 14556, 14557, 14558, 14559, 14560, 14561, 14562,
    14563, 14564, 14568, 14569, 14571, 14572, 14573, 14575, 14576, 14577, 14578, 14580, 14582, 14583, 14584, 14585, 14586, 14587, 14590, 14592, 14596,
    14597, 14598, 14600, 14601, 14603, 14607, 14609, 14610, 14611, 14613, 14614, 14615, 14617, 14618, 14619, 14621, 14622, 14623, 14625, 14626, 14630,
    14631, 14632, 14633, 14634, 14635, 14636, 14637, 14639, 14640, 14641, 14642, 14643, 14644, 14645, 14647, 14664, 14667, 14669, 14671, 14672, 14673,
    14675, 14676, 14677, 14681, 14682, 14684, 14685, 14686, 14687, 14688, 14689, 14690, 14692, 14693, 14694, 14695, 14697, 14698, 14699, 14700, 14701,
    14702, 14704, 14706, 14707, 14708, 14709, 14710, 14712, 14713, 14715, 14716, 14717, 14719, 14720, 14721, 14722, 14724, 14725, 14726, 14727, 14729,
    14730, 14731, 14733, 14734, 14736, 14737, 14738, 14739, 14740, 14741, 14742, 14743, 14745, 14746, 14747, 14748, 14749, 14750, 14751, 14754, 14755,
    14756, 14758, 14759, 14760, 14761, 14763, 14764, 14766, 14767, 14768, 14770, 14771, 14772, 14773, 14774, 14775, 14779, 14780, 14782, 14783, 14784,
    14785, 14787, 14788, 14789, 14790, 14793, 14794, 14795, 14796, 14797, 14798, 14799, 14801, 14802, 14805, 14806, 14807, 14808, 14810, 14811, 14812,
    14813, 14814, 14816, 14817, 14818, 14819, 14820, 14822, 14824, 14825, 14826, 14828, 14829, 14830, 14831, 14852, 14857, 14860, 14866, 14870, 14871,
    14872, 14873, 14874, 14875, 14876, 14877, 14878, 14880, 14881, 14882, 14883, 14884, 14885, 14886, 14887, 14888, 14890, 14891, 14893, 14897, 14899,
    14900, 14901, 14902, 14903, 14904, 14905, 14907, 14908, 14909, 14911, 14912, 14913, 14914, 14915, 14916, 14917, 14918, 14919, 14920, 14921, 14922,
    14923, 14924, 14925, 14926, 14927, 14929, 14930, 14931, 14932, 14934, 14935, 14936, 14937, 14938, 14939, 14940, 14942, 14944, 14945, 14947, 14948,
    14949, 14954, 14955, 14956, 14957, 14958, 14959, 14965, 14967, 14968, 14970, 14971, 14972, 14973, 14974, 14975, 14976, 14977, 14978, 14979, 14980,
    14981, 14983, 14985, 14986, 14987, 14989, 14990, 14991, 14993, 14994, 14995, 14997, 14998, 15000, 15001, 15002, 15003, 15004, 15006, 15010, 15011,
    15012, 15013, 15014, 15016, 15017, 15018, 15019, 15021, 15025, 15027, 15031, 15035, 15042, 15043, 15045, 15047, 15051, 15052, 15053, 15054, 15055,
    15057, 15060, 15061, 15062, 15064, 15065, 15073, 15074, 15077, 15078, 15079, 15080, 15081, 15082, 15083, 15086, 15090, 15092, 15093, 15097, 15098,
    15099, 15100, 15101, 15102, 15103, 15104, 15106, 15111, 15113, 15114, 15115, 15116, 15117, 15118, 15120, 15121, 15122, 15123, 15124, 15125, 15127,
    15128, 15129, 15130, 15133, 15135, 15136, 15138, 15140, 15141, 15144, 15145, 15146, 15147, 15148, 15149, 15150, 15151, 15152, 15153, 15154, 15155,
    15156, 15160, 15161, 15162, 15163, 15165, 15166, 15167, 15170, 15172, 15174, 15176, 15177, 15179, 15181, 15182, 15183, 15206, 15222, 15223, 15225,
    15227, 15228, 15231, 15234, 15236, 15238, 15242, 15243, 15244, 15246, 15247, 15249, 15250, 15251, 15252, 15254, 15256, 15257, 15259, 15261, 15262,
    15264, 15265, 15266, 15267, 15270, 15271, 15273, 15274, 15275, 15276, 15277, 15279, 15281, 15282, 15283, 15286, 15287, 15288, 15289, 15290, 15291,
    15292, 15293, 15294, 15295, 15296, 15297, 15299, 15301, 15303, 15305, 15306, 15308, 15309, 15311, 15312, 15314, 15315, 15316, 15318, 15319, 15320,
    15322, 15323, 15324, 15326, 15327, 15328, 15330, 15334, 15335, 15336, 15337, 15339, 15341, 15342, 15343, 15345, 15346, 15348, 15350, 15351, 15353,
    15354, 15355, 15356, 15358, 15360, 15361, 15362, 15364, 15366, 15367, 15368, 15369, 15370, 15376, 15377, 15378, 15382, 15383, 15386, 15387, 15388,
    15389, 15391, 15392, 15393, 15394, 15395, 15396, 15397, 15398, 15399, 15402, 15403, 15407, 15409, 15410, 15411, 15412, 15413, 15415, 15416, 15417,
    15419, 15420, 15421, 15422, 15423, 15425, 15426, 15427, 15428, 15429, 15430, 15431, 15432, 15433, 15434, 15436, 15437, 15438, 15439, 15440, 15442,
    15443, 15445, 15446, 15447, 15448, 15449, 15450, 15451, 15456, 15457, 15458, 15460, 15461, 15462, 15463, 15464, 15465, 15466, 15467, 15469, 15470,
    15471, 15473, 15474, 15477, 15478, 15479, 15481, 15482, 15483, 15484, 15485, 15486, 15488, 15489, 15491, 15492, 15494, 15495, 15496, 15497, 15498,
    15499, 15500, 15501, 15502, 15503, 15504, 15505, 15507, 15509, 15510, 15511, 15514, 15516, 15517, 15519, 15520, 15521, 15527, 15528, 15529, 15530,
    15550, 15554, 15580, 15587, 15588, 15589, 15590, 15591, 15592, 15593, 15595, 15596, 15597, 15599, 15601, 15603, 15632, 15633, 15634, 15636, 15637,
    15638, 15639, 15640, 15641, 15643, 15644, 15647, 15652, 15655, 15656, 15657, 15662, 15667, 15668, 15669, 15670, 15673, 15676, 15677, 15678, 15680,
    15681, 15682, 15683, 15687, 15688, 15689, 15692, 15696, 15697, 15698, 15700, 15701, 15702, 15703, 15704, 15706, 15708, 15709, 15710, 15713, 15714,
    15715, 15718, 15719, 15720, 15721, 15722, 15724, 15727, 15728, 15729, 15730, 15731, 15733, 15734, 15737, 15738, 15741, 15742, 15748, 15751, 15753,
    15757, 15762, 15764, 15766, 15767, 15771, 15773, 15774, 15775, 15776, 15777, 15778, 15779, 15780, 15782, 15783, 15784, 15785, 15789, 15790, 15792,
    15794, 15797, 15799, 15800, 15801, 15803, 15805, 15806, 15807, 15808, 15822, 15824, 15825, 15827, 15828, 15829, 15832, 15834, 15836, 15838, 15840,
    15843, 15844, 15846, 15851, 15857, 15858, 15859, 15861, 15862, 15863, 15865, 15867, 15870, 15871, 15872, 15873, 15875, 15876, 15877, 15878, 15880,
    15882, 15883, 15885, 15886, 15888, 15890, 15892, 15893, 15894, 15895, 15897, 15898, 15900, 15901, 15902, 15903, 15904, 15907, 15909, 15910, 15913,
    15915, 15916, 15917, 15918, 15919, 15920, 15921, 15922, 15923, 15924, 15926, 15927, 15928, 15930, 15931, 15932, 15933, 15935, 15939, 15940, 15941,
    15943, 15946, 15948, 15949, 15952, 15954, 15955, 15957, 15959, 15960, 15961, 15963, 15965, 15966, 15967, 15969, 15970, 15973, 15975, 15976, 15978,
    15979, 15980, 15981, 15983, 15986, 15987, 15988, 15989, 15990, 15993, 15997, 15998, 15999, 16002, 16004, 16006, 16008, 16011, 16013, 16017, 16018,
    16021, 16022, 16023, 16024, 16026, 16028, 16031, 16034, 16036, 16038, 16039, 16040, 16041, 16042, 16043, 16047, 16048, 16051, 16052, 16057, 16058,
    16059, 16060, 16061, 16063, 16064, 16065, 16066, 16067, 16069, 16071, 16072, 16073, 16075, 16076, 16078, 16079, 16080, 16081, 16088, 16089, 16090,
    16091, 16092, 16094, 16095, 16096, 16099, 16101, 16102, 16104, 16105, 16106, 16107, 16108, 16109, 16110, 16112, 16113, 16114, 16115, 16116, 16117,
    16118, 16119, 16121, 16122, 16125, 16126, 16134, 16137, 16138, 16140, 16141, 16142, 16144, 16145, 16147, 16148, 16150, 16151, 16152, 16155, 16158,
    16159, 16160, 16161, 16162, 16166, 16167, 16168, 16169, 16171, 16172, 16173, 16175, 16176, 16178, 16179, 16180, 16184, 16185, 16186, 16187, 16188,
    16191, 16196, 16197, 16198, 16199, 16200, 16203, 16204, 16205, 16206, 16207, 16208, 16209, 16210, 16211, 16212, 16214, 16215, 16216, 16218, 16219,
    16220, 16221, 16222, 16225, 16226, 16227, 16228, 16230, 16231, 16232, 16239, 16242, 16243, 16247, 16250, 16252, 16253, 16254, 16256, 16257, 16258,
    16260, 16262, 16263, 16265, 16266, 16267, 16270, 16271, 16272, 16274, 16275, 16276, 16277, 16278, 16279, 16280, 16281, 16283, 16285, 16286, 16288,
    16289, 16290, 16292, 16293, 16295, 16298, 16299, 16301, 16303, 16304, 16305, 16307, 16310, 16312, 16313, 16314, 16315, 16317, 16319, 16320, 16322,
    16323, 16325, 16326, 16329, 16330, 16331, 16332, 16333, 16335, 16336, 16337, 16338, 16339, 16340, 16341, 16344, 16346, 16347, 16348, 16350, 16352,
    16355, 16356, 16357, 16358, 16363, 16364, 16370, 16374, 16376, 16378, 16379, 16381, 16382, 16384, 16385, 16386, 16387, 16389, 16391, 16392, 16394,
    16396, 16403, 16404, 16405, 16406, 16408, 16410, 16411, 16414, 16415, 16416, 16420, 16422, 16424, 16425, 16426, 16431, 16432, 16433, 16434, 16437,
    16438, 16439, 16440, 16442, 16444, 16445, 16446, 16448, 16451, 16453, 16457, 16458, 16459, 16462, 16464, 16466, 16467, 16468, 16469, 16470, 16471,
    16472, 16473, 16476, 16477, 16479, 16480, 16481, 16483, 16485, 16486, 16489, 16491, 16492, 16493, 16494, 16495, 16497, 16498, 16499, 16504, 16506,
    16507, 16508, 16509, 16510, 16511, 16513, 16514, 16515, 16517, 16518, 16519, 16520, 16521, 16522, 16524, 16525, 16526, 16527, 16529, 16530, 16532,
    16536, 16540, 16542, 16544, 16545, 16546, 16547, 16548, 16549, 16551, 16552, 16553, 16554, 16555, 16557, 16558, 16562, 16563, 16565, 16568, 16569,
    16570, 16571, 16573, 16576, 16578, 16579, 16581, 16583, 16584, 16585, 16586, 16587, 16588, 16593, 16594, 16645, 16650, 16659, 16664, 16666, 16668,
    16670, 16672, 16676, 16679, 16680, 16683, 16684, 16685, 16687, 16689, 16690, 16692, 16693, 16694, 16696, 16697, 16724, 16726, 16730, 16731, 16733,
    16734, 16735, 16736, 16738, 16739, 16741, 16742, 16743, 16744, 16746, 16747, 16749, 16750, 16751, 16752, 16753, 16757, 16759, 16760, 16761, 16762,
    16763, 16764, 16765, 16769, 16770, 16772, 16776, 16778, 16782, 16783, 16785, 16786, 16787, 16788, 16789, 16793, 16794, 16795, 16797, 16798, 16801,
    16802, 16803, 16804, 16805, 16807, 16808, 16809, 16813, 16814, 16815, 16816, 16817, 16818, 16819, 16820, 16821, 16823, 16824, 16825, 16826, 16828,
    16830, 16831, 16832, 16834, 16835, 16836, 16839, 16840, 16841, 16842, 16843, 16845, 16846, 16847, 16848, 16849, 16850, 16851, 16852, 16853, 16854,
    16855, 16856, 16858, 16864, 16867, 16869, 16872, 16873, 16874, 16875, 16877, 16878, 16879, 16880, 16882, 16883, 16884, 16885, 16886, 16888, 16889,
    16890, 16891, 16893, 16894, 16895, 16896, 16898, 16901, 16903, 16904, 16906, 16907, 16909, 16910, 16912, 16913, 16916, 16919, 16920, 16921, 16923,
    16925, 16926, 16927, 16929, 16931, 16932, 16934, 16935, 16937, 16939, 16941, 16942, 16944, 16945, 16946, 16947, 16948, 16949, 16950, 16951, 16952,
    16955, 16956, 16960, 16962, 16963, 16964, 16965, 16966, 16970, 16973, 16975, 16976, 16978, 16979, 16985, 16986, 16990, 16992, 16993, 16994, 16996,
    17001, 17002, 17004, 17005, 17009, 17010, 17011, 17013, 17019, 17021, 17022, 17023, 17024, 17027, 17028, 17029, 17030, 17031, 17032, 17033, 17034,
    17035, 17036, 17037, 17039, 17041, 17043, 17044, 17045, 17046, 17047, 17048, 17052, 17053, 17055, 17056, 17057, 17060, 17065, 17067, 17068, 17070,
    17072, 17073, 17075, 17076, 17077, 17078, 17079, 17080, 17082, 17083, 17084, 17087, 17088, 17089, 17090, 17091, 17092, 17093, 17095, 17096, 17097,
    17098, 17099, 17100, 17101, 17102, 17103, 17106, 17108, 17109, 17110, 17111, 17112, 17114, 17116, 17118, 17119, 17120, 17121, 17122, 17123, 17124,
    17125, 17126, 17128, 17129, 17131, 17132, 17133, 17134, 17135, 17136, 17137, 17138, 17139, 17140, 17141, 17142, 17143, 17144, 17146, 17147, 17149,
    17150, 17152, 17155, 17156, 17157, 17158, 17159, 17160, 17161, 17162, 17164, 17165, 17166, 17168, 17169, 17170, 17171, 17172, 17173, 17174, 17175,
    17177, 17178, 17179, 17181, 17182, 17183, 17184, 17185, 17186, 17188, 17190, 17191, 17193, 17194, 17195, 17196, 17197, 17198, 17199, 17201, 17203,
    17204, 17205, 17206, 17208, 17209, 17210, 17212, 17213, 17214, 17215, 17217, 17218, 17219, 17220, 17222, 17223, 17224, 17226, 17227, 17228, 17229,
    17231, 17232, 17235, 17237, 17238, 17239, 17240, 17242, 17243, 17244, 17245, 17246, 17249, 17250, 17251, 17252, 17254, 17255, 17256, 17257, 17258,
    17259, 17260, 17264, 17265, 17266, 17267, 17268, 17269, 17270, 17271, 17274, 17275, 17276, 17279, 17280, 17281, 17282, 17283, 17287, 17295, 17300,
    17301, 17302, 17303, 17304, 17305, 17306, 17307, 17308, 17309, 17310, 17311, 17312, 17313, 17314, 17316, 17317, 17318, 17320, 17322, 17323, 17324,
    17325, 17326, 17327, 17328, 17331, 17332, 17333, 17334, 17335, 17336, 17337, 17338, 17339, 17341, 17343, 17344, 17345, 17348, 17349, 17351, 17352,
    17353, 17354, 17355, 17356, 17357, 17358, 17360, 17361, 17363, 17364, 17365, 17366, 17367, 17368, 17369, 17370, 17371, 17372, 17373, 17379, 17380,
    17382, 17383, 17386, 17387, 17388, 17389, 17390, 17391, 17393, 17394, 17395, 17396, 17397, 17398, 17400, 17401, 17402, 17404, 17406, 17407, 17408,
    17409, 17410, 17413, 17414, 17415, 17417, 17418, 17419, 17420, 17421, 17422, 17423, 17426, 17427, 17428, 17429, 17430, 17431, 17432, 17433, 17434,
    17435, 17438, 17439, 17440, 17441, 17443, 17444, 17447, 17448, 17449, 17451, 17452, 17453, 17455, 17456, 17457, 17458, 17459, 17460, 17461, 17462,
    17463, 17466, 17467, 17468, 17469, 17471, 17472, 17474, 17475, 17476, 17478, 17479, 17480, 17481, 17482, 17484, 17486, 17487, 17488, 17489, 17491,
    17492, 17493, 17494, 17495, 17496, 17497, 17498, 17499, 17500, 17501, 17502, 17503, 17504, 17505, 17506, 17508, 17509, 17510, 17511, 17513, 17514,
    17517, 17518, 17520, 17521, 17522, 17523, 17524, 17525, 17526, 17528, 17529, 17530, 17531, 17532, 17534, 17535, 17536, 17537, 17540, 17541, 17543,
    17544, 17545, 17547, 17549, 17550, 17552, 17555, 17556, 17557, 17559, 17560, 17561, 17563, 17564, 17566, 17567, 17568, 17569, 17571, 17572, 17573,
    17575, 17577, 17578, 17579, 17580, 17581, 17582, 17584, 17585, 17586, 17587, 17589, 17590, 17591, 17592, 17594, 17596, 17597, 17598, 17599, 17600,
    17601, 17602, 17603, 17604, 17605, 17606, 17607, 17608, 17609, 17611, 17612, 17614, 17615, 17616, 17618, 17619, 17620, 17621, 17622, 17623, 17626,
    17630, 17631, 17633, 17634, 17636, 17637, 17638, 17641, 17644, 17645, 17646, 17647, 17648, 17650, 17651, 17652, 17653, 17655, 17657, 17658, 17660,
    17661, 17662, 17663, 17664, 17665, 17666, 17667, 17668, 17669, 17670, 17671, 17674, 17675, 17678, 17679, 17680, 17681, 17683, 17684, 17686, 17687,
    17688, 17689, 17691, 17692, 17693, 17694, 17695, 17696, 17697, 17698, 17699, 17700, 17701, 17702, 17703, 17719, 17720, 17721, 17722, 17723, 17724,
    17725, 17726, 17727, 17729, 17730, 17731, 17732, 17734, 17736, 17738, 17739, 17742, 17743, 17744, 17745, 17746, 17749, 17751, 17753, 17754, 17755,
    17756, 17757, 17758, 17759, 17760, 17761, 17762, 17763, 17765, 17766, 17768, 17769, 17771, 17774, 17776, 17779, 17780, 17781, 17783, 17784, 17787,
    17788, 17789, 17790, 17791, 17792, 17793, 17795, 17797, 17798, 17799, 17800, 17801, 17802, 17805, 17818, 17822, 17823, 17824, 17825, 17827, 17829,
    17830, 17831, 17833, 17834, 17835, 17840, 17841, 17846, 17847, 17849, 17850, 17851, 17853, 17855, 17856, 17857, 17858, 17860, 17861, 17862, 17865,
    17866, 17869, 17870, 17871, 17872, 17874, 17875, 17876, 17877, 17955, 17956, 17958, 17960, 17961, 17963, 17965, 17966, 17972, 17973, 17974, 17975,
    17978, 17983, 17984, 17985, 17986, 17987, 17989, 17990, 17992, 17994, 17996, 17998, 18001, 18005, 18007, 18008, 18009, 18010, 18011, 18014, 18018,
    18021, 18024, 18026, 18027, 18028, 18030, 18031, 18032, 18034, 18035, 18037, 18039, 18040, 18041, 18044, 18045, 18048, 18049, 18050, 18052, 18054,
    18057, 18059, 18062, 18202, 18208, 18210, 18212, 18213, 18215, 18218, 18219, 18220, 18224, 18225, 18226, 18228, 18229, 18232, 18233, 18234, 18235,
    18237, 18238, 18240, 18241, 18242, 18244, 18245, 18246, 18248, 18249, 18250, 18251, 18253, 18257, 18258, 18261, 18263, 18265, 18266, 18267, 18268,
    18269, 18270, 18272, 18282, 18283, 18285, 18286, 18463, 18465, 18466, 18467, 18468, 18469, 18470, 18472, 18474, 18476, 18479, 18480, 18481, 18482,
    18484, 18488, 18489, 18490, 18491, 18492, 18495, 18496, 18497, 18499, 18502, 18505, 18506, 18508, 18509, 18510, 18511, 18512, 18514, 18515, 18516,
    18519, 18520, 18523, 18525, 18526, 18527, 18574, 18575, 18577, 18595, 18603, 18604, 18605, 18607, 18609, 18610, 18612, 18615, 18617, 18619, 18621,
    18622, 18623, 18625, 18628, 18630, 18631, 18632, 18634, 18636, 18637, 18639, 18643, 18644, 18645, 18646, 18647, 18650, 18651, 18653, 18654, 18657,
    18658, 18659, 18662, 18663, 18665, 18666, 18668, 18673, 18674, 18676, 18677, 18678, 18679, 18685, 18686, 18687, 18689, 18690, 18692, 18694, 18696,
    18697, 18699, 18701, 18703, 18704, 18708, 18709, 18710, 18711, 18712, 18713, 18714, 18717, 18719, 18723, 18724, 18731, 18732, 18733, 18737, 18740,
    18741, 18743, 18746, 18747, 18748, 18750, 18752, 18753, 18754, 18756, 18758, 18760, 18761, 18762, 18763, 18764, 18765, 18766, 18767, 18768, 18769,
    18770, 18771, 18773, 18777, 18779, 18780, 18781, 18782, 18784, 18786, 18788, 18790, 18792, 18793, 18794, 18795, 18796, 18841, 18845, 18846, 18852,
    18856, 18860, 18861, 18862, 18863, 18864, 18865, 18867, 18868, 18872, 18873, 18874, 18876, 18882, 18883, 18885, 18900, 18901, 18903, 18907, 18910,
    18911, 18913, 18915, 18918, 18919, 18920, 18921, 18922, 18923, 18924, 18926, 18927, 18928, 18929, 18930, 18931, 18933, 18935, 18937, 18938, 18941,
           18944, 18947, 18948, 18949, 18950, 18951, 18952, 18953, 18954, 18955, 18957, 18958, 18959, 18960,
           8321, 8334, 8335, 8336, 8341, 8343, 8344, 8348, 8354, 8360, 8361, 8362, 8363, 8364, 8365, 8366, 8368, 8371, 8372, 8373, 8374, 8375, 8377,
    8378, 8379, 8380, 8382, 8390, 8391, 8404, 8406, 8407, 8408, 8409, 8410, 8411, 8412, 8413, 8414, 8415, 8416, 8418, 8419, 8420, 8421, 8423, 8424, 8427,
    8430, 8431, 8433, 8434, 8435, 8436, 8438, 8439, 8440, 8441, 8443, 8444, 8445, 8446, 8447, 8448, 8454, 8455, 8458, 8459, 8460, 8461, 8462, 8464, 8465,
    8467, 8468, 8469, 8470, 8471, 8472, 8473, 8475, 8476, 8487, 8489, 8490, 8492, 8493, 8495, 8498, 8504, 8506, 8507, 8508, 8510, 8513, 8514, 8518, 8520,
    8522, 8526, 8528, 8529, 8531, 8534, 8536, 8537, 8538, 8539, 8540, 8541, 8543, 8546, 8547, 8548, 8549, 8550, 8553, 8554, 8555, 8556, 8562, 8565, 8566,
    8567, 8571, 8572, 8573, 8578, 8579, 8580, 8581, 8584, 8585, 8586, 8587, 8588, 8589, 8590, 8601, 8613, 8614, 8615, 8616, 8617, 8618, 8619, 8620, 8621,
    8622, 8624, 8626, 8627, 8635, 8638, 8644, 8646, 8647, 8648, 8649, 8650, 8652, 8653, 8654, 8655, 8656, 8658, 8659, 8660, 8661, 8663, 8664, 8666, 8668,
    8669, 8670, 8671, 8672, 8673, 8675, 8677, 8680, 8681, 8682, 8683, 8686, 8687, 8688, 8689, 8690, 8691, 8693, 8695, 8697, 8698, 8699, 8700, 8701, 8702,
    8703, 8704, 8705, 8706, 8707, 8708, 8709, 8711, 8712, 8714, 8715, 8716, 8718, 8719, 8720, 8721, 8724, 8735, 8736, 8738, 8739, 8740, 8742, 8743, 8744,
    8745, 8746, 8747, 8748, 8749, 8750, 8751, 8752, 8754, 8755, 8756, 8757, 8758, 8759, 8762, 8763, 8764, 8766, 8767, 8768, 8769, 8770, 8771, 8772, 8773,
    8774, 8776, 8777, 8778, 8780, 8782, 8783, 8784, 8786, 8787, 8789, 8790, 8791, 8792, 8793, 8794, 8795, 8796, 8797, 8798, 8799, 8800, 8801, 8802, 8804,
    8806, 8807, 8809, 8810, 8811, 8813, 8814, 8817, 8825, 8830, 8832, 8836, 8841, 8852, 8853, 8855, 8856, 8857, 8859, 8860, 8861, 8862, 8863, 8866, 8867,
    8868, 8873, 8874, 8876, 8877, 8878, 8879, 8882, 8883, 8885, 8886, 8888, 8889, 8890, 8891, 8893, 8894, 8897, 8899, 8900, 8902, 8903, 8904, 8905, 8906,
    8907, 8908, 8909, 8911, 8912, 8914, 8915, 8917, 8918, 8919, 8920, 8921, 8922, 8923, 8925, 8926, 8927, 8928, 8931, 8934, 8935, 8936, 8937, 8938, 8939,
    8940, 8941, 8943, 8954, 8955, 8957, 8958, 8959, 8960, 8962, 8963, 8964, 8966, 8968, 8971, 8972, 8974, 8975, 8976, 8977, 8979, 8981, 8982, 8987, 8989,
    8990, 8991, 8993, 8994, 8995, 8996, 8997, 8998, 8999, 9001, 9002, 9004, 9005, 9007, 9008, 9009, 9014, 9018, 9022, 9023, 9024, 9025, 9026, 9028, 9030,
    9033, 9034, 9035, 9037, 9038, 9039, 9040, 9041, 9044, 9049, 9050, 9052, 9053, 9055, 9056, 9057, 9059, 9060, 9061, 9062, 9063, 9064, 9065, 9066, 9067,
    9068, 9069, 9072, 9073, 9074, 9075, 9076, 9079, 9080, 9081, 9083, 9084, 9085, 9087, 9088, 9089, 9090, 9092, 9093, 9094, 9095, 9096, 9108, 9109, 9113,
    9114, 9116, 9134, 9136, 9138, 9142, 9144, 9146, 9148, 9149, 9158, 9169, 9175, 9186, 9188, 9191, 9192, 9194, 9195, 9196, 9199, 9201, 9202, 9203, 9205,
    9206, 9207, 9208, 9209, 9210, 9212, 9214, 9215, 9216, 9217, 9218, 9223, 9225, 9226, 9228, 9230, 9231, 9232, 9233, 9236, 9237, 9238, 9239, 9240, 9241,
    9242, 9243, 9245, 9246, 9247, 9249, 9250, 9252, 9254, 9255, 9262, 9265, 9266, 9267, 9268, 9269, 9272, 9273, 9275, 9277, 9278, 9279, 9280, 9281, 9282,
    9284, 9285, 9286, 9287, 9288, 9289, 9290, 9291, 9292, 9293, 9295, 9297, 9298, 9299, 9300, 9301, 9302, 9303, 9304, 9305, 9307, 9308, 9310, 9311, 9313,
    9316, 9317, 9319, 9320, 9321, 9324, 9325, 9326, 9327, 9328, 9329, 9331, 9332, 9333, 9336, 9337, 9338, 9340, 9341, 9343, 9344, 9345, 9346, 9347, 9348,
    9349, 9350, 9351, 9354, 9356, 9358, 9360, 9361, 9362, 9363, 9364, 9366, 9367, 9369, 9370, 9372, 9373, 9374, 9375, 9377, 9378, 9379, 9383, 9384, 9385,
    9386, 9390, 9391, 9392, 9394, 9395, 9396, 9397, 9398, 9401, 9403, 9404, 9405, 9406, 9407, 9408, 9410, 9411, 9412, 9414, 9415, 9416, 9417, 9421, 9422,
    9423, 9425, 9427, 9428, 9429, 9431, 9432, 9433, 9434, 9435, 9436, 9437, 9438, 9439, 9440, 9442, 9444, 9445, 9446, 9447, 9448, 9449, 9452, 9453, 9454,
    9456, 9457, 9458, 9459, 9460, 9461, 9463, 9464, 9465, 9466, 9467, 9468, 9470, 9471, 9472, 9473, 9474, 9477, 9479, 9480, 9481, 9482, 9483, 9485, 9486,
    9488, 9489, 9491, 9492, 9494, 9497, 9498, 9499, 9500, 9501, 9503, 9504, 9505, 9506, 9507, 9508, 9509, 9510, 9511, 9512, 9513, 9515, 9516, 9517, 9518,
    9519, 9521, 9523, 9524, 9525, 9526, 9527, 9529, 9530, 9531, 9532, 9533, 9534, 9535, 9536, 9538, 9539, 9542, 9543, 9545, 9546, 9548, 9550, 9551, 9553,
    9554, 9556, 9557, 9558, 9559, 9560, 9561, 9562, 9563, 9564, 9565, 9567, 9569, 9570, 9571, 9572, 9573, 9574, 9575, 9576, 9577, 9578, 9579, 9580, 9581,
    9582, 9583, 9584, 9585, 9586, 9587, 9589, 9590, 9591, 9592, 9593, 9594, 9595, 9597, 9598, 9599, 9601, 9602, 9603, 9604, 9605, 9607, 9608, 9609, 9610,
    9612, 9613, 9614, 9615, 9617, 9618, 9619, 9620, 9621, 9627, 9628, 9629, 9630, 9631, 9632, 9633, 9634, 9636, 9637, 9639, 9640, 9644, 9645, 9649, 9650,
    9651, 9653, 9654, 9655, 9656, 9657, 9658, 9660, 9661, 9662, 9665, 9667, 9668, 9669, 9670, 9672, 9674, 9675, 9677, 9678, 9679, 9681, 9682, 9683, 9684,
    9685, 9689, 9690, 9694, 9695, 9696, 9698, 9699, 9700, 9702, 9703, 9704, 9706, 9708, 9709, 9710, 9712, 9713, 9716, 9717, 9718, 9719, 9720, 9721, 9722,
    9724, 9725, 9726, 9727, 9729, 9730, 9741, 9742, 9744, 9749, 9750, 9751, 9757, 9758, 9760, 9761, 9762, 9765, 9767, 10307, 10308, 10353, 10354, 10355,
    10356, 10357, 10359, 10362, 10363, 10366, 10371, 10374, 10375, 10379, 10380, 10381, 10382, 10383, 10385, 10386, 10389, 10390, 10394, 10396, 10432,
    10433, 10434, 10436, 10437, 10438, 10439, 10441, 10442, 10444, 10445, 10446, 10447, 10448, 10449, 10450, 10452, 10453, 10471, 10472, 10473, 10475,
    10478, 10479, 10480, 10481, 10482, 10483, 10484, 10485, 10486, 10487, 10488, 10489, 10490, 10491, 10492, 10493, 10500, 10501, 10502, 10504, 10505,
    10508, 10512, 10516, 10517, 10518, 10519, 10520, 10521, 10522, 10523, 10525, 10529, 10530, 10532, 10533, 10534, 10535, 10537, 10538, 10540, 10544,
    10545, 10546, 10547, 10551, 10553, 10555, 10556, 10558, 10560, 10561, 10562, 10563, 10565, 10567, 10568, 10569, 10570, 10571, 10573, 10574, 10575,
    10577, 10578, 10579, 10580, 10581, 10583, 10584, 10585, 10586, 10590, 10591, 10592, 10593, 10594, 10595, 10597, 10598, 10599, 10600, 10601, 10602,
    10603, 10605, 10606, 10608, 10609, 10610, 10611, 10612, 10613, 10616, 10618, 10622, 10623, 10624, 10625, 10626, 10627, 10629, 10630, 10631, 10633,
    10634, 10635, 10639, 10640, 10641, 10643, 10644, 10645, 10646, 10647, 10648, 10649, 10652, 10654, 10655, 10656, 10657, 10658, 10660, 10662, 10663,
    10665, 10666, 10667, 10668, 10669, 10670, 10672, 10673, 10674, 10675, 10678, 10680, 10681, 10682, 10683, 10685, 10687, 10688, 10689, 10690, 10692,
    10693, 10694, 10695, 10696, 10697, 10698, 10700, 10701, 10702, 10704, 10705, 10706, 10707, 10708, 10710, 10711, 10712, 10713, 10714, 10716, 10719,
    10720, 10721, 10724, 10725, 10726, 10727, 10728, 10729, 10730, 10732, 10733, 10734, 10735, 10736, 10737, 10741, 10743, 10744, 10745, 10746, 10747,
    10748, 10750, 10752, 10755, 10756, 10757, 10758, 10760, 10761, 10762, 10763, 10764, 10765, 10766, 10767, 10768, 10769, 10771, 10772, 10773, 10774,
    10778, 10781, 10782, 10784, 10786, 10787, 10788, 10789, 10790, 10791, 10792, 10793, 10794, 10795, 10796, 10797, 10798, 10799, 10800, 10801, 10803,
    10804, 10805, 10807, 10809, 10810, 10811, 10813, 10815, 10816, 10818, 10819, 10820, 10821, 10822, 10823, 10824, 10827, 10828, 10829, 10830, 10831,
    10832, 10833, 10834, 10836, 10837, 10838, 10839, 10840, 10841, 10842, 10843, 10844, 10845, 10846, 10847, 10848, 10849, 10850, 10855, 10856, 10857,
    10858, 10861, 10862, 10863, 10864, 10865, 10867, 10868, 10869, 10870, 10873, 10874, 10877, 10878, 10881, 10882, 10883, 10884, 10885, 10886, 10887,
    10888, 10890, 10892, 10893, 10894, 10897, 10898, 10900, 10901, 10902, 10903, 10905, 10906, 10908, 10910, 10911, 10913, 10918, 10920, 10921, 10922,
    10923, 10924, 10925, 10926, 10927, 10928, 10929, 10931, 10933, 10936, 10937, 10938, 10939, 10940, 10942, 10946, 10947, 10948, 10949, 10950, 10951,
    10952, 10953, 10954, 10955, 10956, 10958, 10959, 10960, 10961, 10963, 10965, 10966, 10967, 10968, 10969, 10970, 10971, 10972, 10973, 10974, 10975,
    10976, 10979, 10983, 10984, 10987, 10988, 10989, 10990, 10991, 10992, 10993, 10994, 10995, 10997, 10998, 11000, 11001, 11002, 11005, 11024, 11025,
    11026, 11027, 11029, 11030, 11031, 11032, 11036, 11039, 11041, 11042, 11044, 11045, 11046, 11047, 11048, 11049, 11050, 11059, 11064, 11065, 11066,
    11067, 11068, 11069, 11070, 11071, 11072, 11073, 11074, 11075, 11076, 11077, 11078, 11080, 11081, 11083, 11085, 11086, 11087, 11088, 11090, 11091,
    11093, 11094, 11095, 11096, 11103, 11108, 11110, 11111, 11112, 11115, 11116, 11118, 11119, 11120, 11127, 11128, 11129, 11131, 11132, 11133, 11134,
    11135, 11137, 11138, 11139, 11141, 11142, 11143, 11144, 11145, 11146, 11147, 11149, 11150, 11153, 11154, 11157, 11162, 11164, 11166, 11168, 11169,
    11170, 11171, 11172, 11173, 11174, 11176, 11177, 11179, 11182, 11184, 11186, 11187, 11188, 11189, 11190, 11192, 11193, 11194, 11196, 11197, 11198,
    11201, 11202, 11203, 11204, 11205, 11206, 11208, 11210, 11211, 11212, 11213, 11214, 11215, 11217, 11218, 11219, 11224, 11225, 11226, 11227, 11228,
    11229, 11230, 11232, 11233, 11234, 11235, 11236, 11237, 11238, 11239, 11240, 11242, 11243, 11244, 11247, 11249, 11253, 11255, 11256, 11259, 11260,
    11263, 11264, 11265, 11269, 11270, 11271, 11274, 11275, 11278, 11279, 11281, 11282, 11283, 11284, 11285, 11286, 11288, 11290, 11291, 11292, 11293,
    11294, 11296, 11297, 11492, 11493, 11494, 11495, 11500, 11501, 11502, 11503, 11504, 11505, 11506, 11507, 11508, 11509, 11510, 11512, 11514, 11515,
    11518, 11519, 11520, 11521, 11524, 11526, 11527, 11528, 11529, 11531, 11532, 11535, 11537, 11538, 11539, 11540, 11543, 11544, 11545, 11546, 11550,
    11551, 11553, 11554, 11557, 11558, 11559, 11561, 11564, 11565, 11568, 11570, 11571, 11572, 11574, 11575, 11576, 11577, 11578, 11579, 11581, 11582,
    11583, 11584, 11586, 11587, 11589, 11591, 11593, 11598, 11599, 11601, 11603, 11604, 11605, 11606, 11608, 11609, 11610, 11611, 11613, 11614, 11617,
    11618, 11619, 11621, 11622, 11623, 11624, 11625, 11626, 11628, 11629, 11630, 11631, 11632, 11634, 11636, 11637, 11638, 11639, 11640, 11644, 11646,
    11647, 11648, 11649, 11650, 11652, 11653, 11655, 11656, 11657, 11658, 11659, 11660, 11664, 11665, 11666, 11667, 11668, 11672, 11673, 11674, 11675,
    11677, 11678, 11679, 11680, 11681, 11682, 11684, 11685, 11686, 11687, 11691, 11692, 11693, 11695, 11696, 11698, 11700, 11702, 11707, 11709, 11710,
    11712, 11713, 11714, 11715, 11716, 11719, 11720, 11721, 11723, 11724, 11726, 11727, 11728, 11729, 11730, 11733, 11737, 11738, 11739, 11740, 11741,
    11742, 11744, 11745, 11746, 11747, 11749, 11751, 11753, 11754, 11755, 11756, 11758, 11759, 11761, 11762, 11763, 11765, 11766, 11767, 11769, 11771,
    11772, 11774, 11775, 11776, 11777, 11778, 11779, 11780, 11781, 11782, 11783, 11784, 11785, 11786, 11787, 11791, 11792, 11793, 11794, 11795, 11796,
    11797, 11799, 11800, 11801, 11802, 11803, 11804, 11807, 11809, 11811, 11813, 11814, 11815, 11816, 11817, 11818, 11819, 11820, 11821, 11822, 11824,
    11828, 11829, 11830, 11831, 11832, 11833, 11836, 11840, 11841, 11842, 11844, 11845, 11846, 11847, 11848, 11849, 11851, 11852, 11853, 11854, 11857,
    11858, 11861, 11865, 11866, 11867, 11868, 11869, 11870, 11872, 11873, 11874, 11875, 11876, 11878, 11879, 11880, 11881, 11882, 11884, 11885, 11886,
    11887, 11888, 11889, 11891, 11892, 11893, 11894, 11895, 11896, 11897, 11898, 11900, 11901, 11904, 11905, 11906, 11907, 11909, 11910, 11911, 11912,
    11913, 11914, 11915, 11917, 11919, 11920, 11921, 11922, 11923, 11924, 11925, 11926, 11927, 11928, 11929, 11930, 11931, 11933, 11934, 11935, 11936,
    11937, 11938, 11939, 11940, 11943, 11945, 11946, 11947, 11952, 11953, 11956, 11957, 11958, 11959, 11960, 11961, 11963, 11964, 11965, 11969, 11970,
    11971, 11972, 11973, 11974, 11976, 11977, 11979, 11981, 11984, 11988, 11989, 11990, 11991, 11993, 11994, 11995, 11996, 11998, 12000, 12001, 12002,
    12004, 12005, 12006, 12008, 12009, 12010, 12011, 12012, 12013, 12015, 12016, 12017, 12020, 12021, 12023, 12024, 12025, 12026, 12027, 12028, 12029,
    12030, 12031, 12032, 12034, 12036, 12039, 12040, 12041, 12042, 12043, 12044, 12045, 12046, 12048, 12052, 12053, 12054, 12056, 12057, 12060, 12061,
    12062, 12063, 12066, 12067, 12068, 12069, 12070, 12071, 12072, 12073, 12075, 12076, 12077, 12079, 12080, 12083, 12084, 12086, 12089, 12090, 12091,
    12093, 12094, 12095, 12097, 12098, 12100, 12102, 12103, 12104, 12106, 12107, 12108, 12109, 12110, 12111, 12116, 12117, 12118, 12120, 12121, 12122,
    12125, 12127, 12130, 12131, 12132, 12133, 12135, 12136, 12137, 12139, 12140, 12142, 12143, 12145, 12146, 12331, 12364, 12365, 12366, 12367, 12368,
    12369, 12374, 12377, 12378, 12379, 12380, 12381, 12382, 12383, 12384, 12389, 12392, 12401, 12402, 12403, 12404, 12406, 12418, 12419, 12420, 12422,
    12425, 12426, 12429, 12430, 12431, 12432, 12433, 12434, 12436, 12438, 12441, 12442, 12444, 12446, 12447, 12448, 12449, 12450, 12451, 12453, 12454,
    12456, 12459, 12460, 12461, 12462, 12464, 12466, 12469, 12479, 12480, 12481, 12482, 12485, 12486, 12492, 12493, 12495, 12496, 12497, 12498, 12499,
    12502, 12503, 12504, 12505, 12506, 12507, 12508, 12513, 12514, 12515, 12516, 12517, 12518, 12520, 12521, 12522, 12526, 12527, 12528, 12529, 12530,
    12533, 12534, 12535, 12537, 12538, 12539, 12540, 12541, 12542, 12543, 12544, 12545, 12549, 12550, 12551, 12552, 12553, 12554, 12555, 12556, 12557,
    12558, 12560, 12561, 12565, 12566, 13452, 13453, 13454, 13455, 13457, 13460, 13461, 13462, 13463, 13464, 13465, 13466, 13467, 13468, 13469, 13470,
    13471, 13472, 13473, 13474, 13475, 13476, 13477, 13478, 13479, 13480, 13481, 13483, 13484, 13487, 13488, 13491, 13492, 13503, 13504, 13505, 13506,
    13507, 13509, 13510, 13511, 13517, 13518, 13519, 13520, 13521, 13522, 13523, 13524, 13525, 13526, 13527, 13528, 13529, 13530, 13531, 13532, 13533,
    13535, 13536, 13537, 13538, 13539, 13540, 13542, 13543, 13544, 13545, 13546, 13547, 13548, 13549, 13550, 13552, 13553, 13554, 13555, 13557, 13558,
    13559, 13561, 13562, 13563, 13564, 13565, 13567, 13568, 13569, 13570, 13571, 13572, 13573, 13574, 13579, 13580, 13582, 13583, 13584, 13586, 13587,
    13588, 13591, 13592, 13593, 13612, 13614, 13616, 13617, 13618, 13619, 13620, 13621, 13623, 13624, 13625, 13626, 13627, 13629, 13630, 13631, 13632,
    13633, 13634, 13635, 13636, 13637, 13638, 13639, 13641, 13642, 13643, 13644, 13645, 13647, 13648, 13649, 13650, 13651, 13652, 13653, 13654, 13655,
    13656, 13658, 13659, 13660, 13661, 13662, 13664, 13665, 13667, 13668, 13669, 13670, 13671, 13672, 13674, 13675, 13676, 13678, 13680, 13681, 13683,
    13684, 13685, 13687, 13688, 13689, 13690, 13691, 13692, 13693, 13695, 13696, 
           18961, 18963, 18964, 18965, 18966, 18968, 18969, 18970, 18972, 18974, 18976, 18979, 18980, 18982, 18984, 18986, 18987, 18988, 18990, 18991,
    18997, 18999, 19001, 19002, 19003, 19004, 19006, 19008, 19009, 19012, 19015, 19018, 19021, 19022, 19023, 19027, 19033, 19036, 19038, 19046, 19048,
    19051, 19054, 19056, 19058, 19061, 19064, 19065, 19067, 19068, 19070, 19071, 19074, 19075, 19077, 19078, 19079, 19080, 19082, 19085, 19086, 19408,
    19411, 19413, 19416, 19419, 19420, 19421, 19423, 19425, 19427, 19428, 19430, 19432, 19434, 19436, 19438, 19441, 19443, 19444, 19446, 19447, 19451,
    19455, 19456, 19459, 19460, 19462, 19463, 19464, 19465, 19467, 19470, 19471, 19473, 19475, 19477, 19480, 19482, 19483, 19488, 19489, 19490, 19491,
    19493, 19495, 19496, 19498, 19499, 19500, 19501, 19503, 19504, 19507, 19509, 19510, 19513, 19515, 19518, 19519, 19521, 19523, 19525, 19526, 19527,
    19529, 19530, 19531, 19532, 19536, 19538, 19539, 19540, 19543, 19544, 19545, 19547, 19548, 19550, 19552, 19553, 19554, 19556, 19559, 19560, 19561,
    19563, 19564, 19565, 19568, 19570, 19571, 19574, 19575, 19578, 19580, 19581, 19582, 19585, 19586, 19587, 19588, 19589, 19591, 19592, 19593, 19594,
    19596, 19598, 19600, 19601, 19603, 19606, 19607, 19610, 19612, 19613, 19615, 19616, 19618, 19619, 19620, 19621, 19623, 19624, 19625, 19628, 19629,
    19630, 19631, 19632, 19635, 19636, 19637, 19639, 19640, 19645, 19646, 19648, 19649, 19650, 19651, 19652, 19653, 19655, 19656, 19657, 19658, 19659,
    19660, 19661, 19664, 19666, 19667, 19668, 19669, 19670, 19671, 19672, 19673, 19674, 19675, 19678, 19680, 19681, 19682, 19683, 19684, 19685, 19686,
    19687, 19691, 19692, 19694, 19696, 19697, 19698, 19703, 19705, 19706, 19707, 19708, 19709, 19711, 19712, 19713, 19715, 19716, 19717, 19718, 19721,
    19722, 19724, 19725, 19727, 19728, 19729, 19730, 19731, 19733, 19734, 19735, 19738, 19742, 19743, 19744, 19747, 19748, 19749, 19750, 19751, 19752,
    19753, 19757, 19760, 19761, 19762, 19763, 19764, 19765, 19766, 19767, 19768, 19769, 19770, 19772, 19773, 19774, 19776, 19777, 19780, 19781, 19783,
    19785, 19786, 19787, 19788, 19789, 19790, 19792, 19794, 19795, 19798, 19799, 19800, 19801, 19802, 19803, 19804, 19805, 19806, 19807, 19808, 19810,
    19811, 19812, 19813, 19814, 19815, 19816, 19818, 19819, 19822, 19823, 19824, 19825, 19826, 19827, 19828, 19829, 19830, 19831, 19832, 19836, 19837,
    19838, 19839, 19840, 19841, 19842, 19843, 19845, 19846, 19847, 19848, 19849, 19850, 19852, 19853, 19854, 19855, 19856, 19857, 19858, 19859, 19866,
    19867, 19868, 19869, 19875, 19876, 19877, 19878, 19879, 19880, 19881, 19883, 19884, 19886, 19887, 19888, 19889, 19890, 19891, 19892, 19893, 19894,
    19897, 19898, 19899, 19901, 19902, 19903, 19904, 19905, 19906, 19907, 19910, 19911, 19912, 19913, 19915, 19917, 19918, 19921, 19922, 19923, 19924,
    19925, 19926, 19927, 19928, 19929, 19933, 19934, 19935, 19936, 19937, 19938, 19940, 19941, 19942, 19945, 19946, 19947, 19948, 19949, 19950, 19951,
    19952, 19954, 19955, 19956, 19957, 19961, 19962, 19963, 19964, 19966, 19967, 19968, 19969, 19970, 19971, 19972, 19973, 19974, 19975, 19976, 19978,
    19980, 19981, 19982, 19983, 19984, 19985, 19986, 19987, 19989, 19990, 19991, 19992, 19993, 19994, 19995, 19996, 19997, 19998, 19999, 20000, 20001,
    20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20012, 20013, 20014, 20015, 20016, 20017, 20018, 20019, 20021, 20022, 20023, 20025, 20026,
    20027, 20028, 20029, 20033, 20035, 20036, 20038, 20039, 20040, 20041, 20042, 20043, 20044, 20045, 20046, 20047, 20048, 20049, 20050, 20051, 20052,
    20053, 20055, 20056, 20057, 20058, 20059, 20060, 20061, 20062, 20063, 20064, 20066, 20067, 20068, 20069, 20071, 20072, 20073, 20074, 20075, 20077,
    20080, 20081, 20082, 20083, 20084, 20086, 20087, 20088, 20090, 20091, 20092, 20093, 20094, 20095, 20096, 20097, 20098, 20099, 20100, 20102, 20103,
    20104, 20105, 20106, 20107, 20108, 20109, 20110, 20112, 20113, 20115, 20116, 20117, 20119, 20120, 20121, 20122, 20124, 20126, 20127, 20128, 20130,
    20131, 20132, 20133, 20134, 20135, 20136, 20138, 20139, 20140, 20141, 20142, 20143, 20146, 20147, 20148, 20149, 20150, 20152, 20153, 20154, 20155,
    20156, 20157, 20159, 20160, 20161, 20162, 20163, 20164, 20165, 20167, 20168, 20169, 20170, 20171, 20172, 20174, 20175, 20176, 20177, 20178, 20179,
    20180, 20185, 20186, 20187, 20188, 20190, 20191, 20192, 20193, 20194, 20196, 20197, 20198, 20199, 20200, 20201, 20202, 20203, 20204, 20205, 20206,
    20207, 20208, 20209, 20210, 20211, 20212, 20213, 20214, 20215, 20216, 20217, 20218, 20219, 20220, 20221, 20222, 20223, 20224, 20225, 20226, 20228,
    20229, 20230, 20231, 20233, 20235, 20236, 20237, 20238, 20239, 20241, 20242, 20243, 20245, 20248, 20249, 20250, 20251, 20252, 20253, 20254, 20255,
    20256, 20257, 20258, 20259, 20261, 20263, 20264, 20265, 20266, 20267, 20268, 20269, 20270, 20271, 20272, 20273, 20274, 20275, 20276, 20278, 20279,
    20280, 20281, 20282, 20283, 20286, 20288, 20290, 20291, 20292, 20293, 20294, 20295, 20296, 20298, 20299, 20300, 20302, 20303, 20304, 20305, 20307,
    20308, 20309, 20310, 20311, 20314, 20315, 20318, 20319, 20320, 20323, 20327, 20328, 20329, 20330, 20331, 20332, 20334, 20335, 20336, 20337, 20338,
    20339, 20340, 20341, 20342, 20343, 20344, 20345, 20346, 20350, 20351, 20352, 20353, 20354, 20355, 20356, 20357, 20360, 20362, 20363, 20364, 20365,
    20366, 20367, 20368, 20369, 20370, 20371, 20373, 20374, 20375, 20376, 20377, 20379, 20380, 20381, 20383, 20384, 20385, 20386, 20387, 20388, 20389,
    20391, 20392, 20393, 20394, 20396, 20397, 20398, 20400, 20401, 20402, 20403, 20405, 20406, 20407, 20408, 20409, 20410, 20413, 20414, 20416, 20417,
    20418, 20419, 20420, 20421, 20423, 20424, 20425, 20426, 20427, 20428, 20429, 20430, 20431, 20433, 20435, 20436, 20437, 20438, 20439, 20440, 20441,
    20442, 20443, 20444, 20445, 20446, 20447, 20450, 20451, 20452, 20453, 20454, 20456, 20458, 20460, 20461, 20462, 20466, 20467, 20468, 20469, 20470,
    20471, 20472, 20473, 20474, 20475, 20476, 20477, 20478, 20480, 20481, 20483, 20484, 20485, 20486, 20487, 20489, 20490, 20491, 20492, 20494, 20495,
    20496, 20497, 20498, 20499, 20500, 20501, 20502, 20503, 20504, 20505, 20506, 20508, 20509, 20510, 20512, 20513, 20514, 20515, 20517, 20518, 20520,
    20523, 20524, 20525, 20526, 20527, 20528, 20529, 20530, 20531, 20532, 20533, 20534, 20535, 20536, 20537, 20538, 20541, 20542, 20543, 20544, 20545,
    20546, 20547, 20548, 20549, 20550, 20551, 20552, 20553, 20554, 20555, 20556, 20557, 20558, 20559, 20560, 20561, 20562, 20563, 20565, 20566, 20567,
    20568, 20569, 20570, 20571, 20572, 20573, 20574, 20575, 20576, 20578, 20579, 20580, 20581, 20582, 20583, 20585, 20586, 20589, 20590, 20591, 20593,
    20594, 20595, 20596, 20597, 20598, 20599, 20601, 20602, 20603, 20604, 20605, 20606, 20607, 20608, 20609, 20610, 20611, 20612, 20613, 20614, 20618,
    20620, 20621, 20622, 20623, 20624, 20625, 20626, 20627, 20629, 20630, 20631, 20632, 20634, 20635, 20636, 20637, 20638, 20639, 20640, 20641, 20642,
    20643, 20644, 20645, 20646, 20647, 20648, 20649, 20650, 20651, 20652, 20654, 20656, 20657, 20660, 20661, 20662, 20663, 20664, 20665, 20666, 20667,
    20669, 20670, 20672, 20675, 20676, 20677, 20678, 20679, 20680, 20681, 20682, 20684, 20686, 20687, 20688, 20689, 20690, 20691, 20692, 20693, 20694,
    20695, 20696, 20697, 20698, 20699, 20700, 20701, 20702, 20703, 20704, 20705, 20706, 20707, 20708, 20709, 20710, 20713, 20714, 20715, 20718, 20719,
    20721, 20723, 20724, 20725, 20726, 20727, 20728, 20729, 20730, 20731, 20732, 20734, 20735, 20738, 20740, 20741, 20742, 20743, 20744, 20745, 20746,
    20748, 20749, 20750, 20751, 20753, 20754, 20755, 20756, 20757, 20758, 20761, 20762, 20763, 20764, 20765, 20766, 20767, 20768, 20769, 20770, 20771,
    20772, 20774, 20775, 20776, 20777, 20778, 20780, 20781, 20782, 20783, 20785, 20786, 20787, 20788, 20791, 20792, 20794, 20795, 20796, 20800, 20801,
    20802, 20803, 20804, 20806, 20808, 20809, 20810, 20811, 20812, 20813, 20814, 20815, 20817, 20818, 20819, 20820, 20821, 20822, 20823, 20824, 20825,
    20826, 20828, 20829, 20830, 20831, 20838, 20839, 20840, 20841, 20842, 20843, 20844, 20845, 20847, 20848, 20850, 20851, 20852, 20853, 20854, 20855,
    20856, 20857, 20858, 20860, 20862, 20863, 20864, 20866, 20867, 20868, 20869, 20870, 20872, 20874, 20875, 20876, 20877, 20879, 20880, 20881, 20882,
    20883, 20885, 20889, 20893, 20894, 20895, 20896, 20898, 20899, 20901, 20902, 20904, 20918, 20920, 20922, 20924, 20925, 20926, 20927, 20928, 20929,
    20930, 20931, 20932, 20933, 20934, 20935, 20936, 20937, 20938, 20939, 20940, 20941, 20942, 20943, 20949, 20950, 20951, 20952, 20953, 20955, 21010,
    21011, 21012, 21013, 21014, 21015, 21016, 21017, 21018, 21020, 21021, 21022, 21023, 21024, 21025, 21026, 21027, 21028, 21030, 21031, 21034, 21035,
    21040, 21042, 21043, 21044, 21045, 21046, 21047, 21049, 21050, 21051, 21052, 21054, 21056, 21057, 21059, 21060, 21061, 21062, 21064, 21069, 21070,
    21071, 21072, 21073, 21077, 21078, 21113, 21118, 21120, 21122, 21125, 21126, 21127, 21128, 21129, 21130, 21131, 21132, 21133, 21134, 21137, 21138,
    21139, 21140, 21143, 21146, 21148, 21150, 21154, 21157, 21240, 21241, 21243, 21244, 21245, 21246, 21247, 21248, 21249, 21250, 21251, 21254, 21256,
    21257, 21258, 21259, 21260, 21261, 21265, 21266, 21267, 21269, 21271, 21272, 21275, 21276, 21277, 21278, 21279, 21280, 21281, 21283, 21284, 21285,
    21286, 21287, 21288, 21289, 21290, 21291, 21292, 21293, 21294, 21295, 21297, 21298, 21299, 21300, 21301, 21303, 21304, 21305, 21306, 21308, 21309,
    21311, 21313, 21315, 21318, 21319, 21321, 21322, 21323, 21325, 21326, 21327, 21328, 21329, 21330, 21331, 21332, 21333, 21334, 21335, 21337, 21338,
    21339, 21340, 21341, 21342, 21343, 21344, 21345, 21346, 21347, 21349, 21350, 21351, 21352, 21353, 21355, 21356, 21358, 21359, 21360, 21361, 21362,
    21363, 21364, 21365, 21366, 21367, 21368, 21369, 21370, 21371, 21372, 21374, 21377, 21378, 21381, 21382, 21383, 21384, 21385, 21386, 21387, 21390,
    21395, 21396, 21397, 21398, 21399, 21402, 21403, 21405, 21406, 21407, 21408, 21410, 21411, 21412, 21413, 21414, 21418, 21421, 21422, 21423, 21425,
    21427, 21428, 21429, 21431, 21432, 21435, 21436, 21438, 21439, 21440, 21441, 21442, 21443, 21444, 21445, 21446, 21448, 21449, 21450, 21451, 21452,
    21453, 21455, 21456, 21457, 21458, 21459, 21460, 21461, 21462, 21463, 21465, 21467, 21468, 21469, 21470, 21472, 21474, 21475, 21476, 21478, 21479,
    21480, 21481, 21483, 21484, 21486, 21487, 21488, 21490, 21491, 21492, 21495, 21496, 21497, 21498, 21499, 21500, 21501, 21502, 21503, 21505, 21506,
    21507, 21508, 21509, 21510, 21511, 21513, 21514, 21515, 21516, 21517, 21518, 21519, 21520, 21522, 21523, 21524, 21525, 21526, 21528, 21529, 21530,
    21531, 21532, 21534, 21535, 21536, 21537, 21538, 21540, 21541, 21542, 21543, 21546, 21547, 21548, 21549, 21550, 21551, 21553, 21554, 21555, 21556,
    21571, 21573, 21577, 21578, 21579, 21581, 21583, 21584, 21585, 21586, 21587, 21588, 21589, 21590, 21592, 21593, 21594, 21595, 21597, 21598, 21600,
    21601, 21607, 21608, 21610, 21611, 21612, 21614, 21615, 21616, 21617, 21618, 21620, 21621, 21622, 21623, 21624, 21625, 21628, 21629, 21631, 21633,
    21635, 21636, 21637, 21638, 21640, 21641, 21642, 21643, 21644, 21645, 21647, 21651, 21652, 21653, 21654, 21655, 21658, 21660, 21661, 21662, 21663,
    21665, 21666, 21667, 21668, 21669, 21670, 21671, 21672, 21673, 21674, 21676, 21679, 21680, 21685, 21686, 21687, 21688, 21689, 21690, 21691, 21692,
    21694, 21695, 21697, 21699, 21700, 21701, 21702, 21703, 21705, 21706, 21707, 21708, 21709, 21710, 21711, 21712, 21713, 21714, 21715, 21716, 21717,
    21719, 21722, 21723, 21725, 21726, 21728, 21729, 21730, 21731, 21732, 21734, 21735, 21737, 21738, 21739, 21740, 21741, 21742, 21743, 21744, 21745,
    21746, 21747, 21749, 21750, 21751, 21752, 21753, 21754, 21755, 21756, 21757, 21758, 21759, 21760, 21761, 21762, 21763, 21764, 21765, 21767, 21769,
    21770, 21772, 21773, 21774, 21775, 21776, 21778, 21780, 21782, 21783, 21784, 21785, 21786, 21787, 21788, 21791, 21792, 21793, 21794, 21795, 21796,
    21797, 21799, 21800, 21801, 21802, 21803, 21804, 21805, 21807, 21808, 21809, 21811, 21812, 21813, 21814, 21815, 21816, 21817, 21818, 21819, 21820,
    21821, 21822, 21823, 21824, 21825, 21826, 21827, 21828, 21829, 21830, 21832, 21833, 21834, 21835, 21838, 21839, 21840, 21842, 21843, 21844, 21845,
    21847, 21848, 21849, 21850, 21851, 21853, 21854, 21855, 21857, 21859, 21860, 21861, 21863, 21864, 21865, 21867, 21868, 21870, 21871, 21872, 21873,
    21875, 21877, 21880, 21881, 21882, 21883, 21885, 21887, 21888, 21889, 21890, 21891, 21892, 21893, 21894, 21895, 21896, 21897, 21898, 21903, 21904,
    21905, 21906, 21907, 21908, 21909, 21911, 21912, 21913, 21914, 21915, 21916, 21918, 21920, 21921, 21922, 21923, 21924, 21925, 21926, 21927, 21928,
    21929, 21930, 21932, 21933, 21935, 21936, 21937, 21939, 21941, 21942, 21943, 21944, 21945, 21946, 21947, 21948, 21949, 21951, 21952, 21953, 21954,
    21955, 21957, 21958, 21959, 21960, 21961, 21962, 21963, 21964, 21965, 21968, 21969, 21970, 21972, 21973, 21974, 21975, 21976, 21977, 21978, 21979,
    21980, 21981, 21982, 21983, 21985, 21986, 21987, 21988, 21989, 21990, 21991, 21993, 21995, 21996, 21997, 21998, 21999, 22000, 22002, 22003, 22005,
    22006, 22008, 22010, 22011, 22012, 22014, 22015, 22018, 22019, 22020, 22021, 22023, 22025, 22027, 22028, 22029, 22030, 22031, 22032, 22033, 22034,
    22035, 22036, 22037, 22039, 22040, 22041, 22042, 22043, 22044, 22045, 22047, 22048, 22049, 22050, 22051, 22052, 22053, 22054, 22055, 22058, 22060,
    22061, 22062, 22063, 22066, 22067, 22070, 22072, 22073, 22074, 22075, 22077, 22078, 22079, 22081, 22082, 22083, 22084, 22085, 22086, 22087, 22088,
    22089, 22090, 22091, 22093, 22098, 22099, 22100, 22101, 22102, 22105, 22106, 22107, 22108, 22109, 22110, 22111, 22112, 22113, 22114, 22115, 22116,
    22117, 22118, 22119, 22120, 22121, 22122, 22123, 22125, 22126, 22127, 22129, 22130, 22131, 22132, 22133, 22134, 22136, 22137, 22139, 22140, 22141,
    22142, 22143, 22144, 22145, 22146, 22147, 22150, 22151, 22154, 22155, 22157, 22158, 22159, 22160, 22161, 22162, 22163, 22164, 22166, 22167, 22168,
    22170, 22171, 22172, 22173, 22175, 22176, 22179, 22180, 22181, 22182, 22183, 22184, 22185, 22186, 22187, 22188, 22189, 22190, 22191, 22193, 22194,
    22195, 22196, 22197, 22198, 22202, 22204, 22205, 22206, 22207, 22208, 22209, 22211, 22213, 22214, 22233, 22237, 22239, 22240, 22241, 22242, 22243,
    22244, 22245, 22246, 22247, 22248, 22250, 22251, 22252, 22253, 22254, 22255, 22256, 22257, 22258, 22259, 22260, 22261, 22263, 22264, 22265, 22266,
    22267, 22268, 22269, 22270, 22272, 22273, 22274, 22275, 22276, 22277, 22278, 22279, 22281, 22282, 22283, 22284, 22285, 22286, 22287, 22289, 22290,
    22291, 22292, 22293, 22295, 22296, 22297, 22298, 22299, 22300, 22301, 22302, 22304, 22305, 22306, 22307, 22308, 22310, 22312, 22313, 22314, 22315,
    22316, 22317, 22318, 22319, 22320, 22321, 22322, 22323, 22325, 22326, 22327, 22329, 22330, 22331, 22332, 22333, 22334, 22335, 22336, 22337, 22338,
    22340, 22341, 22342, 22343, 22345, 22346, 22347, 22349, 22351, 22352, 22353, 22354, 22356, 22357, 22358, 22359, 22360, 22362, 22363, 22365, 22366,
    22368, 22371, 22372, 22374, 22375, 22376, 22377, 22378, 22379, 22380, 22381, 22382, 22383, 22385, 22386, 22387, 22388, 22389, 22390, 22391, 22392,
    22393, 22394, 22395, 22396, 22397, 22398, 22400, 22402, 22403, 22404, 22405, 22406, 22407, 22408, 22409, 22412, 22413, 22414, 22415, 22416, 22417,
    22418, 22419, 22420, 22421, 22422, 22423, 22424, 22425, 22426, 22428, 22429, 22430, 22431, 22432, 22433, 22434, 22436, 22437, 22438, 22439, 22440,
    22441, 22442, 22444, 22445, 22447, 22448, 22449, 22450, 22451, 22453, 22454, 22455, 22456, 22457, 22458, 22459, 22460, 22465, 22468, 22469, 22470,
    22471, 22472, 22473, 22474, 22475, 22476, 22477, 22478, 22481, 22482, 22483, 22484, 22485, 22486, 22487, 22488, 22489, 22490, 22491, 22492, 22494,
    22495, 22499, 22500, 22502, 22504, 22505, 22506, 22507, 22508, 22509, 22512, 22514, 22515, 22517, 22518, 22519, 22520, 22521, 22522, 22523, 22524,
    22525, 22526, 22527, 22528, 22530, 22532, 22533, 22534, 22536, 22537, 22538, 22539, 22540, 22541, 22543, 22544, 22546, 22547, 22548, 22549, 22550,
    22552, 22553, 22555, 22556, 22557, 22560, 22562, 22563, 22564, 22565, 22566, 22567, 22568, 22569, 22576, 22577, 22579, 22580, 22581, 22585, 22586,
    22587, 22588, 22589, 22590, 22591, 22592, 22595, 22596, 22597, 22598, 22600, 22601, 22602, 22603, 22604, 22605, 22607, 22608, 22609, 22610, 22611,
    22612, 22613, 22614, 22615, 22616, 22618, 22620, 22621, 22622, 22623, 22624, 22625, 22626, 22628, 22629, 22631, 22632, 22633, 22634, 22635, 22636,
    22637, 22640, 22641, 22642, 22643, 22644, 22646, 22647, 22649, 22650, 22653, 22654, 22656, 22659, 22660, 22662, 22664, 22666, 22667, 22669, 22671,
    22672, 22674, 22675, 22676, 22677, 22679, 22681, 22682, 22684, 22685, 22686, 22690, 22691, 22693, 22695, 22700, 22702, 22703, 22706, 22707, 22709,
    22710, 22712, 22713, 22715, 22716, 22717, 22719, 22720, 22721, 22723, 22724, 22726, 22727, 22728, 22729, 22730, 22732, 22733, 22734, 22737, 22739,
    22741, 22742, 22745, 22747, 22748, 22749, 22750, 22751, 22752, 22753, 22754, 22755, 22756, 22757, 22758, 22759, 22760, 22761, 22763, 22764, 22766,
    22767, 22768, 22769, 22770, 22771, 22772, 22774, 22777, 22780, 22781, 22782, 22783, 22784, 22785, 22786, 22787, 22788, 22789, 22790, 22791, 22792,
    22793, 22794, 22795, 22797, 22798, 22799, 22800, 22801, 22802, 22803, 22804, 22805, 22806, 22807, 22808, 22809, 22810, 22811, 22812, 22813, 22814,
    22815, 22816, 22817, 22818, 22819, 22821, 22822, 22823, 22824, 22825, 22826, 22827, 22828, 22829, 22833, 22834, 22835, 22836, 22838, 22839, 22840,
    22842, 22843, 22844, 22845, 22846, 22847, 22848, 22850, 22852, 22853, 22854, 22855, 22856, 22857, 22858, 22859, 22860, 22861, 22862, 22864, 22866,
    22868, 22869, 22870, 22871, 22872, 22873, 22874, 22875, 22876, 22877, 22878, 22880, 22881, 22882, 22883, 22885, 22887, 22889, 22890, 22891, 22892,
    22893, 22894, 22895, 22896, 22897, 22898, 22899, 22900, 22901, 22902, 22903, 22905, 22906, 22907, 22909, 22910, 22911, 22912, 22913, 22914, 22915,
    22916, 22918, 22920, 22921, 22922, 22923, 22924, 22925, 22926, 22927, 22928, 22929, 22930, 22931, 22933, 22934, 22935, 22950, 22952, 22953, 22954,
    22956, 22957, 22958, 22959, 22960, 22961, 22962, 22963, 22964, 22965, 22966, 22967, 22968, 22969, 22970, 22971, 22973, 22974, 22975, 22976, 22977,
    22978, 22980, 22981, 22982, 22983, 22984, 22985, 22986, 22987, 22988, 22989, 22990, 22991, 22992, 22993, 22995, 22996, 22997, 22998, 23000, 23001,
    23002, 23003, 23004, 23005, 23006, 23007, 23008, 23009, 23010, 23011, 23012, 23013, 23014, 23015, 23016, 23017, 23018, 23019, 23020, 23021, 23022,
    23023, 23024, 23025, 23027, 23028, 23029, 23030, 23031, 23032, 23033, 23038, 23039, 23040, 23041, 23042, 23043, 23044, 23045, 23046, 23047, 23048,
    23049, 23050, 23051, 23052, 23053, 23054, 23055, 23056, 23057, 23058, 23060, 23061, 23062, 23063, 23064, 23065, 23067, 23068, 23071, 23073, 23074,
    23075, 23076, 23077, 23078, 23079, 23080, 23081, 23083, 23084, 23085, 23086, 23087, 23088, 23089, 23091, 23092, 23093, 23094, 23095, 23096, 23097,
    23098, 23099, 23100, 23101, 23102, 23103, 23105, 23106, 23107, 23108, 23109, 23111, 23113, 23114, 23115, 23116, 23117, 23118, 23119, 23120, 23121,
    23122, 23123, 23124, 23125, 23126, 23127, 23128, 23129, 23130, 23131, 23132, 23133, 23134, 23135, 23136, 23137, 23138, 23141, 23142, 23143, 23144,
    23146, 23148, 23149, 23151, 23152, 23153, 23154, 23156, 23157, 23158, 23160, 23161, 23162, 23163, 23164, 23165, 23166, 23167, 23168, 23169, 23170,
    23171, 23172, 23173, 23174, 23175, 23176, 23177, 23178, 23179, 23181, 23183, 23184, 23185, 23186, 23187, 23188, 23189, 23190, 23191, 23192, 23194,
    23195, 23196, 23197, 23198, 23199, 23200, 23202, 23203, 23204, 23205, 23206, 23207, 23208, 23210, 23211, 23215, 23216, 23217, 23218, 23219, 23221,
    23222, 23223, 23224, 23225, 23226, 23229, 23231, 23232, 23233, 23234, 23236, 23237, 23238, 23239, 23241, 23243, 23244, 23245, 23247, 23248, 23249,
    23250, 23253, 23254, 23255, 23256, 23257, 23258, 23259, 23260, 23261, 23263, 23264, 23265, 23266, 23267, 23268, 23269, 23270, 23271, 23272, 23273,
    23274, 23275, 23276, 23277, 23278, 23279, 23280, 23281, 23282, 23283, 23284, 23286, 23287, 23288, 23289, 23291, 23292, 23293, 23294, 23295, 23297,
    23298, 23299, 23300, 23301, 23302, 23303, 23304, 23305, 23306, 23307, 23308, 23309, 23310, 23311, 23312, 23315, 23316, 23317, 23318, 23319, 23320,
    23321, 23322, 23323, 23324, 23325, 23326, 23328, 23329, 23330, 23331, 23332, 23333, 23334, 23335, 23336, 23339, 23340, 23342, 23343, 23344, 23347,
    23348, 23349, 23350, 23352, 23354, 23358, 23359, 23360, 23361, 23362, 23363, 23365, 23367, 23368, 23369, 23371, 23372, 23373, 23374, 23376, 23377,
    23378, 23379, 23380, 23382, 23383, 23384, 23385, 23386, 23387, 23388, 23389, 23390, 23391, 23392, 23393, 23394, 23395, 23396, 23397, 23399, 23400,
    23401, 23402, 23403, 23404, 23405, 23406, 23407, 23408, 23409, 23412, 23413, 23419, 23420, 23421, 23422, 23423, 23424, 23425, 23427, 23428, 23429,
    23430, 23431, 23432, 23433, 23434, 23435, 23436, 23437, 23438, 23440, 23469, 23470, 23471, 23472, 23473, 23474, 23475, 23476, 23477, 23478, 23480,
    23481, 23482, 23483, 23484, 23485, 23486, 23487, 23489, 23490, 23491, 23492, 23493, 23495, 23496, 23497, 23498, 23499, 23500, 23501, 23502, 23503,
    23505, 23506, 23507, 23508, 23509, 23510, 23511, 23513, 23515, 23516, 23517, 23518, 23519, 23520, 23521, 23522, 23523, 23524, 23525, 23526, 23528,
    23529, 23530, 23531, 23532, 23533, 23534, 23535, 23536, 23537, 23538, 23539, 23540, 23541, 23542, 
           23554, 23556, 23559, 23561, 23563, 23564, 23565, 23567, 23568, 23569, 23570, 23571, 23572, 23573, 23574, 23575, 23576, 23577, 23578, 23581,
    23582, 23584, 23585, 23586, 23587, 23588, 23591, 23592, 23593, 23595, 23596, 23597, 23598, 23599, 23600, 23601, 23603, 23604, 23607, 23609, 23610,
    23611, 23612, 23613, 23614, 23615, 23618, 23622, 23623, 23624, 23625, 23626, 23627, 23628, 23629, 23630, 23631, 23632, 23633, 23634, 23635, 23636,
    23637, 23638, 23639, 23640, 23641, 23642, 23643, 23645, 23646, 23648, 23649, 23650, 23651, 23652, 23653, 23654, 23655, 23656, 23657, 23658, 23659,
    23660, 23661, 23662, 23663, 23664, 23666, 23669, 23670, 23671, 23672, 23673, 23674, 23676, 23678, 23679, 23680, 23681, 23682, 23683, 23684, 23685,
    23686, 23688, 23689, 23690, 23691, 23692, 23693, 23694, 23695, 23697, 23698, 23699, 23701, 23702, 23704, 23706, 23707, 23708, 23709, 23710, 23712,
    23713, 23714, 23715, 23716, 23717, 23718, 23719, 23720, 23721, 23722, 23723, 23724, 23725, 23726, 23727, 23728, 23729, 23730, 23732, 23734, 23736,
    23738, 23739, 23740, 23742, 23743, 23744, 23754, 23755, 23756, 23757, 23758, 23760, 23761, 23762, 23763, 23764, 23765, 23766, 23767, 23768, 23769,
    23770, 23771, 23772, 23773, 23774, 23775, 23776, 23777, 23778, 23780, 23781, 23782, 23783, 23784, 23785, 23786, 23787, 23788, 23789, 23790, 23793,
    23795, 23796, 23797, 23798, 23801, 23802, 23803, 23804, 23805, 23806, 23807, 23808, 23809, 23811, 23812, 23813, 23814, 23815, 23816, 23817, 23818,
    23819, 23821, 23822, 23824, 23825, 23826, 23827, 23828, 23829, 23830, 23831, 23832, 23833, 23835, 23836, 23837, 23838, 23839, 23840, 23841, 23842,
    23844, 23845, 23846, 23848, 23849, 23851, 23852, 23853, 23854, 23855, 23856, 23857, 23859, 23860, 23861, 23862, 23863, 23864, 23865, 23866, 23867,
    23868, 23887, 23889, 23891, 23892, 23893, 23897, 23899, 23900, 23901, 23902, 23903, 23904, 23905, 23907, 23908, 23909, 23910, 23912, 23915, 23916,
    23918, 23919, 23920, 23922, 23923, 23924, 23925, 23926, 23928, 23931, 23932, 23933, 23934, 23935, 23938, 23940, 23942, 23944, 23945, 23946, 23947,
    23948, 23949, 23950, 23951, 23952, 23954, 23955, 23957, 23958, 23959, 23961, 23963, 23964, 23965, 23966, 23967, 23968, 23969, 23974, 23976, 23977,
    23978, 23979, 23980, 23981, 23982, 23983, 23985, 23986, 23988, 23989, 23990, 23991, 23992, 23993, 23994, 23995, 23996, 23997, 23998, 23999, 24000,
    24002, 24003, 24004, 24005, 24006, 24007, 24008, 24010, 24011, 24012, 24014, 24015, 24016, 24017, 24018, 24019, 24020, 24021, 24022, 24023, 24024,
    24025, 24026, 24027, 24028, 24029, 24030, 24031, 24032, 24033, 24034, 24035, 24055, 24056, 24057, 24058, 24060, 24061, 24062, 24064, 24065, 24066,
    24067, 24068, 24069, 24070, 24071, 24072, 24074, 24075, 24077, 24078, 24079, 24080, 24081, 24082, 24083, 24084, 24085, 24088, 24089, 24091, 24092,
    24093, 24094, 24095, 24096, 24097, 24098, 24100, 24101, 24102, 24103, 24104, 24106, 24107, 24108, 24109, 24110, 24111, 24112, 24113, 24115, 24116,
    24117, 24119, 24121, 24122, 24123, 24124, 24135, 24136, 24138, 24139, 24141, 24142, 24143, 24144, 24145, 24147, 24149, 24150, 24151, 24152, 24153,
    24154, 24155, 24156, 24159, 24160, 24162, 24163, 24164, 24165, 24166, 24167, 24168, 24169, 24170, 24171, 24172, 24174, 24175, 24176, 24178, 24179,
    24180, 24181, 24182, 24183, 24184, 24185, 24188, 24189, 24190, 24191, 24219, 24222, 24223, 24224, 24225, 24226, 24228, 24231, 24232, 24233, 24238,
    24240, 24241, 24242, 24243, 24244, 24247, 24248, 24249, 24250, 24252, 24253, 24255, 24256, 24257, 24258, 24259, 24260, 24261, 24263, 24264, 24265,
    24266, 24267, 24268, 24269, 24271, 24272, 24274, 24275, 24277, 24278, 24279, 24280, 24281, 24282, 24283, 24284, 24285, 24286, 24287, 24288, 24289,
    24290, 24291, 24292, 24293, 24294, 24295, 24296, 24298, 24299, 24300, 24303, 24304, 24305, 24307, 24308, 24309, 24310, 24311, 24312, 24313, 24314,
    24317, 24319, 24320, 24321, 24322, 24323, 24324, 24325, 24326, 24327, 24328, 24329, 24330, 24332, 24333, 24334, 24335, 24336, 24337, 24338, 24339,
    24340, 24341, 24342, 24343, 24344, 24345, 24346, 24347, 24350, 24351, 24352, 24353, 24354, 24355, 24357, 24358, 24359, 24361, 24363, 24364, 24365,
    24366, 24367, 24368, 24369, 24370, 24371, 24373, 24374, 24375, 24376, 24378, 24380, 24381, 24386, 24387, 24388, 24389, 24391, 24392, 24393, 24395,
    24397, 24400, 24401, 24403, 24410, 24411, 24412, 24413, 24416, 24417, 24418, 24420, 24421, 24422, 24423, 24424, 24425, 24426, 24430, 24431, 24433,
    24434, 24435, 24437, 24438, 24440, 24441, 24442, 24443, 24444, 24445, 24446, 24447, 24448, 24450, 24451, 24452, 24453, 24455, 24456, 24457, 24458,
    24459, 24460, 24461, 24462, 24463, 24465, 24467, 24468, 24470, 24471, 24472, 24473, 24474, 24475, 24476, 24477, 24478, 24479, 24481, 24482, 24483,
    24486, 24487, 24488, 24489, 24490, 24492, 24493, 24494, 24495, 24496, 24497, 24498, 24499, 24503, 24504, 24505, 24506, 24507, 24509, 24510, 24512,
    24513, 24514, 24516, 24517, 24518, 24520, 24521, 24522, 24523, 24524, 24525, 24526, 24528, 24529, 24530, 24532, 24533, 24534, 24536, 24537, 24539,
    24540, 24541, 24542, 24544, 24545, 24547, 24548, 24549, 24550, 24552, 24554, 24555, 24557, 24558, 24559, 24560, 24561, 24562, 24563, 24564, 24565,
    24567, 24568, 24569, 24570, 24571, 24573, 24574, 24575, 24576, 24577, 24578, 24580, 24581, 24582, 24583, 24584, 24585, 24586, 24587, 24588, 24589,
    24590, 24591, 24593, 24594, 24596, 24597, 24598, 24599, 24600, 24602, 24603, 24604, 24605, 24606, 24608, 24609, 24610, 24611, 24612, 24613, 24614,
    24615, 24616, 24617, 24618, 24619, 24620, 24621, 24622, 24623, 24624, 24625, 24626, 24627, 24628, 24629, 24630, 24631, 24632, 24633, 24634, 24635,
    24636, 24637, 24639, 24640, 24641, 24642, 24643, 24644, 24646, 24647, 24648, 24649, 24650, 24651, 24652, 24653, 24655, 24656, 24657, 24658, 24660,
    24662, 24664, 24665, 24667, 24668, 24669, 24670, 24672, 24673, 24674, 24675, 24676, 24677, 24678, 24679, 24680, 24681, 24683, 24684, 24685, 24686,
    24687, 24689, 24690, 24691, 24692, 24693, 24694, 24695, 24696, 24697, 24698, 24699, 24700, 24701, 24702, 24703, 24704, 24705, 24706, 24707, 24708,
    24709, 24710, 24711, 24714, 24715, 24717, 24721, 24722, 24723, 24725, 24726, 24727, 24729, 24730, 24731, 24732, 24733, 24734, 24738, 24739, 24740,
    24741, 24742, 24743, 24744, 24745, 24747, 24749, 24752, 24753, 24754, 24755, 24758, 24760, 24761, 24762, 24764, 24765, 24766, 24767, 24769, 24770,
    24772, 24773, 24774, 24775, 24776, 24777, 24779, 24780, 24781, 24782, 24783, 24784, 24785, 24786, 24787, 24788, 24789, 24790, 24791, 24792, 24793,
    24794, 24795, 24797, 24799, 24800, 24801, 24802, 24804, 24805, 24806, 24807, 24808, 24809, 24810, 24811, 24812, 24813, 24815, 24818, 24819, 24821,
    24822, 24823, 24824, 24825, 24826, 24827, 24829, 24830, 24831, 24832, 24833, 24834, 24835, 24836, 24837, 24838, 24839, 24840, 24841, 24842, 24843,
    24844, 24845, 24846, 24847, 24848, 24850, 24851, 24852, 24853, 24854, 24855, 24856, 24857, 24858, 24861, 24862, 24863, 24865, 24866, 24867, 24868,
    24869, 24871, 24872, 24874, 24875, 24876, 24877, 24878, 24879, 24880, 24881, 24883, 24884, 24885, 24886, 24887, 24888, 24889, 24890, 24891, 24892,
    24894, 24895, 24896, 24897, 24898, 24899, 24900, 24901, 24902, 24903, 24904, 24905, 24906, 24908, 24909, 24911, 24912, 24913, 24914, 24915, 24916,
    24919, 24920, 24921, 24922, 24923, 24924, 24925, 24929, 24930, 24931, 24932, 24933, 24934, 24935, 24936, 24937, 24939, 24940, 24941, 24942, 24943,
    24946, 24947, 24948, 24949, 24950, 24951, 24952, 24953, 24954, 24955, 24957, 24959, 24960, 24961, 24962, 24963, 24965, 24966, 24968, 24969, 24970,
    24971, 24972, 24974, 24976, 24977, 24978, 24979, 24983, 24985, 24986, 24987, 24988, 24989, 24990, 24991, 24993, 24995, 24996, 24997, 24998, 24999,
    25000, 25001, 25002, 25003, 25005, 25006, 25007, 25008, 25009, 25010, 25011, 25012, 25013, 25014, 25015, 25017, 25019, 25022, 25024, 25025, 25026,
    25027, 25030, 25031, 25032, 25034, 25035, 25036, 25037, 25038, 25039, 25040, 25041, 25042, 25045, 25046, 25047, 25048, 25049, 25050, 25051, 25052,
    25053, 25054, 25056, 25058, 25059, 25060, 25062, 25064, 25065, 25066, 25067, 25068, 25069, 25070, 25071, 25072, 25074, 25076, 25077, 25078, 25080,
    25081, 25082, 25083, 25084, 25085, 25087, 25088, 25089, 25090, 25091, 25093, 25094, 25095, 25096, 25097, 25098, 25099, 25105, 25106, 25107, 25108,
    25109, 25110, 25111, 25112, 25113, 25115, 25117, 25118, 25120, 25121, 25122, 25123, 25124, 25125, 25126, 25127, 25128, 25129, 25130, 25131, 25132,
    25134, 25135, 25137, 25138, 25139, 25140, 25141, 25142, 25143, 25144, 25145, 25147, 25149, 25150, 25152, 25153, 25154, 25155, 25156, 25157, 25158,
    25159, 25163, 25164, 25165, 25166, 25167, 25168, 25169, 25170, 25171, 25172, 25173, 25174, 25175, 25177, 25178, 25179, 25181, 25183, 25184, 25185,
    25186, 25187, 25191, 25192, 25193, 25194, 25195, 25196, 25197, 25198, 25199, 25200, 25201, 25202, 25203, 25204, 25205, 25206, 25207, 25208, 25209,
    25210, 25211, 25212, 25213, 25214, 25215, 25216, 25217, 25218, 25219, 25220, 25221, 25222, 25223, 25224, 25226, 25227, 25228, 25230, 25231, 25232,
    25233, 25234, 25237, 25238, 25239, 25240, 25241, 25242, 25243, 25245, 25246, 25248, 25251, 25252, 25254, 25255, 25256, 25257, 25258, 25259, 25260,
    25261, 25262, 25263, 25264, 25265, 25266, 25267, 25268, 25269, 25270, 25272, 25273, 25274, 25275, 25276, 25277, 25278, 25279, 25280, 25281, 25282,
    25283, 25284, 25287, 25288, 25289, 25290, 25291, 25294, 25295, 25297, 25298, 25299, 25300, 25301, 25302, 25303, 25304, 25307, 25308, 25310, 25312,
    25313, 25314, 25315, 25316, 25317, 25319, 25320, 25321, 25322, 25323, 25324, 25325, 25326, 25327, 25328, 25329, 25330, 25332, 25333, 25334, 25336,
    25337, 25338, 25339, 25340, 25341, 25342, 25343, 25344, 25345, 25346, 25347, 25348, 25349, 25350, 25352, 25353, 25354, 25356, 25359, 25360, 25361,
    25362, 25364, 25366, 25367, 25368, 25369, 25370, 25372, 25374, 25375, 25377, 25378, 25379, 25380, 25381, 25383, 25384, 25385, 25387, 25388, 25389,
    25390, 25391, 25392, 25393, 25394, 25395, 25397, 25399, 25400, 25401, 25403, 25404, 25405, 25408, 25409, 25410, 25411, 25412, 25413, 25414, 25415,
    25416, 25417, 25420, 25421, 25423, 25424, 25426, 25428, 25429, 25431, 25433, 25434, 25435, 25437, 25439, 25440, 25441, 25443, 25444, 25445, 25446,
    25447, 25448, 25449, 25450, 25451, 25452, 25524, 25528, 25529, 25530, 25531, 25533, 25534, 25535, 25536, 25537, 25538, 25539, 25542, 25543, 25544,
    25545, 25546, 25547, 25548, 25549, 25550, 25551, 25552, 25553, 25554, 25555, 25559, 25562, 25564, 25566, 25567, 25568, 25569, 25570, 25571, 25573,
    25574, 25575, 25576, 25577, 25578, 25579, 25581, 25582, 25583, 25584, 25585, 25587, 25588, 25589, 25590, 25591, 25592, 25593, 25594, 25595, 25597,
    25599, 25600, 25601, 25602, 25604, 25606, 25609, 25610, 25611, 25614, 25615, 25616, 25618, 25619, 25621, 25622, 25623, 25624, 25625, 25626, 25628,
    25629, 25630, 25633, 25634, 25635, 25636, 25638, 25639, 25642, 25643, 25644, 25645, 25646, 25647, 25648, 25651, 25652, 25653, 25654, 25655, 25656,
    25657, 25658, 25659, 25660, 25661, 25662, 25663, 25665, 25666, 25667, 25668, 25669, 25670, 25671, 25672, 25673, 25674, 25675, 25677, 25678, 25679,
    25680, 25681, 25682, 25683, 25684, 25686, 25687, 25688, 25689, 25690, 25691, 25692, 25693, 25694, 25695, 25696, 25697, 25701, 25702, 25703, 25704,
    25705, 25706, 25708, 25709, 25710, 25711, 25712, 25713, 25714, 25716, 25717, 25718, 25719, 25720, 25721, 25723, 25725, 25726, 25727, 25728, 25729,
    25730, 25731, 25732, 25733, 25735, 25736, 25737, 25739, 25740, 25741, 25742, 25743, 25745, 25746, 25748, 25749, 25751, 25752, 25753, 25755, 25756,
    25758, 25759, 25762, 25763, 25764, 25765, 25766, 25767, 25768 ]

    df = df[df['run'].isin(good_runs)]
    return df



##### old to new flux ######
