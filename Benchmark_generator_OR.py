import numpy as np
import pandas as pd
import copy
import P1DFG
import DSR



def Benchmark_generator_ORF(Imb_lvl=3, th_c=0.6, th_r=0.6, LOG=False, Mutations = False, Sim_matrix_r = 'ALL_625.npy' ):
    
    import dataframe_utils as dfu
    
    
    
    if LOG:print('-'*45)
    if LOG:print('Importing data')
    if LOG:print('-'*45)
    M2OR_full_data = dfu.import_M2OR()
    if LOG:print("M2OR data has been imported.")
    if LOG:print("M2OR_receptors data has been imported.")
    M2OR_compounds = dfu.import_M2OR_compounds()
    print("M2OR_compound data has been imported.")
    if LOG:print('-'*45)
    
    # dfu.plot_receptor_ligand_pairs(M2OR_full_data)
    
    
    #%% Filterin Mutations
    
    if LOG:print('Filtering Mutations')
    if Mutations == False:
    
        receptors = pd.read_csv('M2OR/main_receptors.csv', sep=';',index_col='id')
        mutations_mask = receptors['mutation'].notna()
        receptors_with_mutation = ((receptors[mutations_mask]).index).tolist()   
        M2OR_mutations, M2OR_Not_mutations = dfu.search_experiments(M2OR_full_data, 'main_receptors_id', receptors_with_mutation, True)
        M2OR_FILT1 = M2OR_Not_mutations
    else:
        M2OR_FILT1 = M2OR_full_data 
    # dfu.plot_receptor_ligand_pairs(M2OR_full_data, highlight_df=M2OR_mutations)
    # dfu.plot_receptor_ligand_pairs(M2OR_FILT1)
    #%% Filterin not single molecule pairs
    if LOG:print('Filtering not single molecule pairs')
    
    compounds = pd.read_csv('M2OR/main_compounds.csv', sep=';',index_col='id')
    compounds_mask = compounds['mixture']!='mono'
    compounds_not_mono = ((compounds[compounds_mask]).index).tolist() 
    M2OR_MONO, M2OR_Not_MONO = dfu.search_experiments(M2OR_FILT1, 'main_compounds_id', compounds_not_mono, True)
    M2OR_FILT1  = M2OR_Not_MONO
    
    #%%
    if LOG:print('Filtering Infinity Inbalance')
    M2OR_data_filt = P1DFG.filter_inf_imb(M2OR_FILT1,Imb_lvl,False) #filter experiments with only negative or positive responses.
    M2OR_INF_INB = M2OR_FILT1.loc[~M2OR_FILT1.index.isin(M2OR_data_filt.index)]
    
    # dfu.plot_receptor_ligand_pairs(M2OR_FILT1, highlight_df=M2OR_INF_INB)
    # dfu.plot_receptor_ligand_pairs(M2OR_FILT1, highlight_df=M2OR_INF_INB, x_quantile=0.85)
    # dfu.plot_receptor_ligand_pairs(M2OR_INF_INB)
    # dfu.plot_receptor_ligand_pairs(M2OR_data_filt)
    
    # distribution_by_receptors = dfu.get_experiments_distribution(M2OR_data_filt, dfu.Receptor_id)
    # distribution_by_compound = dfu.get_experiments_distribution(M2OR_data_filt, dfu.Compound_id)
    
    # size_by_receptors = dfu.get_size_df(distribution_by_receptors, Imb_lvl)
    # size_by_compound  = dfu.get_size_df(distribution_by_compound , Imb_lvl)
    
    #%%
    if LOG:print('-'*45)    
    if LOG:print('Getting_size_by_receptors')
    size_by_receptor = dfu.get_size_df((dfu.get_experiments_distribution(M2OR_data_filt, dfu.Receptor_id)), Imb_lvl)
    if LOG:print('-'*45) 
    if LOG:print('Getting_size_by_compounds')
    size_by_compound  = dfu.get_size_df((dfu.get_experiments_distribution(M2OR_data_filt, dfu.Compound_id)), Imb_lvl)
    if LOG:print('-'*45)    
    
    
    
    
    #%%
    if LOG:print('Pair Assigment')
    if LOG:print('-'*45) 
    if LOG:print('-'*45) 
    
    size_by_receptor = P1DFG.Get_All_d_Exp(size_by_receptor, 'main_receptors_id', M2OR_data_filt)
    size_by_compound = P1DFG.Get_All_d_Exp(size_by_compound, 'main_compounds_id', M2OR_data_filt)
    
    
    # P1DFG.plot_bubble(size_by_receptor, title='Receptor Imbalance')
    # P1DFG.plot_bubble(size_by_compound, title='Ligant Imbalance')
    #%%
    
    size_by_receptor, size_by_compound = DSR.Pos_receptor_Soft(size_by_receptor, size_by_compound, M2OR_compounds,M2OR_data_filt, LOG=True)
    size_by_receptor, size_by_compound = DSR.Pos_Hard_Method(size_by_receptor, size_by_compound , M2OR_data_filt, LOG=True)
    #%%

    size_by_receptor, size_by_compound = DSR.Neg_receptor_Soft(size_by_receptor, size_by_compound, M2OR_compounds,M2OR_data_filt,LOG=True)
    #%%
    Rec_e = dfu.search_experiments(M2OR_data_filt,'main_receptors_id', 102)
    # dfu.plot_receptor_ligand_pairs(M2OR_data_filt, highlight_df=Rec_e)
    Lig_e = dfu.search_experiments(M2OR_data_filt,'main_compounds_id', Rec_e['main_compounds_id'])
    Lig_e2 = dfu.search_experiments(M2OR_data_filt,'main_compounds_id', [112, 122])
    
    # dfu.plot_receptor_ligand_pairs(Lig_e, highlight_df=Rec_e)
    # dfu.plot_receptor_ligand_pairs(Rec_e)
    # dfu.plot_receptor_ligand_pairs(Lig_e, highlight_df=Lig_e2)
    # dfu.plot_receptor_ligand_pairs(Lig_e2)
    size_by_compound_e  = dfu.get_size_df((dfu.get_experiments_distribution(Lig_e2, dfu.Compound_id)), Imb_lvl)
    # P1DFG.plot_bubble(size_by_compound_e, title='Ligant Imbalance')
    
    # rec2_ids = [6043, 10124, 15265, 24427, 26246, 30549, 31431, 7373, 28047]
    # rec3_ids = [6871, 7759, 8578, 9695, 10949, 11400, 11855, 12300, 12768, 13934, 15710, 16793, 18428, 19983, 20449, 20897, 21332, 21706, 22134, 22598, 23036, 23504, 23944, 24914, 25807, 26698, 28586, 29045, 29491, 30998, 32404, 32841, 34662, 35159, 35689, 36204, 37064, 37944, 38771, 39868, 41147, 41583]
    # rec2_df = M2OR_data_filt.loc[rec2_ids] 
    # rec3_df = M2OR_data_filt.loc[rec3_ids] 
    # dfu.plot_receptor_ligand_pairs(Rec_e)
    # dfu.plot_receptor_ligand_pairs(Rec_e, highlight_df=rec2_df)
    # dfu.plot_receptor_ligand_pairs(Rec_e, highlight_df=rec3_df)
    
    
    #%%
    
    size_by_receptor, size_by_compound = DSR.Neg_Hard_Method(size_by_receptor, size_by_compound, M2OR_data_filt, LOG=True)
    
    
    size_by_receptor, size_by_compound = DSR.Pos_Ligant_Hard_Method(size_by_receptor, size_by_compound , M2OR_data_filt)
    size_by_receptor, size_by_compound = DSR.Neg_Ligant_Hard_Method(size_by_receptor, size_by_compound , M2OR_data_filt)
    
    size_by_receptor, size_by_compound = DSR.Fix_balance(size_by_receptor, size_by_compound, Imb_lvl, M2OR_data_filt)
    
    
     #%%
     
    List_R = copy.deepcopy(size_by_receptor)
    List_L = copy.deepcopy(size_by_compound)
    
    
    
    
    #%%
    
    # ——— compute with old method ———
    old_neg_cr = sum(np.array(List_R['Negative']))
    old_neg_cl = sum(np.array(List_L['Negative']))
    old_pos_cr = sum(np.array(List_R['Positive']))
    old_pos_cl = sum(np.array(List_L['Positive']))
    
    old_total_cr = old_neg_cr + old_pos_cr
    old_total_cl = old_neg_cl + old_pos_cl
    
    # ——— compute with new method ———
    new_neg_cr = List_R['pair_id_neg'].apply(len).sum()
    new_neg_cl = List_L['pair_id_neg'].apply(len).sum()
    new_pos_cr = List_R['pair_id_pos'].apply(len).sum()
    new_pos_cl = List_L['pair_id_pos'].apply(len).sum()
    
    new_total_cr = new_neg_cr + new_pos_cr
    new_total_cl = new_neg_cl + new_pos_cl
    
    # ——— build comparison table ———
    summary = pd.DataFrame({
        'Datapoints Require': [
            old_neg_cr, old_neg_cl,
            old_pos_cr, old_pos_cl,
            old_total_cr, old_total_cl
        ],
        'Datapoints Assigne': [
            new_neg_cr, new_neg_cl,
            new_pos_cr, new_pos_cl,
            new_total_cr, new_total_cl
        ]
    }, index=[
        'Neg_CR','Neg_CL',
        'Pos_CR','Pos_CL',
        'Total_CR','Total_CL'
    ])
    
    # ——— print it neatly ———
    if LOG:print()
    if LOG:print(summary.to_string())
    if LOG:print()
    
     #%% 
    import dataframe_utils as dfu 
    
    All_pairs_BM =  dfu.flatten_pair_ids(List_L);
    M2OR_BM_DATASET = M2OR_full_data.loc[All_pairs_BM]
    # dfu.plot_receptor_ligand_pairs(M2OR_BM_DATASET,size = 50)
    
     #%%   
     
    
    # P1DFG.plot_bubble(List_R, title='Receptor Imbalance')
    # P1DFG.plot_bubble(List_L, title='Ligant Imbalance')
     #%% 
    
    # import dataframe_utils as dfu 
    
    # distribution_by_receptors_N = dfu.get_experiments_distribution(M2OR_full_data, dfu.Receptor_id)
    # size_by_receptors_N = dfu.get_size_df(distribution_by_receptors_N, Imb_lvl)
    
    # extra_indices = size_by_receptors_N.index.difference(size_by_receptor.index)
    # positions = [size_by_receptors_N.index.get_loc(idx) for idx in extra_indices]
    # positions = np.array(positions)
    
    
    print('xxxxxxxxxxxxxxx')
  
    
    splits_by_cluster_receptors,List_R_c = dfu.clustering_and_split_receptors(List_R,th_r, Sim_matrix_r)
    
    
    if LOG:print("splits_by_cluster_receptors")
    
    splits_by_cluster_compound,List_L_c = dfu.clustering_and_split_compounds(M2OR_compounds,List_L,th_c)
    
    print('xxxxxxxxxxxxxxxx')
    if LOG:print("splits_by_cluster_compound")
    
    
    #%% 
    # print('xxxxxxxxxxxxxxxx')
    pair_id_bm_receptor_fold = []
    pair_id_bm_ligant_fold = []
    
    for i in range(len(splits_by_cluster_receptors)):
        if LOG:print(i)
        list_de = splits_by_cluster_receptors[i].tolist()
        Cluster_N = dfu.search_experiments(List_R, 'cluster', list_de)
        All_pairs_BM = dfu.flatten_pair_ids(Cluster_N)
        M2OR_BM_DATASET = M2OR_full_data.loc[All_pairs_BM]
        # dfu.plot_receptor_ligand_pairs(M2OR_BM_DATASET, size=50)
        if LOG:print(len(M2OR_BM_DATASET))
    
        # append to the end of the list instead of assigning by index
        pair_id_bm_receptor_fold.append(All_pairs_BM)
        
        
    # print('xxxxxxxxxxxxxxx')
        
    for j in range(len(splits_by_cluster_compound)):
        if LOG:print(j)
        list_de = (splits_by_cluster_compound[j]).tolist()
        Cluster_N = dfu.search_experiments(List_L,'cluster', list_de)
        
        All_pairs_BM =  dfu.flatten_pair_ids(Cluster_N);
        M2OR_BM_DATASET = M2OR_full_data.loc[All_pairs_BM]
        # dfu.plot_receptor_ligand_pairs(M2OR_BM_DATASET,size = 50)
        if LOG:print(len(M2OR_BM_DATASET ))
        
        pair_id_bm_ligant_fold.append(All_pairs_BM)
        
        

    
    #%%   
    
    
    return  pair_id_bm_receptor_fold, pair_id_bm_ligant_fold, List_L, List_R
