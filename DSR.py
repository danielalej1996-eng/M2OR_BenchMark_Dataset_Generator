import P1DFG
import Similitud_compounds as sc
import numpy as np
import pandas as pd
import dataframe_utils as dfu


def Pos_receptor_Soft(
    List_R,
    List_L,
    M2OR_compounds,
    M2OR_data_filt,
    LOG = False
):
    """Ensure each receptor keeps only the required number of positive pairs.

    The function iterates over every receptor in **``List_R``** and removes
    excess positive ligand–receptor pairs, prioritising chemical diversity.

    **Key columns expected in the inputs**
    -------------------------------------
    ``List_R``
        * ``Exp_Av_P``   – list‑like column containing *pair_ids* (positive)
        * ``Positive``   – *int*, required number of positive pairs
        * ``pos_cond_c`` – numeric flag (> 1 ⇒ receptor currently exceeds quota)

    ``List_L`` (ligand‑centric table)
        * Same column set as ``List_R`` (but ligand‑oriented)

    ``M2OR_data_filt`` (experiment table, *indexed by pair_id*)
        * ``main_compounds_id``
        * ``main_receptors_id``

    Parameters
    ----------
    List_R, List_L : ``pd.DataFrame``
    LOG            : ``bool`` – master verbosity switch
    M2OR_compounds : ``pd.DataFrame`` – compound descriptors (for similarity)
    M2OR_data_filt : ``pd.DataFrame`` – filtered experiment metadata

    Returns
    -------
    tuple (``List_R``, ``List_L``) – updated DataFrames with excess positive
    pairs removed.
    """
    
    
    
    # ------------------------------------------------------------------
    # Pre‑computation & initial bookkeeping
    # ------------------------------------------------------------------
    if LOG:
        print("-" * 45)
    Compound_sim_matrix = sc.Compounds_sim_matrix(M2OR_compounds)
    if LOG:
        print("-" * 45)

    # Make sure helper columns are present/updated
    List_R["pair_id_pos"] = List_R["Exp_Av_P"]
    List_L["pair_id_pos"] = List_L["Exp_Av_P"]
    List_R, List_L = P1DFG.updated_pos_conditions(List_R, List_L)

    # ------------------------------------------------------------------
    # Main loop – iterate over receptors
    # ------------------------------------------------------------------
    for i in range(len(List_R)):
        # For debug purposes, receptor index 2 is always logged verbosely
        #LOG = LOG #or (i == 2)

        rec_id = List_R.index[i]

        # Skip receptors that already fulfil their quota
        if List_R.at[rec_id, "pos_cond_c"] <= 1:
            continue

        # ------------------------------------------------------------------
        # Gather receptor‑level information
        # ------------------------------------------------------------------
        Pos_PA = List_R.at[rec_id, "Positive"]          # required pair count
        Pairs_id_r = List_R.at[rec_id, "pair_id_pos"]    # current positive pairs

        if LOG:
            print(f"Receptor: {rec_id}")
            print(f"Positive pairs required: {Pos_PA}")
            print(f"Positive pairs available: {len(Pairs_id_r)}")
            print("-" * 45)
            print(
                f"Criteria: remove {len(Pairs_id_r) - Pos_PA} positive pairs "
                f"for receptor {rec_id}"
            )
            print(
                "Priority: maximise ligand variety. "
                "The most diverse ligands will be kept. "
                "Ligands that already satisfy the required number of "
                "positive responsive pairs are preserved."
            )
            print("-" * 45)
            print("Similarity matrix for compounds")
            print("-" * 60)
            print(
                f"{'No.':<5} {'Pair_ID':<10} {'Rec':<10} {'Mol':<10} "
                f"{'Num_Exp_Mol':<12} {'Mol_PC':<10}"
            )
            print("-" * 60)

        # ------------------------------------------------------------------
        # Build *mol_exp_list*: metadata for every positive pair in receptor
        # ------------------------------------------------------------------
        mol_exp_list = []
        for j, pair_id in enumerate(Pairs_id_r):
            Data_pair = M2OR_data_filt.loc[pair_id]
            mol = Data_pair["main_compounds_id"]
            mol_pos_cond = List_L.at[mol, "pos_cond_c"]
            Num_exp_Mol = np.size(List_L.at[mol, "pair_id_pos"])

            mol_exp_list.append(
                {
                    "mol_id": mol,
                    "Num_exp_Mol": Num_exp_Mol,
                    "mol_pos_cond": mol_pos_cond,
                    "pair_id": pair_id,
                }
            )

            if LOG:
                print(
                    f"{j:<5} {pair_id:<10} {Data_pair['main_receptors_id']:<10} "
                    f"{mol:<10} {Num_exp_Mol:<12} {round(mol_pos_cond, 2):<10}"
                )

        if LOG:
            print("-" * 60)

        mol_exp_list = pd.DataFrame(mol_exp_list)

        # ------------------------------------------------------------------
        # STEP 1 – Keep any ligand that is *under‑represented* (<=1 remaining)
        # ------------------------------------------------------------------
        mol_exp_list_to_keep_bg = mol_exp_list[mol_exp_list["mol_pos_cond"] <= 1]
        if LOG and len(mol_exp_list_to_keep_bg) > 1:
            print("-" * 45)
            print("Molecules with complete condition:", list(mol_exp_list_to_keep_bg["pair_id"]))
            print("-" * 45)

        # Remove these ligands from further pruning consideration
        mol_exp_list = mol_exp_list[mol_exp_list["mol_pos_cond"] > 1]

        # Remaining pairs still to *drop* for this receptor
        Pos_PA = int(Pos_PA - len(mol_exp_list_to_keep_bg))

        # ------------------------------------------------------------------
        # STEP 2 – Decide which additional pairs to remove based on how many
        #          surplus pairs are still present (Pos_PA <, =, or > 1)
        # ------------------------------------------------------------------
        if Pos_PA < 1:
            # --------------------------------------------------------------
            # Case A – All remaining pairs belong to over‑represented ligands.
            #          Remove them all.
            # --------------------------------------------------------------
            Pairs_to_remove = list(mol_exp_list["pair_id"])
            if LOG:
                print("-" * 45)
                print("Pairs_to_Remove:", Pairs_to_remove)
                print("-" * 45)
                print(f"{'No.':<5} {'Pair_id':<10} {'Rec':<5} {'Mol':<5} {'Remove':<10}")
                print("-" * 45)

            for ptr, pair_to_drop in enumerate(Pairs_to_remove):
                mol_d = M2OR_data_filt.loc[pair_to_drop]["main_compounds_id"]
                List_R = P1DFG.delete_pair_P(List_R, rec_id, pair_to_drop)
                List_L = P1DFG.delete_pair_P(List_L, mol_d, pair_to_drop)
                if LOG:
                    print(
                        f"{ptr:<5} {pair_to_drop:<10} {rec_id:<5} {mol_d:<5} {'Remove':<10}"
                    )

            # Recompute conditions after bulk deletion
            List_R, List_L = P1DFG.updated_pos_conditions(List_R, List_L)

        elif Pos_PA == 1:
            # --------------------------------------------------------------
            # Case B – Exactly one additional positive pair must be kept.
            #          Choose the ligand with the *fewest* experiments.
            # --------------------------------------------------------------
            mol_exp_list_sorted = mol_exp_list.sort_values(by="Num_exp_Mol", ascending=True)
            mol_exp_list_lowest = mol_exp_list_sorted.head(Pos_PA)
            least_mol_ids = list(mol_exp_list_lowest["mol_id"])

            # Determine which pairs belong to the selected ligand(s)
            Pairs_to_keep = list(
                dfu.search_experiments(mol_exp_list, "mol_id", least_mol_ids)["pair_id"]
            )
            Pairs_to_remove = [
                int(x) for x in list(mol_exp_list["pair_id"]) if int(x) not in Pairs_to_keep
            ]

            if LOG:
                print("-" * 45)
                print("Pair to Keep:", Pairs_to_keep)
                print("Pairs to Remove:", Pairs_to_remove)
                print("-" * 45)
                print(f"{'No.':<5} {'Pair_id':<10} {'Rec':<5} {'Mol':<5} {'Remove':<10}")
                print("-" * 45)

            for ptr, pair_to_drop in enumerate(Pairs_to_remove):
                mol_d = M2OR_data_filt.loc[pair_to_drop]["main_compounds_id"]
                List_R = P1DFG.delete_pair_P(List_R, rec_id, pair_to_drop)
                List_L = P1DFG.delete_pair_P(List_L, mol_d, pair_to_drop)
                if LOG:
                    print(
                        f"{ptr:<5} {pair_to_drop:<10} {rec_id:<5} {mol_d:<5} {'Remove':<10}"
                    )

            List_R, List_L = P1DFG.updated_pos_conditions(List_R, List_L)

        else:  # Pos_PA > 1
            # --------------------------------------------------------------
            # Case C – Need to keep *Pos_PA* additional pairs. Use diversity‑
            # driven selection (similar to original algorithm).
            # --------------------------------------------------------------
            if LOG:
                print(i)
                print("xxxxxxxxxxxxxxx")

            if len(mol_exp_list) == 1:
                # Only one ligand remains – drop all its pairs except one
                Pairs_to_remove = list(mol_exp_list["pair_id"])
                mol_d = mol_exp_list["mol_id"].iloc[0]

                for ptr, pair_to_drop in enumerate(Pairs_to_remove):
                    List_R = P1DFG.delete_pair_P(List_R, rec_id, pair_to_drop)
                    List_L = P1DFG.delete_pair_P(List_L, mol_d, pair_to_drop)
                    if LOG:
                        if ptr == 0:
                            print("-" * 45)
                            print(
                                f"{'No.':<5} {'Pair_id':<10} {'Rec':<5} "
                                f"{'Mol':<5} {'Remove':<10}"
                            )
                            print("-" * 45)
                        print(
                            f"{ptr:<5} {pair_to_drop:<10} {rec_id:<5} {mol_d:<5} {'Remove':<10}"
                        )

                List_R, List_L = P1DFG.updated_pos_conditions(List_R, List_L)

            else:
                # More than one ligand – perform diversity‑based selection
                if len(mol_exp_list) > Pos_PA:
                    # Strip to *Pos_PA* least‑represented ligands first
                    mol_exp_list_sorted = mol_exp_list.sort_values(
                        by="Num_exp_Mol", ascending=True
                    )
                    mol_exp_list_lowest = mol_exp_list_sorted.head(Pos_PA)
                    least_mol_ids = list(mol_exp_list_lowest["mol_id"])
                    List_Mol = list(mol_exp_list["mol_id"])
                else:
                    # Use every ligand when they are fewer than *Pos_PA*
                    least_mol_ids = list(mol_exp_list["mol_id"])
                    List_Mol = least_mol_ids
                    Pos_PA = len(mol_exp_list)

                # Similarity matrix restricted to these ligands
                Matrix_Sim_Compound_E = Compound_sim_matrix.loc[
                    List_Mol, [str(i) for i in List_Mol]
                ]

                # Diversity metrics
                least_mol_avg = P1DFG.calculate_avg_similarity(
                    Matrix_Sim_Compound_E, least_mol_ids
                )
                combo = P1DFG.select_diverse_subset(
                    Matrix_Sim_Compound_E, int(Pos_PA)
                )
                sim_combo = P1DFG.calculate_avg_similarity(
                    Matrix_Sim_Compound_E, np.array(combo)
                )
                _, sim_tot = P1DFG.find_most_different_k(
                    Matrix_Sim_Compound_E, len(Matrix_Sim_Compound_E)
                )

                if LOG:
                    print("Avg similarity (total):", sim_tot)
                    print("Most differentiated Mol:", combo, "Avg sim:", sim_combo)
                    print("Least represented Mol:", least_mol_ids, "Avg sim:", least_mol_avg)

                # # Decide whether to keep least‑represented or max‑diverse set
                # if least_mol_avg < sim_tot:
                #     Pairs_to_keep = list(
                #         dfu.search_experiments(mol_exp_list, "mol_id", least_mol_ids)[
                #             "pair_id"
                #         ]
                #     )
                # else:
                #     Pairs_to_keep = list(
                #         dfu.search_experiments(mol_exp_list, "mol_id", combo)["pair_id"]
                #     )
                    
                Pairs_to_keep = list(
                    dfu.search_experiments(mol_exp_list, "mol_id", combo)["pair_id"]
                )
                Pairs_to_remove = [
                    int(x)
                    for x in list(mol_exp_list["pair_id"])
                    if int(x) not in Pairs_to_keep
                ]

                if LOG:
                    print("-" * 45)
                    print("Pairs_to_Keep:", Pairs_to_keep)
                    print("Pairs_to_Remove:", Pairs_to_remove)
                    print("-" * 45)
                    print(
                        f"{'No.':<5} {'Pair_id':<10} {'Rec':<5} "
                        f"{'Mol':<5} {'Remove':<10}"
                    )
                    print("-" * 45)

                for ptr, pair_to_drop in enumerate(Pairs_to_remove):
                    mol_d = M2OR_data_filt.loc[pair_to_drop]["main_compounds_id"]
                    List_R = P1DFG.delete_pair_P(List_R, rec_id, pair_to_drop)
                    List_L = P1DFG.delete_pair_P(List_L, mol_d, pair_to_drop)
                    if LOG:
                        print(
                            f"{ptr:<5} {pair_to_drop:<10} {rec_id:<5} {mol_d:<5} {'Remove':<10}"
                        )

                List_R, List_L = P1DFG.updated_pos_conditions(List_R, List_L)
                List_R, List_L = P1DFG.update_imbalance(List_R, List_L)
    # ------------------------------------------------------------------
    # Return cleaned DataFrames
    # ------------------------------------------------------------------
    return List_R, List_L


def Neg_receptor_Soft(
    List_R,
    List_L,
    M2OR_compounds,
    M2OR_data_filt,
    LOG=False,
):
    """Ensure each receptor keeps only the required number of negative pairs.

    Same logic as Pos_receptor_Soft but for negative responses:
    * ``Exp_Av_N``
    * ``Negative``
    * ``neg_cond_c``
    * uses ``delete_pair_N`` to drop pairs.
    """
    # ------------------------------------------------------------------
    # Pre‑computation & initial bookkeeping
    # ------------------------------------------------------------------
    
    
    if LOG:
        print("-" * 45)
    Compound_sim_matrix = sc.Compounds_sim_matrix(M2OR_compounds)
    if LOG:
        print("-" * 45)
        
        

    List_R["pair_id_neg"] = List_R["Exp_Av_N"]
    List_L["pair_id_neg"] = List_L["Exp_Av_N"]
    List_R, List_L = P1DFG.updated_neg_conditions(List_R, List_L)

    # ------------------------------------------------------------------
    # Main loop – iterate over receptors
    # ------------------------------------------------------------------
    for i in range(len(List_R)):
        rec_id = List_R.index[i]
        if rec_id == 102:
            LOG = True
        else:
            LOG = False
        # Skip receptors that already fulfil their quota
        if List_R.at[rec_id, "neg_cond_c"] <= 1:
            continue

        Neg_PA = List_R.at[rec_id, "Negative"]         # required pair count
        Pairs_id_r = List_R.at[rec_id, "pair_id_neg"]  # current negative pairs

        if LOG:
            print(f"Receptor: {rec_id}")
            print(f"Negative pairs required: {Neg_PA}")
            print(f"Negative pairs available: {len(Pairs_id_r)}")
            print("-" * 45)
            print(
                f"Criteria: remove {len(Pairs_id_r) - Neg_PA} negative pairs "
                f"for receptor {rec_id}"
            )
            print(
                "Priority: maximise ligand variety. "
                "The most diverse ligands will be kept. "
                "Ligands that already satisfy the required number of "
                "negative responsive pairs are preserved."
            )
            print("-" * 45)
            print("Similarity matrix for compounds")
            print("-" * 60)
            print(
                f"{'No.':<5} {'Pair_ID':<10} {'Rec':<10} {'Mol':<10} "
                f"{'Num_Exp_Mol':<12} {'Mol_NC':<10}"
            )
            print("-" * 60)

        # ------------------------------------------------------------------
        # Build *mol_exp_list*: metadata for every negative pair in receptor
        # ------------------------------------------------------------------
        mol_exp_list = []
        for j, pair_id in enumerate(Pairs_id_r):
            Data_pair = M2OR_data_filt.loc[pair_id]
            mol = Data_pair["main_compounds_id"]
            mol_neg_cond = List_L.at[mol, "neg_cond_c"]
            Num_exp_Mol = np.size(List_L.at[mol, "pair_id_neg"])

            mol_exp_list.append({
                "mol_id": mol,
                "Num_exp_Mol": Num_exp_Mol,
                "mol_neg_cond": mol_neg_cond,
                "pair_id": pair_id,
            })

            if LOG:
                print(
                    f"{j:<5} {pair_id:<10} {Data_pair['main_receptors_id']:<10} "
                    f"{mol:<10} {Num_exp_Mol:<12} {round(mol_neg_cond,2):<10}"
                )

        if LOG:
            print("-" * 60)

        mol_exp_list = pd.DataFrame(mol_exp_list)

        # ------------------------------------------------------------------
        # STEP 1 – Keep any ligand that is *under‑represented* (<=1 remaining)
        # ------------------------------------------------------------------
        mol_exp_list_to_keep_bg = mol_exp_list[mol_exp_list["mol_neg_cond"] <= 1]
        if LOG and len(mol_exp_list_to_keep_bg) > 1:
            print("-" * 45)
            print("Molecules with complete condition:", list(mol_exp_list_to_keep_bg["pair_id"]))
            print("-" * 45)

        # Remove these ligands from further pruning consideration
        mol_exp_list = mol_exp_list[mol_exp_list["mol_neg_cond"] > 1]

        # Remaining pairs still to *drop* for this receptor
        Neg_PA = int(Neg_PA - len(mol_exp_list_to_keep_bg))

        # ------------------------------------------------------------------
        # STEP 2 – Decide which additional pairs to remove based on surplus
        # ------------------------------------------------------------------
        if Neg_PA < 1:
            # Case A – remove all remaining over‑represented ligands
            Pairs_to_remove = list(mol_exp_list["pair_id"])
            if LOG:
                print("-" * 45)
                print("Pairs_to_Remove:", Pairs_to_remove)
                print("-" * 45)
                print(f"{'No.':<5} {'Pair_id':<10} {'Rec':<5} {'Mol':<5} {'Remove':<10}")
                print("-" * 45)
            for ptr, pid in enumerate(Pairs_to_remove):
                mol_d = M2OR_data_filt.loc[pid]["main_compounds_id"]
                List_R = P1DFG.delete_pair_N(List_R, rec_id, pid)
                List_L = P1DFG.delete_pair_N(List_L, mol_d, pid)
                if LOG:
                    print(
                        f"{ptr:<5} {pid:<10} {rec_id:<5} {mol_d:<5} {'Remove':<10}"
                    )
            List_R, List_L = P1DFG.updated_neg_conditions(List_R, List_L)

        elif Neg_PA == 1:
            # Case B – keep single ligand with fewest experiments
            mol_exp_list_sorted = mol_exp_list.sort_values(by="Num_exp_Mol", ascending=True)
            least_mol_ids = list(mol_exp_list_sorted.head(1)["mol_id"])
            Pairs_to_keep = list(
                dfu.search_experiments(mol_exp_list, "mol_id", least_mol_ids)["pair_id"]
            )
            Pairs_to_remove = [pid for pid in mol_exp_list["pair_id"] if pid not in Pairs_to_keep]

            if LOG:
                print("-" * 45)
                print("Pair to Keep:", Pairs_to_keep)
                print("Pairs to Remove:", Pairs_to_remove)
                print("-" * 45)
                print(f"{'No.':<5} {'Pair_id':<10} {'Rec':<5} {'Mol':<5} {'Remove':<10}")
                print("-" * 45)

            for ptr, pid in enumerate(Pairs_to_remove):
                mol_d = M2OR_data_filt.loc[pid]["main_compounds_id"]
                List_R = P1DFG.delete_pair_N(List_R, rec_id, pid)
                List_L = P1DFG.delete_pair_N(List_L, mol_d, pid)
                if LOG:
                    print(
                        f"{ptr:<5} {pid:<10} {rec_id:<5} {mol_d:<5} {'Remove':<10}"
                    )
            List_R, List_L = P1DFG.updated_neg_conditions(List_R, List_L)

        else:
            # Case C – diversity‑driven selection
            if len(mol_exp_list) == 1:
                Pairs_to_remove = list(mol_exp_list["pair_id"])
                mol_d = mol_exp_list["mol_id"].iloc[0]
                for ptr, pid in enumerate(Pairs_to_remove):
                    List_R = P1DFG.delete_pair_N(List_R, rec_id, pid)
                    List_L = P1DFG.delete_pair_N(List_L, mol_d, pid)
                    if LOG:
                        if ptr == 0:
                            print("-" * 45)
                            print(f"{'No.':<5} {'Pair_id':<10} {'Rec':<5} {'Mol':<5} {'Remove':<10}")
                            print("-" * 45)
                        print(
                            f"{ptr:<5} {pid:<10} {rec_id:<5} {mol_d:<5} {'Remove':<10}"
                        )
                List_R, List_L = P1DFG.updated_neg_conditions(List_R, List_L)
            else:
                if len(mol_exp_list) > Neg_PA:
                    least_mol_ids = list(
                        mol_exp_list.sort_values(by="Num_exp_Mol", ascending=True)
                        .head(Neg_PA)["mol_id"]
                    )
                else:
                    least_mol_ids = list(mol_exp_list["mol_id"])
                    Neg_PA = len(mol_exp_list)
                List_Mol = list(mol_exp_list["mol_id"])
                Matrix_Sim_Compound_E = Compound_sim_matrix.loc[
                    List_Mol, [str(x) for x in List_Mol]
                ]
                # if LOG:
                    # plot_mds(Matrix_Sim_Compound_E)
                
  
                combo = P1DFG.select_diverse_subset(
                    Matrix_Sim_Compound_E, Neg_PA
                )
                Matrix_Sim_Compound_combo = Compound_sim_matrix.loc[
                    combo, [str(x) for x in combo ]
                ]
                Matrix_Sim_Compound_least = Compound_sim_matrix.loc[
                    least_mol_ids, [str(x) for x in least_mol_ids ]
                ]
                
                # _, sim_tot = P1DFG.find_most_different_k(
                #     Matrix_Sim_Compound_E, len(Matrix_Sim_Compound_E)
                # )
                
                
                sim_tot = avg_distance_to_centroid(Matrix_Sim_Compound_E)
                sim_comb = avg_distance_to_centroid(Matrix_Sim_Compound_combo)
                sim_least = avg_distance_to_centroid(Matrix_Sim_Compound_least)
                if LOG:
                    print("Avg similarity (total):", sim_tot)
                    print("Most differentiated Mol:", sim_comb)
                    print("Least represented Mol:", sim_least)

                # if sim_least  < sim_tot:
                #     Pairs_to_keep = list(
                #         dfu.search_experiments(mol_exp_list, "mol_id", least_mol_ids)["pair_id"]
                #     )
                # else:
                Pairs_to_keep = list(
                    dfu.search_experiments(mol_exp_list, "mol_id", combo)["pair_id"]
                )
                    
                    
                Pairs_to_remove = [pid for pid in mol_exp_list["pair_id"] if pid not in Pairs_to_keep]

                if LOG:
                    print("-" * 45)
                    print("Pairs_to_Keep:", Pairs_to_keep)
                    print("Pairs_to_Remove:", Pairs_to_remove)
                    print("-" * 45)
                    print(f"{'No.':<5} {'Pair_id':<10} {'Rec':<5} {'Mol':<5} {'Remove':<10}")
                    print("-" * 45)

                for ptr, pid in enumerate(Pairs_to_remove):
                    mol_d = M2OR_data_filt.loc[pid]["main_compounds_id"]
                    List_R = P1DFG.delete_pair_N(List_R, rec_id, pid)
                    List_L = P1DFG.delete_pair_N(List_L, mol_d, pid)
                    if LOG:
                        print(
                            f"{ptr:<5} {pid:<10} {rec_id:<5} {mol_d:<5} {'Remove':<10}"
                        )
                List_R, List_L = P1DFG.updated_neg_conditions(List_R, List_L)
                List_R, List_L = P1DFG.update_imbalance(List_R, List_L)

    # ------------------------------------------------------------------
    # Return cleaned DataFrames
    # ------------------------------------------------------------------
    return List_R, List_L


def Pos_Hard_Method(
        List_R, List_L, M2OR_data_filt, LOG = False):
    cont = 0
    for i in range (len(List_R)):
        
        Receptor_id = List_R.index[i]
        Pos_cond_R =  List_R.at[Receptor_id,'pos_cond_c']     
        Pairs_R = List_R.at[Receptor_id,'pair_id_pos']
        Pairs_A = List_R.at[Receptor_id,'Positive']
        
        if Pos_cond_R > 1:
            
            cont += 1
            if LOG:print('-' * 45)
            if LOG:print(f"{'No.':<5} {'Rec_id':<10} {'Pos_Cond':<10} {'No_Pairs':<10} {'Req':<10}")
            if LOG:print(f"{str(cont):<5} {str(Receptor_id):<10} {str(round(Pos_cond_R,2)):<10} {str(len(Pairs_R)):<10} {str(Pairs_A):<10}")
            if LOG:print('-' * 45)
           
            if LOG:print('-' * 45)
            if LOG:print(f"{'No.':<5} {'Pair_id':<10} {'Rec':<5} {'Mol':<5} {'NoE_Mol':<10} {'Mol_NC':<10}")
            if LOG:print('-' * 45) 
             
            mol_exp_list = []
            
            for j in range(len(Pairs_R)):
                Data_pair = M2OR_data_filt.loc[Pairs_R[j]]
                mol = Data_pair['main_compounds_id'] 
                mol_pos_cond =  List_L.at[mol,'pos_cond_c']       
                Num_exp_Mol = np.size(List_L.at[mol,'pair_id_pos'])     
                mol_exp_list.append({
                    'mol_id': mol,
                    'Num_exp_Mol': Num_exp_Mol,
                    'mol_pos_cond': mol_pos_cond,
                    'pair_id': Pairs_R[j]
                }) 
                if LOG:print(f"{str(j):<5} {str(Pairs_R[j]):<10} {str(Data_pair['main_receptors_id']):<5} {str(mol):<5} {str(Num_exp_Mol):<10} {str(round(mol_pos_cond,2)):<10}")
            mol_exp_list = pd.DataFrame(mol_exp_list)
            if LOG:print('-' * 45)
            
            mol_exp_list = mol_exp_list.sort_values(by='Num_exp_Mol', ascending=True)
            mol_exp_list_lowest = mol_exp_list.head(int(Pairs_A))                
            least_mol_ids = list(mol_exp_list_lowest['mol_id'])                 
            Pairs_to_keep = list(dfu.search_experiments(mol_exp_list, 'mol_id', least_mol_ids)['pair_id'])
            Pairs_to_remove = [int(x) for x in list(mol_exp_list['pair_id']) if int(x) not in Pairs_to_keep]
        
            if LOG:print('-' * 45)      
            if LOG:print("Pairs_to_Keep:",list(Pairs_to_keep))   
            if LOG:print("Pairs_to_Remove:", Pairs_to_remove)
               
            if LOG:print('-' * 45)
            if LOG:print(f"{'No.':<5} {'Pair_id':<10} {'Rec':<5} {'Mol':<5} {'Remove':<10}")
            if LOG:print('-' * 45) 
            
            for ptr in range(len(Pairs_to_remove)): 
                mol_d = M2OR_data_filt.loc[Pairs_to_remove[ptr]]['main_compounds_id']      
                List_R = P1DFG.delete_pair_P(List_R, Receptor_id, Pairs_to_remove[ptr])
                List_L = P1DFG.delete_pair_P(List_L, mol_d, Pairs_to_remove[ptr])
                List_L.at[mol_d,'Positive'] =  List_L.at[mol_d,'Positive']-1
                if LOG:print(f"{str(ptr):<5} {str(Pairs_to_remove[ptr]):<10} {str(Receptor_id):<5} {str(mol_d):<5} {'Remove':<10}")
            
            List_R, List_L = P1DFG.updated_pos_conditions(List_R, List_L) 
            List_R, List_L = P1DFG.update_imbalance(List_R, List_L)
            
    return List_R, List_L  

def Neg_Hard_Method(
        List_R, List_L, M2OR_data_filt, LOG = False):
    cont = 0
    for i in range (len(List_R)):
        
        Receptor_id = List_R.index[i]
        Neg_cond_R =  List_R.at[Receptor_id,'neg_cond_c']     
        Pairs_R = List_R.at[Receptor_id,'pair_id_neg']
        Pairs_A = List_R.at[Receptor_id,'Negative']
        
        if Neg_cond_R > 1:
            
            cont += 1
            if LOG:print('-' * 45)
            if LOG:print(f"{'No.':<5} {'Rec_id':<10} {'Neg_Cond':<10} {'No_Pairs':<10} {'Req':<10}")
            if LOG:print(f"{str(cont):<5} {str(Receptor_id):<10} {str(round(Neg_cond_R,2)):<10} {str(len(Pairs_R)):<10} {str(Pairs_A):<10}")
            if LOG:print('-' * 45)
           
            if LOG:print('-' * 45)
            if LOG:print(f"{'No.':<5} {'Pair_id':<10} {'Rec':<5} {'Mol':<5} {'NoE_Mol':<10} {'Mol_NC':<10}")
            if LOG:print('-' * 45) 
             
            mol_exp_list = []
            
            for j in range(len(Pairs_R)):
                Data_pair = M2OR_data_filt.loc[Pairs_R[j]]
                mol = Data_pair['main_compounds_id'] 
                mol_neg_cond =  List_L.at[mol,'neg_cond_c']       
                Num_exp_Mol = np.size(List_L.at[mol,'pair_id_neg'])     
                # Append mol, Num_exp_Mol, and mol_neg_cond as a tuple
                # Append as a dictionary
                mol_exp_list.append({
                    'mol_id': mol,
                    'Num_exp_Mol': Num_exp_Mol,
                    'mol_neg_cond': mol_neg_cond,
                    'pair_id': Pairs_R[j]
                }) 
                if LOG:print(f"{str(j):<5} {str(Pairs_R[j]):<10} {str(Data_pair['main_receptors_id']):<5} {str(mol):<5} {str(Num_exp_Mol):<10} {str(round(mol_neg_cond,2)):<10}")
            mol_exp_list = pd.DataFrame(mol_exp_list)
            if LOG:print('-' * 45)
            
            mol_exp_list = mol_exp_list.sort_values(by='Num_exp_Mol', ascending=True)
            mol_exp_list_lowest = mol_exp_list.head(int(Pairs_A))                
            least_mol_ids = list(mol_exp_list_lowest['mol_id'])                 
            Pairs_to_keep = list(dfu.search_experiments(mol_exp_list, 'mol_id', least_mol_ids)['pair_id'])
            Pairs_to_remove = [int(x) for x in list(mol_exp_list['pair_id']) if int(x) not in Pairs_to_keep]
        
            if LOG:print('-' * 45)      
            if LOG:print("Pairs_to_Keep:",list(Pairs_to_keep))   
            if LOG:print("Pairs_to_Remove:", Pairs_to_remove)
               
            if LOG:print('-' * 45)
            if LOG:print(f"{'No.':<5} {'Pair_id':<10} {'Rec':<5} {'Mol':<5} {'Remove':<10}")
            if LOG:print('-' * 45) 
            
            for ptr in range(len(Pairs_to_remove)): 
                mol_d = M2OR_data_filt.loc[Pairs_to_remove[ptr]]['main_compounds_id']      
                List_R = P1DFG.delete_pair_N(List_R, Receptor_id, Pairs_to_remove[ptr])
                List_L = P1DFG.delete_pair_N(List_L, mol_d, Pairs_to_remove[ptr])
                List_L.at[mol_d,'Negative'] =  List_L.at[mol_d,'Negative']-1
                if LOG:print(f"{str(ptr):<5} {str(Pairs_to_remove[ptr]):<10} {str(Receptor_id):<5} {str(mol_d):<5} {'Remove':<10}")
            
            List_R, List_L = P1DFG.updated_neg_conditions(List_R, List_L)  
            List_R, List_L = P1DFG.updated_pos_conditions(List_R, List_L)  
            List_R, List_L = P1DFG.update_imbalance(List_R, List_L)
    return List_R, List_L  

def Pos_Ligant_Hard_Method(
        List_R, List_L, M2OR_data_filt, LOG = False):
    
    cont = 0 
    for i in range(len(List_L)):
             
        Ligant_id = List_L.index[i]
        Pos_cond_L =  List_L.at[Ligant_id,'pos_cond_c']     
        Pairs_L = List_L.at[Ligant_id,'pair_id_pos']
        Pairs_A = List_L.at[Ligant_id,'Positive']
        
        if Pos_cond_L > 1:
            
            cont += 1
            if LOG:print('-' * 45)
            if LOG:print(f"{'No.':<5} {'Rec_id':<10} {'Pos_Cond':<10} {'No_Pairs':<10} {'Req':<10}")
            if LOG:print(f"{str(cont):<5} {str(Ligant_id):<10} {str(round(Pos_cond_L,2)):<10} {str(len(Pairs_L)):<10} {str(Pairs_A):<10}")
            if LOG:print('-' * 45)
           
            if LOG:print('-' * 45)
            if LOG:print(f"{'No.':<5} {'Pair_id':<10} {'Lig':<5} {'Rec':<5} {'NoE_Rec':<10} {'Rec_NC':<10}")
            if LOG:print('-' * 45) 
             
            Receptor_exp_list = []
            
            for j in range(len(Pairs_L)):
                Data_pair = M2OR_data_filt.loc[Pairs_L[j]]
                Receptor = Data_pair['main_receptors_id'] 
                Receptor_pos_cond =  List_R.at[Receptor,'pos_cond_c']       
                Num_exp_Receptor = np.size(List_R.at[Receptor,'pair_id_pos'])     
                Receptor_exp_list.append({
                    'Receptor_id': Receptor,
                    'Num_exp_Rec': Num_exp_Receptor,
                    'Rec_pos_cond': Receptor_pos_cond,
                    'pair_id': Pairs_L[j]
                }) 
                if LOG:print(f"{str(j):<5} {str(Pairs_L[j]):<10} {str(Data_pair['main_compounds_id']):<5} {str(Receptor):<5} {str(Num_exp_Receptor):<10} {str(round(Receptor_pos_cond,2)):<10}")
            Receptor_exp_list = pd.DataFrame(Receptor_exp_list)
            if LOG:print('-' * 45)
            
            Receptor_exp_list  = Receptor_exp_list.sort_values(by='Num_exp_Rec', ascending=True)
            Receptor_exp_list_lowest = Receptor_exp_list.head(int(Pairs_A))                
            least_Receptors_ids = list(Receptor_exp_list_lowest['Receptor_id'])                 
            Pairs_to_keep = list(dfu.search_experiments(Receptor_exp_list, 'Receptor_id', least_Receptors_ids)['pair_id'])
            Pairs_to_remove = [int(x) for x in list(Receptor_exp_list['pair_id']) if int(x) not in Pairs_to_keep]
        
            if LOG:print('-' * 45)      
            if LOG:print("Pairs_to_Keep:",list(Pairs_to_keep))   
            if LOG:print("Pairs_to_Remove:", Pairs_to_remove)
               
            if LOG:print('-' * 45)
            if LOG:print(f"{'No.':<5} {'Pair_id':<10} {'Ligant':<5} {'Rec':<5} {'Remove':<10}")
            if LOG:print('-' * 45) 
            
            for ptr in range(len(Pairs_to_remove)): 
                Rec_d = M2OR_data_filt.loc[Pairs_to_remove[ptr]]['main_receptors_id']      
                List_L = P1DFG.delete_pair_P(List_L, Ligant_id, Pairs_to_remove[ptr])
                List_R = P1DFG.delete_pair_P(List_R, Rec_d, Pairs_to_remove[ptr])
                List_R.at[Rec_d,'Positive'] =  List_R.at[Rec_d,'Positive']-1
                if LOG:print(f"{str(ptr):<5} {str(Pairs_to_remove[ptr]):<10} {str(Ligant_id):<5} {str(Rec_d):<5} {'Remove':<10}")
            
            List_R, List_L = P1DFG.updated_pos_conditions(List_R, List_L) 
            List_R, List_L = P1DFG.update_imbalance(List_R, List_L)
            
    return List_R, List_L
        
def Neg_Ligant_Hard_Method(
    List_R,
    List_L,
    M2OR_data_filt,
    LOG=False,
):
    """Ensure each receptor keeps only the required number of negative pairs.

    Same logic as Pos_Ligant_Hard_Method but for negatives:
      * ``Exp_Av_N``
      * ``Negative``
      * ``neg_cond_c``
      * uses ``delete_pair_N`` to drop pairs
      * updates via ``updated_neg_conditions``
    """
    cont = 0
    for i in range(len(List_L)):
        Ligant_id = List_L.index[i]
        Neg_cond_L = List_L.at[Ligant_id, 'neg_cond_c']
        Pairs_L = List_L.at[Ligant_id, 'pair_id_neg']
        Pairs_A = List_L.at[Ligant_id, 'Negative']

        if Neg_cond_L > 1:
            cont += 1
            if LOG: print('-' * 45)
            if LOG: print(f"{'No.':<5} {'Rec_id':<10} {'Neg_Cond':<10} {'No_Pairs':<10} {'Req':<10}")
            if LOG: print(f"{str(cont):<5} {str(Ligant_id):<10} {str(round(Neg_cond_L,2)):<10} {str(len(Pairs_L)):<10} {str(Pairs_A):<10}")
            if LOG: print('-' * 45)
            if LOG: print('-' * 45)
            if LOG: print(f"{'No.':<5} {'Pair_id':<10} {'Lig':<5} {'Rec':<5} {'NoE_Rec':<10} {'Rec_NC':<10}")
            if LOG: print('-' * 45)

            Receptor_exp_list = []
            for j in range(len(Pairs_L)):
                Data_pair = M2OR_data_filt.loc[Pairs_L[j]]
                Receptor = Data_pair['main_receptors_id']
                Receptor_neg_cond = List_R.at[Receptor, 'neg_cond_c']
                Num_exp_Receptor = np.size(List_R.at[Receptor, 'pair_id_neg'])
                Receptor_exp_list.append({
                    'Receptor_id': Receptor,
                    'Num_exp_Rec': Num_exp_Receptor,
                    'Rec_neg_cond': Receptor_neg_cond,
                    'pair_id': Pairs_L[j]
                })
                if LOG: print(f"{str(j):<5} {str(Pairs_L[j]):<10} {str(Data_pair['main_compounds_id']):<5} {str(Receptor):<5} {str(Num_exp_Receptor):<10} {str(round(Receptor_neg_cond,2)):<10}")

            Receptor_exp_list = pd.DataFrame(Receptor_exp_list)
            if LOG: print('-' * 45)

            # choose the receptors with fewest experiments
            Receptor_exp_list = Receptor_exp_list.sort_values(by='Num_exp_Rec', ascending=True)
            Receptor_exp_list_lowest = Receptor_exp_list.head(int(Pairs_A))
            least_Receptors_ids = list(Receptor_exp_list_lowest['Receptor_id'])
            Pairs_to_keep = list(dfu.search_experiments(Receptor_exp_list, 'Receptor_id', least_Receptors_ids)['pair_id'])
            Pairs_to_remove = [int(x) for x in Receptor_exp_list['pair_id'] if int(x) not in Pairs_to_keep]

            if LOG: print('-' * 45)
            if LOG: print("Pairs_to_Keep:", Pairs_to_keep)
            if LOG: print("Pairs_to_Remove:", Pairs_to_remove)
            if LOG: print('-' * 45)
            if LOG: print(f"{'No.':<5} {'Pair_id':<10} {'Ligant':<5} {'Rec':<5} {'Remove':<10}")
            if LOG: print('-' * 45)

            for ptr, pid in enumerate(Pairs_to_remove):
                Rec_d = M2OR_data_filt.loc[pid]['main_receptors_id']
                List_L = P1DFG.delete_pair_N(List_L, Ligant_id, pid)
                List_R = P1DFG.delete_pair_N(List_R, Rec_d, pid)
                List_R.at[Rec_d, 'Negative'] = List_R.at[Rec_d, 'Negative'] - 1
                if LOG: print(f"{str(ptr):<5} {str(pid):<10} {str(Ligant_id):<5} {str(Rec_d):<5} {'Remove':<10}")

            List_R, List_L = P1DFG.updated_neg_conditions(List_R, List_L)
            List_R, List_L = P1DFG.update_imbalance(List_R, List_L)

    return List_R, List_L

def Fix_balance(
        List_R, List_L,Imb_lvl,M2OR_data_filt,LOG = False):
    
    cont = 0 
    if LOG:print('-' * 45)
    if LOG:print(f"{'No.':<5} {'Rec_id':<10} {'Imb':<10} {'Pos':<10} {'Neg':<10}")
    if LOG:print('-' * 45)
    for i in range(len(List_R)):
        Receptor_id = List_R.index[i]
        Imb_rec = List_R.at[Receptor_id,'Imbalance']
        Positive_p = List_R.at[Receptor_id,'Positive']
        Negative_p = List_R.at[Receptor_id,'Negative']
        if Imb_rec > Imb_lvl:
            cont += 1     
            if Positive_p > Negative_p:
                List_R.at[Receptor_id,'Positive'] = Imb_lvl * Negative_p
            if Negative_p > Positive_p:
                List_R.at[Receptor_id,'Negative'] = Imb_lvl * Positive_p
            if LOG:print(f"{str(cont):<5} {str(Receptor_id):<10} {str(round(Imb_rec,2)):<10} {str(round(Positive_p,2)):<10} {str(round(Negative_p,2)):<10} ")
        List_R.at[Receptor_id,'Number_exp'] = List_R.at[Receptor_id,'Positive'] + List_R.at[Receptor_id,'Negative']
    if LOG:print('-' * 45)        
    P1DFG.update_imbalance(List_R, List_L)
    P1DFG.updated_neg_conditions(List_R, List_L)
    P1DFG.updated_pos_conditions(List_R, List_L)

    List_R, List_L= Pos_Hard_Method(List_R, List_L, M2OR_data_filt)
    List_R, List_L= Neg_Hard_Method(List_R, List_L, M2OR_data_filt)
    
    for j in range(len(List_L)):
        compound_id = List_L.index[j]
        List_L.at[compound_id,'Number_exp'] = List_L.at[compound_id,'Positive'] + List_L.at[compound_id,'Negative']
    
    
    return List_R, List_L

def plot_mds(similarity_matrix, n_components=2, random_state=0):
    """
    Perform Multidimensional Scaling (MDS) on a similarity matrix and plot
    the resulting 2D embedding.

    Parameters
    ----------
    similarity_matrix : pandas.DataFrame
        Square matrix (n x n) where entry (i, j) is similarity of item i to item j.
    n_components : int, optional
        Number of dimensions for the MDS embedding (default: 2).
    random_state : int, optional
        Random seed for reproducibility (default: 0).

    Returns
    -------
    embedding : pandas.DataFrame
        DataFrame of shape (n_items, n_components) with MDS coordinates.
    """

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS

    # Convert similarity to dissimilarity (distance)
    # Ensure values in [0, 1], then distance = 1 - similarity
    dissimilarity = 1.0 - similarity_matrix.values

    # Initialize MDS with precomputed dissimilarities
    mds = MDS(n_components=n_components,
              dissimilarity='precomputed',
              random_state=random_state)

    # Fit-transform to get embedding coordinates
    coords = mds.fit_transform(dissimilarity)

    # Wrap into DataFrame for labeling
    embedding = pd.DataFrame(coords,
                             index=similarity_matrix.index,
                             columns=[f"Dim{i+1}" for i in range(n_components)])

    # Plot scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(embedding.iloc[:, 0], embedding.iloc[:, 1])
    for label, x, y in zip(embedding.index, embedding.iloc[:, 0], embedding.iloc[:, 1]):
        plt.text(x, y, str(label), fontsize=9, ha='center', va='center')

    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.title('MDS projection of similarity matrix')
    plt.tight_layout()
    plt.show()

    return embedding


def avg_distance_to_centroid(similarity_matrix):
    """
    Compute the average Euclidean distance of each item (row) in a similarity
    matrix to the centroid of all items.

    Parameters
    ----------
    similarity_matrix : pandas.DataFrame
        Square matrix (n × n) where each entry is a similarity score.

    Returns
    -------
    float
        Mean Euclidean distance of all rows to their centroid.
    """
    # Extract raw array (n_items × n_items)
    coords = similarity_matrix.values

    # Compute centroid (mean vector across rows)
    centroid = coords.mean(axis=0)

    # Euclidean distances from each row to centroid
    distances = np.linalg.norm(coords - centroid, axis=1)

    # Return the average of those distances
    return distances.mean()

