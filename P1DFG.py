import dataframe_utils as dfu
import numpy as np


def filter_inf_imb(M2OR_data, Imb_lvl, LOG = False):
    
    Past = 0
    Present = 0

    if LOG:print('-'*45)
    if LOG:print(f"{'Past':<5}  {'Present':<5}")
    if LOG:print('-'*45)
    while True:
        
        Past = len(M2OR_data)
        
        distribution_by_receptors = dfu.get_experiments_distribution(M2OR_data, dfu.Receptor_id)
        distribution_by_compound = dfu.get_experiments_distribution(M2OR_data, dfu.Compound_id)
        
        receptors_to_filtered = list((distribution_by_receptors[(distribution_by_receptors['Positive'] == 0) | (distribution_by_receptors['Negative'] == 0)]).index)
        ligands_to_filtered = list((distribution_by_compound[(distribution_by_compound['Positive'] == 0) | (distribution_by_compound['Negative'] == 0)]).index)
        
        _, M2OR_data_no_inf_receptors = dfu.search_experiments(M2OR_data, 'main_receptors_id', receptors_to_filtered, return_complement = True)
        _, M2OR_data_filt = dfu.search_experiments(M2OR_data_no_inf_receptors, 'main_compounds_id', ligands_to_filtered, return_complement = True)
        
        M2OR_data = M2OR_data_filt
        
        Present = len(M2OR_data)
        if LOG:print(f'{str(Past):<5}  {str(Present):<5}')
        if Past == Present:
            break
    if LOG:print('-'*45)
    
    return M2OR_data



def Get_All_d_Exp(list_size,Parameter,Db):
# Find the pairs in the Db, and put the pair_id in Exp_av_P 'Experiments avaliables positives' if binding
# and inf Exp_av_N if not.  
    

    for k in range(len(list_size)):
        D = list_size.index[k]
        Exp_D = dfu.search_experiments(Db, Parameter, int(D))
        list_size.at[D, 'Exp_Av_P'] = list((Exp_D[Exp_D['responsive'] == 1]).index)
        list_size.at[D,'Exp_Av_N'] =  list((Exp_D[Exp_D['responsive'] == 0]).index)
        
    return list_size
# def Get_list_exp():
    
def compare_exp_av_p(df1, df2, label1="DF1", label2="DF2"):
    # Count number of elements per row
    count_df1 = df1['Exp_Av_P'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
    count_df2 = df2['Exp_Av_P'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)

    # Total count
    total_df1 = count_df1.sum()
    total_df2 = count_df2.sum()

    print(f"Total elements in '{label1}': {total_df1}")
    print(f"Total elements in '{label2}': {total_df2}")

    # Flatten and convert to sets for comparison
    set_df1 = set([item for sublist in df1['Exp_Av_P'] if isinstance(sublist, (list, np.ndarray)) for item in sublist])
    set_df2 = set([item for sublist in df2['Exp_Av_P'] if isinstance(sublist, (list, np.ndarray)) for item in sublist])

    # Differences
    only_in_df1 = set_df1 - set_df2
    only_in_df2 = set_df2 - set_df1

    print(f"\nElements only in '{label1}': {sorted(only_in_df1)}")
    print(f"Elements only in '{label2}': {sorted(only_in_df2)}")    

def updated_pos_conditions(List_A, List_B, log = False):
    index_A = List_A.index
    for i in range (len(List_A)):
        R = index_A[i]
        List_A.at[R,'pos_cond_c'] = (np.size(List_A.at[R,'pair_id_pos']))/List_A.at[R,'Positive']
        
    index_B = List_B.index
    for i in range (len(List_B)):
        R = index_B[i]
        List_B.at[R,'pos_cond_c'] = (np.size(List_B.at[R,'pair_id_pos']))/List_B.at[R,'Positive']
    
    if log: print('Conditions_Updated')
    return(List_A, List_B)   

def update_imbalance(List_A, List_B, log = False):
    index_A = List_A.index
    for i in range (len(List_A)):
        R = index_A[i]
        if (List_A.at[R,'Negative']) >= (List_A.at[R,'Positive']):
            List_A.at[R,'Imbalance'] = (List_A.at[R,'Negative'])/(List_A.at[R,'Positive'])
        else:
            List_A.at[R,'Imbalance'] = (List_A.at[R,'Positive'])/(List_A.at[R,'Negative'])
        
    index_B = List_B.index
    for i in range (len(List_B)):
        R = index_B[i]
        List_B.at[R,'neg_cond_c'] = (np.size(List_B.at[R,'pair_id_neg']))/List_B.at[R,'Negative']
    
    if log: print('Conditions_Updated')
    return(List_A, List_B) 

def updated_neg_conditions(List_A, List_B, log = False):
    index_A = List_A.index
    for i in range (len(List_A)):
        R = index_A[i]
        List_A.at[R,'neg_cond_c'] = (np.size(List_A.at[R,'pair_id_neg']))/List_A.at[R,'Negative']
        
    index_B = List_B.index
    for i in range (len(List_B)):
        R = index_B[i]
        List_B.at[R,'neg_cond_c'] = (np.size(List_B.at[R,'pair_id_neg']))/List_B.at[R,'Negative']
    
    if log: print('Conditions_Updated')
    return(List_A, List_B) 

def delete_pair_P(df, idx, pair_to_remove):
    # 1) Pull out whatever is stored at that cell
    cell = df.at[idx, 'pair_id_pos']
    #print("Before:", cell, "→ type:", type(cell))

    # 2) If it’s not already a list, coerce it into one
    if not isinstance(cell, list):
        # If it’s a numpy array, use .tolist(); if it's a tuple, use list()
        try:
            cell = cell.tolist()
        except AttributeError:
            cell = list(cell)

    # 3) Now you can safely remove the unwanted pair
    filtered = [x for x in cell if x != pair_to_remove]

    # 4) Write it back as a list
    df.at[idx, 'pair_id_pos'] = filtered
    #rint("After: ", filtered, "→ type:", type(df.at[idx, 'pair_id_pos']))

    # 5) And now len() will work:
    #print("Length is now:", len(df.at[idx, 'pair_id_pos']))

    return df
    
def delete_pair_N(df, idx, pair_to_remove):
    # 1) Pull out whatever is stored at that cell
    cell = df.at[idx, 'pair_id_neg']
    #print("Before:", cell, "→ type:", type(cell))
    
    # 2) Coerce to list if needed
    if not isinstance(cell, list):
        try:
            cell = cell.tolist()    # for numpy arrays
        except AttributeError:
            cell = list(cell)       # for tuples or other iterables
    
    # 3) Filter out the unwanted pair
    filtered = [x for x in cell if x != pair_to_remove]
    
    # 4) Write back a pure Python list
    df.at[idx, 'pair_id_neg'] = filtered
    #print("After: ", filtered, "→ type:", type(df.at[idx, 'pair_id_neg']))
    
    # 5) Now len() will work
    #print("Length is now:", len(df.at[idx, 'pair_id_neg']))
    
    return df


import itertools
def calculate_avg_similarity(filtered_matrix, mol_list):

    
    if len(mol_list) < 2:
        return None  # or return 0.0, depending on your use case

    sims = []
    for a, b in itertools.combinations(mol_list, 2):
        sim = filtered_matrix.loc[a, str(b)]
        sims.append(sim)

    avg_sim = np.mean(sims)
    return avg_sim


def select_diverse_subset(sim_matrix, k, mds_components=10, max_tries=10, random_state=42):
    import numpy as np
    from sklearn.manifold import MDS
    from sklearn.cluster import KMeans

    n = len(sim_matrix)
    if k > n:
        raise ValueError(f"k={k} is larger than the number of items n={n}.")

    # Convert similarity -> distance (assumes sim in [0,1] and symmetric with zeros diag)
    distance_matrix = 1 - sim_matrix.values  # if sim_matrix is a DataFrame
    IDX = sim_matrix.index

    # MDS: set n_init explicitly to avoid the FutureWarning
    embedding = MDS(
        n_components=mds_components,
        dissimilarity='precomputed',
        random_state=random_state,
        n_init=1  # or 4 to preserve older behavior
    )
    features = embedding.fit_transform(distance_matrix)

    # Try KMeans until all clusters are represented
    labels = None
    kmeans = None
    for t in range(max_tries):
        kmeans = KMeans(
            n_clusters=k,
            n_init='auto',      # or set an int like 10 for reproducibility
            random_state=random_state + t
        )
        labels = kmeans.fit_predict(features)
        if np.unique(labels).size == k:
            break

    # If somehow still missing clusters, fall back by picking from the present clusters,
    # then fill the remainder by farthest-from-selected heuristic.
    selected = []
    used = set()

    # Select one representative per *present* cluster (closest to its center)
    present_clusters = np.unique(labels)
    for cluster_id in present_clusters:
        cluster_indices = np.where(labels == cluster_id)[0]
        if cluster_indices.size == 0:
            continue  # shouldn't happen after the loop, but be defensive
        center = kmeans.cluster_centers_[cluster_id]
        # pick the point closest to the center within that cluster
        dists = np.linalg.norm(features[cluster_indices] - center, axis=1)
        closest_local = cluster_indices[np.argmin(dists)]
        selected.append(IDX[closest_local])
        used.add(closest_local)

    # If we still need more to reach k (e.g., some clusters were empty), fill greedily
    while len(selected) < k:
        remaining = [i for i in range(n) if i not in used]
        if not remaining:
            break
        # pick the item that is farthest from the already selected set (diversity boost)
        if used:
            # compute min distance to any selected feature
            sel_feats = features[list(used)]
            dmin = []
            for i in remaining:
                dmin.append(np.min(np.linalg.norm(sel_feats - features[i], axis=1)))
            next_idx = remaining[int(np.argmax(dmin))]
        else:
            next_idx = remaining[0]
        selected.append(IDX[next_idx])
        used.add(next_idx)

    return selected



def find_most_different_k(filtered_matrix, k):
    mol_ids = list(filtered_matrix.index)

    # Case 1: Just 1 molecule — pick randomly
    if k == 1:
        return [mol_ids[0]], None

    # Case 2: More than 1 — do full comparison
    min_avg_sim = float('inf')
    best_combo = None

    for combo in itertools.combinations(mol_ids, k):
        sims = []
        for a, b in itertools.combinations(combo, 2):
            sim = filtered_matrix.loc[a, str(b)]
            sims.append(sim)

        avg_sim = np.mean(sims)

        if avg_sim < min_avg_sim:
            min_avg_sim = avg_sim
            best_combo = combo

    return best_combo, min_avg_sim


#     return

import matplotlib.pyplot as plt

def plot_bubble(df, title=None, max_bubble_area=3000):
    """
    Draws a bubble plot for the given DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must have:
          - index: identifier (will be shown on the x-axis)
          - 'Imbalance': y-axis values
          - 'Number_exp': used to size the bubbles
          - 'Positive', 'Negative': to choose colors
    title : str, optional
        Title to place above the plot.
    max_bubble_area : int, default=3000
        Maximum bubble area (matplotlib 's' parameter) for the largest Number_exp.
    """
    # Prepare x positions and labels
    identifiers = df.index.astype(str)
    x = range(len(identifiers))
    y = df['Imbalance']
    
    # Scale Number_exp → bubble areas
    # so the largest bubble has area = max_bubble_area
    norm = df['Number_exp'] / df['Number_exp'].max()
    sizes = norm * max_bubble_area
    
    # Choose colors
    colors = []
    for pos, neg in zip(df['Positive'], df['Negative']):
        if pos > neg:
            colors.append('red')
        elif neg > pos:
            colors.append('blue')
        else:
            colors.append('white')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        x, y,
        s=sizes,
        c=colors,
        edgecolor=['black' if c=='white' else 'none' for c in colors],
        linewidth=0.5,
        alpha=0.7
    )
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(identifiers, rotation=45, ha='right')
    ax.set_ylabel('Imbalance')
    ax.set_xlabel('Identifier')
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Positive > Negative',
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Negative > Positive',
               markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Tie',
               markerfacecolor='white', markeredgecolor='black', markersize=10),
    ]
    ax.legend(handles=legend_elements, title='Response')
    
    plt.tight_layout()
    return fig
