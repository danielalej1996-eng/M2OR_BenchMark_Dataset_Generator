import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt

Receptor_id = 'main_receptors_id'
Compound_id = 'main_compounds_id'
Responsive = 'responsive'
AA_sequence = 'sequence'
Smiles = 'smiles'

AA_lut = {'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9,
          'L':10, 'M':11, 'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18,
          'W':19, 'Y':20}
Compounds_lut = {' ': 1 ,'#': 2 ,'(': 3 ,')': 4 ,'+': 5 ,'-': 6 ,'/': 7 ,
                 '1': 8 ,'2': 9 ,' 3 ': 10 ,' 4 ': 11 ,' 5 ': 12 ,'=': 13 ,' @ ': 14 ,
                 'B': 15 ,'C': 16 ,' H ': 17 ,' N ': 18 ,' O ': 19 ,' P ': 20 ,' S ': 21 ,
                 '[': 22 ,'\\': 23 ,']': 24 ,' l ': 25 ,' r ': 26 }

def extract_unique_letters(strings):

    unique_letters = set()   
    for s in strings:
        unique_letters.update(s)
    return sorted(unique_letters)

def extract_column(df, column_name):

    if column_name in df.columns:
        return df[column_name]
    else:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        
def import_M2OR():
    pairs = pd.read_csv('M2OR/pairs.csv', sep=';',index_col='id')
    
    
    # pairs_test = pairs[:10]
    # pairs = pairs_test
    
    Empty = np.zeros(len(pairs))
    data = {Receptor_id: Empty, AA_sequence: Empty, Compound_id: Empty, Smiles: Empty, Responsive: Empty}
    M2OR_data = pd.DataFrame(data=data, index = pairs.index)
    
    M2OR_data = M2OR_data.astype({
        Receptor_id: 'int64',    # Integer type
        AA_sequence: 'object',   # String type (pandas uses 'object' for strings)
        Compound_id: 'int64',    # Integer type
        Smiles: 'object',        # String type
        Responsive: 'int64'     # Integer type
    })
     
    for i in range(len(pairs)):
    
    
        
        M2OR_data.at[pairs.index[i],Receptor_id] = pairs.at[pairs.index[i], Receptor_id]
        M2OR_data.at[pairs.index[i],AA_sequence] = pairs.at[pairs.index[i], 'mutated_sequence']
        M2OR_data.at[pairs.index[i],Compound_id] = pairs.at[pairs.index[i], Compound_id]
        M2OR_data.at[pairs.index[i],Smiles] = pairs.at[pairs.index[i], Smiles]
        M2OR_data.at[pairs.index[i],Responsive] = pairs.at[pairs.index[i], Responsive]
     
    return M2OR_data

def import_M2OR_receptors():
    receptors = pd.read_csv('M2OR/main_receptors.csv', sep=';',index_col='id')
    Empty = np.zeros(len(receptors))
    data = {AA_sequence: Empty}
    receptors_sequences = pd.DataFrame(data=data, index = receptors.index)

    receptors_sequences = receptors_sequences.astype({
        
        AA_sequence: 'object',   # String type (pandas uses 'object' for strings)

    })
         
    for i in range (len(receptors_sequences)):
         receptors_sequences.at[receptors.index[i],AA_sequence] = receptors.at[receptors.index[i],'mutated_sequence']
    return receptors_sequences


def import_M2OR_compounds():
    compounds = pd.read_csv('M2OR/main_compounds.csv', sep=';',index_col='id')
    Empty = np.zeros(len(compounds))
    data = {Smiles: Empty}
    Compound_smile = pd.DataFrame(data=data, index = compounds.index)
    
    Compound_smile = Compound_smile.astype({
        
        Smiles: 'object',   # String type (pandas uses 'object' for strings)
    
    })
         
    for i in range (len(Compound_smile)):
         Compound_smile.at[compounds.index[i],Smiles] = compounds.at[compounds.index[i],'smiles']
     
    return Compound_smile

def import_M2OR_pairs():
    pairs = pd.read_csv('M2OR/pairs.csv', sep=';',index_col='id')
    
    
    # pairs_test = pairs[:10]
    # pairs = pairs_test
    
    Empty = np.zeros(len(pairs))
    data = {Receptor_id: Empty,  Compound_id: Empty,  Responsive: Empty}
    M2OR_data = pd.DataFrame(data=data, index = pairs.index)
    
    M2OR_data = M2OR_data.astype({
        Receptor_id: 'int64',    # Integer type
        Compound_id: 'int64',    # Integer type
        Responsive: 'int64'     # Integer type
    })
     
    for i in range(len(pairs)):
         
        M2OR_data.at[pairs.index[i],Receptor_id] = pairs.at[pairs.index[i], Receptor_id]
        M2OR_data.at[pairs.index[i],Compound_id] = pairs.at[pairs.index[i], Compound_id]
        M2OR_data.at[pairs.index[i],Responsive] = pairs.at[pairs.index[i], Responsive]
     
    return M2OR_data



def search_experiments(df, search_column, search_value, return_complement=False):
    """
    Searches for experiments in a DataFrame based on a column and value(s).

    Parameters:
    df (pd.DataFrame): The DataFrame containing experiment data.
    search_column (str): The column to search in (e.g., 'Molecule' or 'Protein').
    search_value (int, str, or list): The value(s) to search for in the specified column.
    return_complement (bool): If True, returns both the filtered DataFrame and the original DataFrame
                              without the filtered rows. If False, returns only the filtered DataFrame.

    Returns:
    pd.DataFrame or tuple: If return_complement is False, returns the filtered DataFrame.
                           If return_complement is True, returns a tuple (filtered_df, complement_df).
    """
    if search_column not in df.columns:
        raise ValueError(f"Column '{search_column}' not found in DataFrame.")

    # Condition 1: If search_value is a single integer or string
    if isinstance(search_value, (int, str)):
        filtered_df = df[df[search_column] == search_value]
    
    # Condition 2: If search_value is a list or vector
    elif isinstance(search_value, (list, pd.Series)):
        filtered_df = df[df[search_column].isin(search_value)]
    
    else:
        raise TypeError("search_value must be an integer, string, or list.")

    # If return_complement is True, calculate the complement DataFrame
    if return_complement:
        complement_df = df[~df.index.isin(filtered_df.index)]
        return filtered_df, complement_df
    
    # Otherwise, return only the filtered DataFrame
    return filtered_df

def get_experiments_distribution(df,column): 

    List = np.unique(df[column])
    data_df_synthesis= {'Number_exp': np.zeros(len(List)), 'Positive': np.zeros(len(List)), 'Negative': np.zeros(len(List))}
    df_exp_s = pd.DataFrame(data = data_df_synthesis, index = List)
    
    
    
    for i in range(len(List)):   

        df_experiments = search_experiments(df, column, int(List[i]))
        pos, neg = search_experiments(df_experiments, Responsive, 1, return_complement=True)
        df_exp_s.at[int(List[i]),'Number_exp'] = len(df_experiments)
        df_exp_s.at[int(List[i]),'Positive'] = len(pos)
        df_exp_s.at[int(List[i]),'Negative'] = len(neg)
        
    return df_exp_s


def get_size_df(Distibution, flex):

   TG_DS = Distibution.loc[ Distibution['Positive'] != 0]
   TG_DS = TG_DS.loc[TG_DS ['Negative'] != 0]
   
   for i in range(len(TG_DS)):
       
    if (TG_DS.at[TG_DS.index[i],'Negative']/TG_DS.at[TG_DS.index[i],'Positive'] <=flex and TG_DS.at[TG_DS.index[i],'Positive']/TG_DS.at[TG_DS.index[i],'Negative'] <=flex):
    
        TG_DS.at[TG_DS.index[i],'Positive'] = TG_DS.at[TG_DS.index[i],'Positive']
        TG_DS.at[TG_DS.index[i],'Negative'] = TG_DS.at[TG_DS.index[i],'Negative']
        if TG_DS.at[TG_DS.index[i],'Negative']/TG_DS.at[TG_DS.index[i],'Positive'] >= 1:
            TG_DS.at[TG_DS.index[i],'Imbalance'] = TG_DS.at[TG_DS.index[i],'Negative']/TG_DS.at[TG_DS.index[i],'Positive']
        else:
            TG_DS.at[TG_DS.index[i],'Imbalance'] = TG_DS.at[TG_DS.index[i],'Positive']/TG_DS.at[TG_DS.index[i],'Negative']
      
    if (TG_DS.at[TG_DS.index[i],'Negative']/TG_DS.at[TG_DS.index[i],'Positive'] >flex):
    
        TG_DS.at[TG_DS.index[i],'Positive'] = TG_DS.at[TG_DS.index[i],'Positive']
        TG_DS.at[TG_DS.index[i],'Negative'] = TG_DS.at[TG_DS.index[i],'Positive']*flex
        TG_DS.at[TG_DS.index[i],'Imbalance'] = TG_DS.at[TG_DS.index[i],'Negative']/TG_DS.at[TG_DS.index[i],'Positive'] 
     
    if (TG_DS.at[TG_DS.index[i],'Positive']/TG_DS.at[TG_DS.index[i],'Negative'] >flex):
    
        TG_DS.at[TG_DS.index[i],'Positive'] = TG_DS.at[TG_DS.index[i],'Negative']*flex
        TG_DS.at[TG_DS.index[i],'Negative'] = TG_DS.at[TG_DS.index[i],'Negative']
        TG_DS.at[TG_DS.index[i],'Imbalance'] = TG_DS.at[TG_DS.index[i],'Positive']/TG_DS.at[TG_DS.index[i],'Negative']

    TG_DS.at[TG_DS.index[i],'Number_exp'] =  TG_DS.at[TG_DS.index[i],'Positive'] + TG_DS.at[TG_DS.index[i],'Negative']
    TG_DS.at[TG_DS.index[i],'pair_id_pos'] = None
    TG_DS.at[TG_DS.index[i],'pair_id_neg'] = None
    TG_DS.at[TG_DS.index[i],'pos_cond_c'] = 0
    TG_DS.at[TG_DS.index[i],'neg_cond_c'] = 0
    TG_DS.at[TG_DS.index[i],'Exp_Av_P'] = None
    TG_DS.at[TG_DS.index[i],'Exp_Av_N'] = None
        
   return TG_DS

def move_element(split_arrays, source_idx, dest_idx):
    """
    Move a random element from split_arrays[source_idx] to split_arrays[dest_idx].
    split_arrays: list of numpy arrays or pandas Series
    source_idx: index of the fold to take from (the “high” one)
    dest_idx:   index of the fold to give to  (the “low” one)
    Returns the updated list of folds.
    """
    src = split_arrays[source_idx]
    dst = split_arrays[dest_idx]

    # turn each into a python list for easy remove/append
    src_list = list(src)
    dst_list = list(dst)

    # pick and move one element
    element = random.choice(src_list)
    src_list.remove(element)
    dst_list.append(element)

    # put them back, preserving original type
    if isinstance(src, pd.Series):
        split_arrays[source_idx] = pd.Series(src_list, index=range(len(src_list)))
    else:
        split_arrays[source_idx] = np.array(src_list)

    if isinstance(dst, pd.Series):
        split_arrays[dest_idx] = pd.Series(dst_list, index=range(len(dst_list)))
    else:
        split_arrays[dest_idx] = np.array(dst_list)

    return split_arrays

# # Receptors_LUT 

# Df = Compounds
# Column_to_transform = Smiles
# Characters = []
# for i in range(len(Df)):

#     Characters_temp = dfu.extract_unique_letters(Df.at[Df.index[i],Column_to_transform])
#     Characters = np.append(Characters, Characters_temp)
#     Characters = np.unique(Characters)

# print(Characters)

# for i in range(len(Characters)):
#     print("'",Characters[i],"':",i+1,",")

def splits_folds(folds, List):



    unique_clusters = np.unique(List['cluster'])
    
    # create a new DataFrame with an empty “num exp” column
    unique_clusters = pd.DataFrame({
        'cluster': unique_clusters,
        'Number_exp': [pd.NA] * len(unique_clusters)
    })
    
    for i in range (len(unique_clusters)):
        #print(unique_clusters.at[i,'cluster'])
        Fi = search_experiments(List, 'cluster', [unique_clusters.at[i,'cluster']])
        #print(np.sum(Fi['Number_exp']))
        unique_clusters.at[i,'Number_exp'] = np.sum(Fi['Number_exp'])
        
    
    cluster = unique_clusters['cluster']
    np.random.seed(1)
    # np.random.shuffle(cluster)   
    split_arrays = np.array_split(cluster, folds)
    Mem_Sum_exp = np.zeros(5)
    
    while True: 
        
        
        for i in range(5):
            Mem_Sum_exp[i] = sum(unique_clusters.loc[split_arrays[i],'Number_exp'])
            
        
        #print(Mem_Sum_exp)
        H = int(np.where(Mem_Sum_exp == np.max(Mem_Sum_exp))[0][0])
        L = int(np.where(Mem_Sum_exp == np.min(Mem_Sum_exp))[0][0])

          
        if(np.max(Mem_Sum_exp)-np.min(Mem_Sum_exp))<=1:
            #print(Mem_Sum_exp)
            break
        else:
            
            split_arrays = move_element(split_arrays,H,L)
            

        
    return split_arrays


blast = pd.read_csv('M2OR/blast.csv', sep=';',index_col='id')
receptors = pd.read_csv('M2OR/main_receptors.csv', sep=';',index_col='id')

def Chek_similitud(List_RA, List_RB, size):
    Matrix_list_sim_all = np.load('ALL_625.npy')



    Matrix_list_sim = np.zeros([len(List_RA),len(List_RB)])
    List = np.array(size.index)


    for i in range (len(List_RA)):

        for j in range (len(List_RB)):
            A = int(List_RA [i])
            B = int(List_RB [j])


            Matrix_list_sim[i,j] = Matrix_list_sim_all[(np.where(List == A)[0][0]),(np.where(List == B)[0][0])]
            
    return Matrix_list_sim

def move_elements(A, B, C):

    A = np.append(A, B[C])  # Add selected elements to A
    B = np.delete(B, C)  # Remove selected elements from B
    
    return A, B


def Sim_treatment(splits,A,B,size_a):
    
    List_A = splits[A]
    List_B = splits[B]
    TresHold = 90
    
    
    Matrix_sim = Chek_similitud(List_A, List_B, size_a)
    Sim_Receptors = np.unique(np.where(Matrix_sim >= TresHold)[1])
    List_A, List_B = move_elements(List_A, List_B, Sim_Receptors)
    
    
    
    Cont2 = 0 
    while True:
        Cont = 0
        
        if sum(size_a.loc[List_A,'Number_exp']) >= sum(size_a.loc[List_B,'Number_exp']):
            while True:
                Cont += 1
                if Cont  % 1000 == 0:
                    print('Loop1:',Cont)
                if Cont == 2000:
                    Cont2 += 1
                    break
                Candidate_to_pass = np.random.choice(List_A)
                frs = np.delete(List_A,np.where(List_A == Candidate_to_pass))
                Cand_similitud = Chek_similitud(frs, [Candidate_to_pass], size_a)
                derer = Cand_similitud[np.where(Cand_similitud >= TresHold)]
                if derer.sum() == 0:
                    List_A = frs
                    List_B = np.append(List_B, Candidate_to_pass)
                    break
            
                

            
    
        
        if sum(size_a.loc[List_B,'Number_exp']) >= sum(size_a.loc[List_A,'Number_exp']):
            
            while True:
                Cont += 1
                if Cont  % 1000 == 0:
                    print('Loop2:',Cont)
                if Cont == 2000:
                    Cont2 += 1
                    break
                Candidate_to_pass = np.random.choice(List_B)
                frs = np.delete(List_B,np.where(List_B == Candidate_to_pass))
                Cand_similitud = Chek_similitud(frs, [Candidate_to_pass], size_a)
                derer = Cand_similitud[np.where(Cand_similitud >= TresHold)]
                if derer.sum() == 0:
                    List_A = np.append(List_A, Candidate_to_pass) 
                    List_B = frs
                    break

    
        if abs(sum(size_a.loc[List_B,'Number_exp']) - sum(size_a.loc[List_A,'Number_exp'])) <= 3:
            break
        if Cont2 == 1:
            break

    
    splits[A] = List_A
    splits[B] = List_B  
    
    Mem_Sum_exp = np.zeros(5)
    for i in range(5):
        Mem_Sum_exp[i] = sum(size_a.loc[splits[i],'Number_exp'])
        
    # print(Mem_Sum_exp)
    
    return splits


from sklearn.cluster import AgglomerativeClustering




def clustering_and_split_receptors(size_by_receptors,th, Sim_matrix_r):
    
    similarity_matrix = np.load(Sim_matrix_r)
    
    ids_to_keep = size_by_receptors.index.to_numpy()
    idx0 = ids_to_keep - 1
    filtered_similarity_matrix = similarity_matrix[np.ix_(idx0, idx0)]
    
    #filtered_similarity_matrix = np.delete(similarity_matrix, positions, axis=0)  # remove rows
    #filtered_similarity_matrix = np.delete(filtered_similarity_matrix, positions, axis=1)  # remove columns
    
    
    distance_matrix = (100 - filtered_similarity_matrix)/100
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='single',
        distance_threshold=1-th
    )
    
    labels = clustering.fit_predict(distance_matrix)
    
    size_by_receptors['cluster'] = labels
    
    
    splits_by_cluster = splits_folds(5, size_by_receptors)
    
   
    return splits_by_cluster, size_by_receptors


from Similitud_compounds import Compounds_sim_matrix

def clustering_and_split_compounds(M2OR_compounds, size_by_compound,th):
    
    
    CSM = Compounds_sim_matrix(M2OR_compounds)
    
    matrix_CMS = CSM.to_numpy()
    
    index_c = np.array(size_by_compound.index)
    
    
    # Adjust the index_c from 1-based to 0-based
    index_c_0_based = [i - 1 for i in index_c]
    
    # Keep the rows and columns based on index_c_0_based
    CSM_kept = CSM.iloc[index_c_0_based, index_c_0_based]  # Select rows and columns
    
    matrix_CMS = CSM_kept.to_numpy()


    
    distance_matrix = (1 - matrix_CMS)
    
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='single',
        distance_threshold=1-th
    )
    
    labels = clustering.fit_predict(distance_matrix)
    
    size_by_compound['cluster'] = labels
    
    size_by_cluster = size_by_compound.groupby('cluster')['Number_exp'].sum().reset_index()
    splits_by_cluster = splits_folds(5, size_by_cluster)

    return splits_by_cluster, size_by_compound



import matplotlib.patches as mpatches
import pandas as pd

def plot_receptor_ligand_pairs(df,
                               x_col="main_receptors_id",
                               y_col="main_compounds_id",
                               label_col="responsive",
                               sample=None,
                               highlight_df=None,
                               size=30,
                               alpha=0.6,
                               highlight_size=None,
                               highlight_color="yellow",
                               highlight_marker="o",
                               highlight_alpha=1.0,
                               x_quantile=None,          # NEW: keep only points with x ≥ this quantile (0–1)
                               figsize=(16, 12),
                               ordinal=True,
                               colors={0: "blue", 1: "red"}):
    """
    Scatter-plot every receptor–ligand pair in `df` (coloured by `label_col`),
    optionally restricting to only the top receptors by x_quantile,
    then over-plot the rows in `highlight_df`.
    
    Parameters
    ----------
    x_quantile : float or None
        If set to e.g. 0.85, only the points whose x-value is ≥ the 85th percentile
        will be shown (i.e. top 15%).
    """
    # — optionally down-sample
    if sample is not None and sample < len(df):
        df = df.sample(sample, random_state=0)

    # — encode to ordinals or numeric
    if ordinal:
        cat_x = pd.Categorical(df[x_col])
        cat_y = pd.Categorical(df[y_col])
        x_vals = cat_x.codes
        y_vals = cat_y.codes
    else:
        x_vals = df[x_col].to_numpy()
        y_vals = df[y_col].to_numpy()

    # — apply x_quantile filter if requested
    if x_quantile is not None:
        cutoff = np.quantile(x_vals, x_quantile)
        mask = x_vals >= cutoff
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]
        df = df.iloc[mask]            # keep df in sync for colours
    # — map labels → colours
    colour_vals = df[label_col].map(colors).to_numpy()

    # — plot main points
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x_vals, y_vals, s=size, alpha=alpha, c=colour_vals, label="all points")

    # — now plot highlights
    if highlight_df is not None and not highlight_df.empty:
        # encode highlight coords
        if ordinal:
            xh = pd.Categorical(highlight_df[x_col], categories=cat_x.categories).codes
            yh = pd.Categorical(highlight_df[y_col], categories=cat_y.categories).codes
        else:
            xh = highlight_df[x_col].to_numpy()
            yh = highlight_df[y_col].to_numpy()
        # if x_quantile was set, drop any highlights below the same cutoff
        if x_quantile is not None:
            mask_h = xh >= cutoff
            xh, yh = xh[mask_h], yh[mask_h]
        s2 = highlight_size or (size * 2)
        ax.scatter(xh, yh,
                   s=s2,
                   c=highlight_color,
                   marker=highlight_marker,
                   alpha=highlight_alpha,
                   edgecolors="black",
                   linewidths=1.2,
                   label="highlighted")

    # — labels & tidy
    ax.set_xlabel("Receptor ID" if not ordinal else "Receptor (ordinal index)")
    ax.set_ylabel("Ligand ID"   if not ordinal else "Ligand (ordinal index)")
    ax.set_title("Receptor–Ligand Interaction Map")
    if ordinal:
        ax.set_xticks([]); ax.set_yticks([])

    # — legend
    handles = [mpatches.Patch(color=v, label=f"{label_col} = {k}") for k, v in colors.items()]
    if highlight_df is not None:
        handles.append(mpatches.Patch(color=highlight_color, label="highlighted"))
    ax.legend(handles=handles, title="Legend")

    plt.tight_layout()
    plt.show()

def flatten_pair_ids(df,
                     pos_col="pair_id_pos",
                     neg_col="pair_id_neg",
                     *,
                     deduplicate=False,      # True → keep only unique IDs
                     return_numpy=True,      # False → return a Python list
                     parse_strings=True):    # True → convert "[1,2]" strings → lists

    # explode() turns each list element into its own row
    long_ids = pd.concat([
        df[pos_col].explode(),
        df[neg_col].explode()
    ]).dropna().astype(int)

    if deduplicate:
        long_ids = pd.Series(long_ids.unique())

    vector = long_ids.to_numpy() if return_numpy else long_ids.tolist()
    return vector