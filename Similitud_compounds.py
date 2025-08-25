import pandas as pd
import numpy as np 
import dataframe_utils as dfu
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator    
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import os

filename = "compound_sim_matrix.csv"


def tanimoto_sim(fp1, fp2):
    return  np.dot(fp1, fp2) / (np.sum(fp1) + np.sum(fp2) - np.dot(fp1, fp2))

def Compound_sim(Smile_A, Smile_B):


    
    Mol_A = Chem.MolFromSmiles(Smile_A)
    Mol_B = Chem.MolFromSmiles(Smile_B)
    
    
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp_A = np.array(list(fp_gen.GetFingerprint(Mol_A)))
    fp_B = np.array(list(fp_gen.GetFingerprint(Mol_B)))
    
    
    
    sim = tanimoto_sim(fp_B , fp_A)
    return sim

def get_sim_matrix(compounds):
    
    smiles_list = compounds['smiles'].tolist()
    n = len(smiles_list)

    # Create an empty (n x n) matrix
    sim_matrix = np.zeros((n, n))

    # Fill the matrix
    for i in tqdm(range(n), desc="Computing similarities"):
        for j in range(i, n):
            if i == j:
                sim = 1.0
            else:
                sim = Compound_sim(smiles_list[i], smiles_list[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    sim_df = pd.DataFrame(sim_matrix, columns=compounds.index, index=compounds.index)
 
    return sim_df
    
#%%

def Compounds_sim_matrix(compounds):
    if os.path.exists(filename):
        
        print("Similarity matrix already exists. Loading from file...")
        sim_df = pd.read_csv(filename, index_col=0)
        
    else:   
    
        sim_df = get_sim_matrix(compounds)    
        sim_df.to_csv(filename)
        print(f"Similarity matrix saved to {filename}")

    return sim_df 


