import Benchmark_generator_OR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt







class benchmark_dataset_or:
    def __init__(self, Imb_lvl=3, th_c=0.6, th_r = 0.6, LOG=False, Mutations = False, Sim_matrix_r = 'receptor_sim_matrix.npy' ):
        self.Imb_lvl = Imb_lvl
        self.th_c = th_c
        self.th_r = th_r
        self.Sim_matrix_r = Sim_matrix_r
        self.Sim_matrix_Receptors = np.load(self.Sim_matrix_r)
        self.LOG = LOG
        self.Mutations = Mutations
        
        self.receptors = pd.read_csv('M2OR/main_receptors.csv', sep=';',index_col='id')
        self.ligands = pd.read_csv('M2OR/main_compounds.csv', sep=';',index_col='id')
        self.pairs = pd.read_csv('M2OR/pairs.csv', sep=';',index_col='id')
        
        
        self.cold_receptor_split, self.cold_ligand_split, self.List_L, self.List_R = Benchmark_generator_OR.Benchmark_generator_ORF(Imb_lvl=self.Imb_lvl, th_c=self.th_c, th_r=self.th_r, LOG=self.LOG, Mutations = self.Mutations, Sim_matrix_r = self.Sim_matrix_r)
    
    def get_parameters_generation(self):
       """
       Return all attributes and data from the object in a dictionary.
       """
       return {
           "Imb_lvl": self.Imb_lvl,
           "th": self.th,
           "Sim_matrix_r": self.Sim_matrix_r,
           "LOG": self.LOG,
           "Mutations": self.Mutations,
           "pair_id_bm_receptor_fold": self.cold_receptor_split,
           "pair_id_bm_ligand_fold": self.cold_ligand_split,
           "List_L": self.List_L,
           "List_R": self.List_R
       }   
   



    
if __name__ == "__main__":
  
    Imb_3 = benchmark_dataset_or()
    Imb_10 = benchmark_dataset_or(Imb_lvl=10)
    Imb_20 = benchmark_dataset_or(Imb_lvl=20)

    
