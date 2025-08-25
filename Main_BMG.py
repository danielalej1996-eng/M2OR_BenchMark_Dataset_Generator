import Benchmark_generator_OR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def plot_one_vs_rest_gallery(similarity_matrix, pairs, rec_folds,
                             labels=None,
                             suptitle="One-vs-Rest (Test vs Train/Val)",
                             cmap="viridis", vmin=None, vmax=None,
                             hide_tick_numbers=True,
                             a4_orientation="portrait", dpi=300,
                             save_path=None,
                             id_col="main_receptors_id",
                             x_label=None, y_label=None):

   


    # unwrap rec_folds if it came as a single-item tuple
    if isinstance(rec_folds, tuple) and len(rec_folds) == 1 and isinstance(rec_folds[0], (list, tuple)):
        rec_folds = rec_folds[0]

    K = len(rec_folds)
    if labels is None:
        labels = [chr(65 + i) for i in range(K)]  # A, B, C, ...

    mat = np.asarray(similarity_matrix, dtype=float)

    # derive friendly axis token from id_col if labels not provided
    def _friendly_label(col):
        if col == "main_receptors_id":
            return "receptors_id"
        if col == "main_compounds_id":
            return "compounds_id"
        return col if col.endswith("_id") else (col + "_id")

    token = _friendly_label(id_col)
    if x_label is None:
        x_label = f"{token} (Train/Val set)"
    if y_label is None:
        y_label = f"{token} (Test set)"

    # helper: get entity IDs (1-based) for provided indices
    def ent_ids(idx):
        # idx are indices into `pairs` (whatever you used for fold construction)
        return np.unique(pairs.loc[idx, id_col])

    # build all panels (Rb = current fold rows; Ra = union of others as columns)
    panels = []
    mins, maxs = [], []
    for i in range(K):
        ids_test = rec_folds[i]
        ids_train = np.concatenate([np.asarray(rec_folds[j]) for j in range(K) if j != i], axis=0)

        Rb = ent_ids(ids_test)           # rows (test)
        Ra = ent_ids(ids_train)          # cols (train/val)

        # 1-based -> 0-based for slicing
        sub = mat[np.ix_(Rb - 1, Ra - 1)]
        panels.append((labels[i], sub, Ra, Rb))

        # accumulate for global color scale
        if sub.size:
            with np.errstate(all='ignore'):
                m1, m2 = np.nanmin(sub), np.nanmax(sub)
            if np.isfinite(m1): mins.append(m1)
            if np.isfinite(m2): maxs.append(m2)

    if vmin is None:
        vmin = (min(mins) if mins else None)
    if vmax is None:
        vmax = (max(maxs) if maxs else None)

    # A4 size in inches
    figsize = (8.27, 11.69) if a4_orientation == "portrait" else (11.69, 8.27)

    fig, axes = plt.subplots(K, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    axes = np.atleast_1d(axes)

    mappables = []
    for ax, (lab, sub, Ra, Rb) in zip(axes, panels):
        if sub.size:
            im = ax.imshow(sub, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            mappables.append(im)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            im = None

        ax.set_title(f"{lab} vs {' '.join([L for L in labels if L != lab])}", fontsize=12)

        if hide_tick_numbers:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(axis='both', length=0)
        else:
            ax.set_xticks(range(sub.shape[1]))
            ax.set_yticks(range(sub.shape[0]))
            ax.set_xticklabels(Ra, rotation=90)
            ax.set_yticklabels(Rb)

    # shared colorbar on the right (only one)
    if mappables:
        fig.colorbar(mappables[-1], ax=axes, fraction=0.025, pad=0.02)

    # Shared axis labels for the whole figure (Matplotlib >= 3.4; fallback provided)
    try:
        fig.supxlabel(x_label, fontsize=12)
        fig.supylabel(y_label, fontsize=12)
    except AttributeError:
        axes[-1].set_xlabel(x_label, fontsize=12)
        mid_ax = axes[len(axes) // 2]
        mid_ax.set_ylabel(y_label, fontsize=12)

    fig.suptitle(suptitle, fontsize=16)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return [(lab, sub) for lab, sub, _, _ in panels]


class benchmark_dataset_or:
    def __init__(self, Imb_lvl=3, th=0.6, LOG=False, Mutations = False, Sim_matrix_r = 'ALL_625.npy' ):
        self.Imb_lvl = Imb_lvl
        self.th = th
        self.Sim_matrix_r = Sim_matrix_r
        self.Sim_matrix_Receptors = np.load(self.Sim_matrix_r)
        self.LOG = LOG
        self.Mutations = Mutations
        
        self.receptors = pd.read_csv('M2OR/main_receptors.csv', sep=';',index_col='id')
        self.ligands = pd.read_csv('M2OR/main_compounds.csv', sep=';',index_col='id')
        self.pairs = pd.read_csv('M2OR/pairs.csv', sep=';',index_col='id')
        
        
        self.pair_id_bm_receptor_fold, self.pair_id_bm_ligant_fold, self.List_L, self.List_R = Benchmark_generator_OR.Benchmark_generator_ORF(Imb_lvl=self.Imb_lvl, th=self.th, LOG=self.LOG, Mutations = self.Mutations, Sim_matrix_r = self.Sim_matrix_r)
    
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
           "pair_id_bm_receptor_fold": self.pair_id_bm_receptor_fold,
           "pair_id_bm_ligant_fold": self.pair_id_bm_ligant_fold,
           "List_L": self.List_L,
           "List_R": self.List_R
       }   
   



    
if __name__ == "__main__":
    
    Imb_3 = benchmark_dataset_or()
    
    data = {
        'pair_id_bm_receptor_fold': Imb_3.pair_id_bm_receptor_fold,
        'pair_id_bm_ligant_fold':   Imb_3.pair_id_bm_ligant_fold
    }
#%%    
    Rec_folds = Imb_3.pair_id_bm_receptor_fold,
    pairs = Imb_3.pairs
    similarity_matrix = Imb_3.Sim_matrix_Receptors
    
    df = pd.read_csv("compound_sim_matrix.csv")

    similarity_matrix_compounds = df.iloc[:, 1:].to_numpy(dtype=float)   # keep all rows, skip first col
   
    
# _ = plot_one_vs_rest_gallery(
#     similarity_matrix,
#     pairs=pairs,
#     rec_folds=Rec_folds,
#     labels=["A","B","C","D","E"],
#     suptitle="Test vs Train/Val for all folds",
#     cmap="viridis",
#     hide_tick_numbers=True,
#     a4_orientation="portrait",   # vertical stack
#     dpi=300,
#     save_path="one_vs_rest_A4.png"  # opcional
# )

# plot_one_vs_rest_gallery(
#     similarity_matrix_compounds, pairs, Imb_3.pair_id_bm_ligant_fold,
#     id_col="main_compounds_id",
#     suptitle="Ligands: Test vs Train/Val for all folds"
# )
    # with open('bm_pair_ids_imb.pkl', 'wb') as f:
    #     pickle.dump(data, f)

    # print("Saved both folds to bm_pair_ids.pkl")