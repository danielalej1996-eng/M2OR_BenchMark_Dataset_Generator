# import numpy as np
# import pandas as pd
import dataframe_utils as dfu
# import Graphs_Funtions
# import random
# from Similitud_c import bio_seq_comp, bio_seq_comp_data
# from sklearn.cluster import AgglomerativeClustering
# import math
# import Similitud_compounds as sc
# import itertools
# from tqdm import tqdm
# import copy
# import P1DFG
# from Similitud_compounds import Compounds_sim_matrix
# import DSR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize



def bubble_plot_newX(
    df,
    id_col=None,
    num_exp_col='Number_exp',
    pos_col='Positive',
    neg_col='Negative',
    size_scale=7,
    vmax_pct=99,
    cmap_name='coolwarm',
    over_color='#FFD400',    # +∞ color
    under_color='#3B0F70',   # −∞ color
    inf_label_size=16,       # BIG ±∞ tags
    x_label='X = 1 − 1/|IR|',
    y_label='ID',
    title='Bubble plot: size=A=Pos+Neg, color=IR (±∞ highlighted)',
    x_min=-0.05, x_max=1.1, x_tick_step=0.1
):
    ids = (df[id_col] if id_col else df.index.to_series(name='ID')).to_numpy()
    pos = df[pos_col].astype(float).to_numpy()
    neg = df[neg_col].astype(float).to_numpy()
    num_exp = df[num_exp_col].astype(float).to_numpy()  # only used for size A
    A = pos + neg

    # Signed IR (keep ±∞)
    with np.errstate(divide='ignore', invalid='ignore'):
        ir_pos = pos / neg          # -> +inf when neg==0 & pos>0
        ir_neg = -(neg / pos)       # -> -inf when pos==0 & neg>0
        IR = np.where(pos > neg, ir_pos, ir_neg)  # 0/0 stays NaN

    # --- NEW X = 1 - 1/|IR| ---
    with np.errstate(divide='ignore', invalid='ignore'):
        X = 1 - 1 / np.abs(IR)      # = 1 for |IR|=∞, NaN for IR=NaN, -∞ for IR=0

    # Color mapping (finite center, ±∞ as over/under)
    finite_IR = IR[np.isfinite(IR)]
    v = np.nanpercentile(np.abs(finite_IR), vmax_pct) if finite_IR.size else 1.0
    v = float(v) if v > 0 else 1.0

    IR_for_plot = IR.copy()
    IR_for_plot[np.isposinf(IR_for_plot)] = v + 1   # +∞ → "over"
    IR_for_plot[np.isneginf(IR_for_plot)] = -v - 1  # −∞ → "under"

    norm = Normalize(vmin=-v, vmax=v, clip=False)
    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_over(over_color)
    cmap.set_under(under_color)

    # Keep rows where X is finite; allow ±∞ in IR (already mapped)
    mask = np.isfinite(X) & np.isfinite(IR_for_plot)
    Xm, Ym = X[mask], ids[mask]
    Am, Cm = A[mask], IR_for_plot[mask]

    # --- draw higher X on top ---
    order = np.argsort(Xm)              # low → high; high drawn last (on top)
    Xm, Ym, Am, Cm = Xm[order], Ym[order], Am[order], Cm[order]

    # Plot (no borders)
    fig, ax = plt.subplots(figsize=(6, 10))
    sc = ax.scatter(
        Xm, Ym,
        s=Am * size_scale,
        c=Cm, cmap=cmap, norm=norm,
        alpha=0.75,
        edgecolors='none', linewidths=0.0
    )

    # Full x-axis from -0.2 to 1.2 with labels
    ax.set_xlim(x_min, x_max)
    ticks = np.arange(x_min, x_max + 1e-9, x_tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:.1f}' for t in ticks])

    # --- slimmer/shorter colorbar ---
    cbar = plt.colorbar(
        sc, ax=ax, extend='both',
        fraction=0.025,   # width of the colorbar (smaller = thinner)
        pad=0.1,         # gap between plot and bar
        shrink=0.60,      # shorten the bar length
        aspect=30         # long:short ratio; bigger => thinner
    )
    cbar.set_label('Imbalance Ration (IR)')
    
    # keep the big ±∞ tags
    cbar.ax.text(0.5, 1.06, '+∞', transform=cbar.ax.transAxes,
                 ha='center', va='bottom', fontsize=16, fontweight='bold')
    cbar.ax.text(0.5, -0.10, '−∞', transform=cbar.ax.transAxes,
                 ha='center', va='top', fontsize=16, fontweight='bold')
    

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.25)
    plt.tight_layout()
    plt.show()


# --- usage:
# bubble_plot(df)                     # if your IDs are the index
# bubble_plot(df, id_col='your_id')   # if IDs live in a column


Imb_lvl = 3
LOG = False
Mutations = False
if LOG:print('-'*45)
if LOG:print('Importing data')
if LOG:print('-'*45)
M2OR_full_data = dfu.import_M2OR()
print("M2OR data has been imported.")
M2OR_receptors = dfu.import_M2OR_receptors()
print("M2OR_receptors data has been imported.")
M2OR_compounds = dfu.import_M2OR_compounds()
print("M2OR_compound data has been imported.")
if LOG:print('-'*45)

dfu.plot_receptor_ligand_pairs(M2OR_full_data)




if LOG:print('-'*45)    
if LOG:print('Getting_size_by_receptors')
size_by_receptor = (dfu.get_experiments_distribution(M2OR_full_data, dfu.Receptor_id))
if LOG:print('-'*45) 
if LOG:print('Getting_size_by_compounds')
size_by_compound  = (dfu.get_experiments_distribution(M2OR_full_data, dfu.Compound_id))
if LOG:print('-'*45)    

bubble_plot_newX(size_by_receptor, x_label='Degree of Imbalance',
    y_label='Receptor_ID',
    title='Receptor Imbalance')

bubble_plot_newX(size_by_compound, x_label='Degree of Imbalance',
    y_label='Ligand_ID',
    title='Ligand Imbalance')
