import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
import math

# Needleman–Wunsch with EMBOSS defaults
def NeedlemanWunsch(seq1, seq2):
    blosum62 = substitution_matrices.load("BLOSUM62")
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = blosum62
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    aln = aligner.align(seq1, seq2)[0]
    # Get aligned sequences
    a_str, b_str = str(aln).splitlines()[0], str(aln).splitlines()[2]
    # % identity
    matches = sum(x == y for x, y in zip(a_str, b_str))
    pct_identity = 100 * matches / len(a_str)
    return pct_identity


receptors = pd.read_csv('M2OR/main_receptors.csv', sep=';',index_col='id')

# Prepare similarity matrix
ids = receptors.index.tolist()
n = len(ids)
sim_matrix = pd.DataFrame(0.0, index=ids, columns=ids)

# Total unique pairs including diagonal
total_comparisons = n * (n + 1) // 2
progress_step = 0.2  # %
next_threshold = progress_step

done = 0
for i in range(n):
    for j in range(i, n):
        if i == j:
            sim = 100.0
        else:
            seq1 = receptors.at[ids[i], 'mutated_sequence']
            seq2 = receptors.at[ids[j], 'mutated_sequence']
            sim = NeedlemanWunsch(seq1, seq2)
        sim_matrix.iat[i, j] = sim
        sim_matrix.iat[j, i] = sim

        # Progress tracking
        done += 1
        progress = (done / total_comparisons) * 100
        if progress >= next_threshold:
            print(f"{progress:.1f}% completed...")
            # Advance threshold until it's ahead of current progress
            while next_threshold <= progress:
                next_threshold += progress_step

# Save results
sim_matrix.to_csv("similarity_matrix.csv")
print("Similarity matrix saved to similarity_matrix.csv")






    