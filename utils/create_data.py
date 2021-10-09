import numpy as np
import os

from rdkit import Chem
from torch_geometric.data import Batch

import networkx as nx

""" 
    We modified source code for create_data.py by the following sources 
    Github: 
        - https://github.com/thinng/GraphDTA
        - https://github.com/595693085/DGraphDTA
        - https://github.com/ngminhtri0394/GEFA
"""


LIST_ATOM = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {v:(i+1) for i, v in enumerate(SEQ_VOC)}

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

PRO_RES_TABLE = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

PRO_RES_ALIPHATIC_TABLE = ['A', 'I', 'L', 'M', 'V']
PRO_RES_AROMATIC_TABLE = ['F', 'W', 'Y']
PRO_RES_POLAR_NEUTRAL_TABLE = ['C', 'N', 'Q', 'S', 'T']
PRO_RES_ACIDIC_CHARGED_TABLE = ['D', 'E']
PRO_RES_BASIC_CHARGED_TABLE = ['H', 'K', 'R']

RES_WEIGHT_TABLE = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

RES_PKA_TABLE = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

RES_PKB_TABLE = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

RES_PKX_TABLE = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

RES_PL_TABLE = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

RES_HYDROPHOBIC_PH2_TABLE = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

RES_HYDROPHOBIC_PH7_TABLE = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}


def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    dic = dict(map(lambda key: (key, (dic[key] - min_value) / interval), dic.keys()))
    dic['X'] = (max_value + min_value) / 2.0
    return dic

RES_WEIGHT_TABLE = dic_normalize(RES_WEIGHT_TABLE)
RES_PKA_TABLE = dic_normalize(RES_PKA_TABLE)
RES_PKB_TABLE = dic_normalize(RES_PKB_TABLE)
RES_PKX_TABLE = dic_normalize(RES_PKX_TABLE)
RES_PL_TABLE = dic_normalize(RES_PL_TABLE)
RES_HYDROPHOBIC_PH2_TABLE = dic_normalize(RES_HYDROPHOBIC_PH2_TABLE)
RES_HYDROPHOBIC_PH7_TABLE = dic_normalize(RES_HYDROPHOBIC_PH7_TABLE)

PROTEIN_PROPERTIES = 12

def residue_features(residue):
    res_property1 = [1 if residue in PRO_RES_ALIPHATIC_TABLE else 0, 
                     1 if residue in PRO_RES_AROMATIC_TABLE else 0,
                     1 if residue in PRO_RES_POLAR_NEUTRAL_TABLE else 0,
                     1 if residue in PRO_RES_ACIDIC_CHARGED_TABLE else 0,
                     1 if residue in PRO_RES_BASIC_CHARGED_TABLE else 0]
    res_property2 = [RES_WEIGHT_TABLE[residue], RES_PKA_TABLE[residue], RES_PKB_TABLE[residue], RES_PKX_TABLE[residue],
                     RES_PL_TABLE[residue], RES_HYDROPHOBIC_PH2_TABLE[residue], RES_HYDROPHOBIC_PH7_TABLE[residue]]
                     
    return np.array(res_property1 + res_property2)

def atom_features(atom):
    feature = np.array(one_of_k_encoding_unk(atom.GetSymbol(), LIST_ATOM) +
                    one_of_k_encoding(atom.GetDegree(), range(11)) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), range(11)) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), range(11)) +
                    [atom.GetIsAromatic()])
    
    return feature/sum(feature)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    
    features = [atom_features(atom) for atom in mol.GetAtoms()]

    edges = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]

    g = nx.Graph(edges).to_directed()
    edge_index = [[e1, e2] for e1, e2 in g.edges]
    return c_size, features, edge_index

def seq_cat(seq, seq_dict = SEQ_DICT, max_seq_len = 1000):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(seq[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

def convert_canonical_to_iso(ligand, isomericSmiles = True):
    return Chem.MolToSmiles(Chem.MolFromSmiles(ligand), isomericSmiles = isomericSmiles)

def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(PRO_RES_TABLE), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                continue
            count = 0
            for res in line:
                if res not in PRO_RES_TABLE:
                    count += 1
                    continue
                pfm_mat[PRO_RES_TABLE.index(res), count] += 1
                count += 1
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    return pssm_mat

def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(PRO_RES_TABLE)))
    pro_property = np.zeros((len(pro_seq), PROTEIN_PROPERTIES))
    for i in range(len(pro_seq)):
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], PRO_RES_TABLE)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    other_feature = seq_feature(pro_seq)
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)

def target_to_feature(target_key, target_sequence, aln_dir):
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    feature = target_feature(aln_file, target_sequence)
    return feature

def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)
    target_edge_index = np.vstack((index_row, index_col)).T
    target_feature = target_to_feature(target_key, target_sequence, aln_dir)
    target_edge_index = np.array(target_edge_index)
    return target_size, target_feature, target_edge_index

def collate(dataset):
    batch_drug = Batch.from_data_list([data[0] for data in dataset])
    batch_protein = Batch.from_data_list([data[1] for data in dataset])
    return batch_drug, batch_protein