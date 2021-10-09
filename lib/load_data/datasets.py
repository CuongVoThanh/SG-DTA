import os
import numpy as np
import pandas as pd
import json, pickle
from tqdm import tqdm
from collections import OrderedDict
import logging

import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric import data as gemetric_data

from utils.create_data import seq_cat, convert_canonical_to_iso, smile_to_graph, target_to_graph, CHARISOSMISET, SEQ_DICT


class Datasets(InMemoryDataset):
    def __init__(self,
                root: str = './data',
                mode: str = 'train',
                dataset: str = 'davis',
                type_data: str = 'dataDTA',
                fold: int = 1,
                kfold: int = 5,
                transform = None,
                pre_transform = None) -> None:
        super(Datasets, self).__init__(root, transform, pre_transform)

        assert fold in range(1, 7), 'x must be greater than 1 and less than 6'

        self.root = os.path.join(root, dataset)
        self.dataset = dataset
        self.mode = mode
        self.fold = fold
        self.kfold = kfold
        self.type_data = type_data
        
        if not os.path.isdir(os.path.join(self.root, self.type_data)):
            os.mkdir(os.path.join(self.root, self.type_data))

        # k+1 is full-fold mode
        # validation set does not necessary on fullfold mode
        if self.mode == "test":
            self.file_name_test = self.dataset + '_test'
            self.files_name_train = f'train_{6}'
        else:            
            self.files_name_train = f'train_{self.fold}'

            if self.fold != 6:
                self.file_name_test = f'val_{self.fold}'                
            else:
                self.file_name_test = self.dataset + '_test'
        
        self.logger = logging.getLogger(__name__)

        self.process()

        if self.mode == "train":
            self.edge_index_drug_protein = torch.load(os.path.join(self.root, self.type_data, self.files_name_train + "_edge_index_drug_protein.pt"))
            self.data_mol, self.data_pro = torch.load(os.path.join(self.root, self.type_data, self.files_name_train + ".pt"))
        else:
            self.data_mol, self.data_pro = torch.load(os.path.join(self.root, self.type_data, self.file_name_test + ".pt"))
            self.edge_index_drug_protein = torch.load(os.path.join(self.root, self.type_data, self.files_name_train + "_edge_index_drug_protein.pt"))
            
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass
    
    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass
        
    def create_csv(self) -> None:
        if self.__is_exist_files(self.root, self.file_name_test, self.files_name_train, ".csv"):
            return

        ligands, proteins, affinity, k_folds, test_index = self.get_raw_data(self.root)

        if self.dataset == 'davis':
            affinity = [-np.log10(y/1e9) for y in affinity]
            affinity = np.asarray(affinity)
        
        drugs = [convert_canonical_to_iso(ligand) for ligand in ligands.values()]
        proteins_seq = list(proteins.values())
        proteins_name = list(proteins.keys())
        rows, cols = np.where(np.isnan(affinity)==False)

        if self.fold < 6:
            train_fold = k_folds[:(self.fold-1)] + k_folds[self.fold:]
            train_fold = [index for fold in train_fold for index in fold]
            val_fold = k_folds[self.fold-1]

            rows_fold_train, cols_fold_train = rows[train_fold], cols[train_fold]
            self.write_to_csv(self.root, self.files_name_train, rows_fold_train, cols_fold_train, 
                                drugs, proteins_seq, proteins_name, self.dataset, affinity)

            rows_fold_val, cols_fold_val = rows[val_fold], cols[val_fold]
            self.write_to_csv(self.root, self.file_name_test, rows_fold_val, cols_fold_val,
                                drugs, proteins_seq, proteins_name, self.dataset, affinity)        
        else:
            train_all = [index for fold in k_folds for index in fold]
            rows_train, cols_train = rows[train_all], cols[train_all]
            self.write_to_csv(self.root, self.files_name_train, rows_train, cols_train, 
                                drugs, proteins_seq, proteins_name, self.dataset, affinity)

            rows_val, cols_val = rows[test_index], cols[test_index]        

            self.write_to_csv(self.root, self.file_name_test, rows_val, cols_val, drugs, proteins_seq, proteins_name, self.dataset, affinity)
            
        self.logger.info('Processed dataset successfully')
        return

    def process(self) -> None:
        self.create_csv()
        root = os.path.join(self.root, self.type_data)
        if self.__is_exist_files(root, self.file_name_test, self.files_name_train, ".pt"):
            return

        des_train = "Train all fold" if self.fold == 6 else f"Train On Fold {self.fold}"
        des_test = "Test all fold" if self.fold == 6 else f"Validate On Fold {self.fold}"

        self.__pre_preprocessing_data(self.files_name_train, f'Convert {des_train} to torch format')
        self.__pre_preprocessing_data(self.file_name_test, f'Convert {des_test} to torch format')

        return

    def __pre_preprocessing_data(self, file_name: str, message: str) -> None:
        """ 
            Save data after collating
        """
        df_data = pd.read_csv(os.path.join(self.root, file_name + ".csv"))
        data_list_mol,data_list_pro = self.convert_to_torch_data(self.type_data, df_data, message, self.dataset)
        edge_index_drug_protein = torch.LongTensor([data_mol.edge_index_drug_protein for data_mol in data_list_mol]).transpose(1, 0)

        torch.save((data_list_mol, data_list_pro), os.path.join(self.root, self.type_data, file_name + ".pt"))

        edge_index_drug_protein = torch.cat((edge_index_drug_protein, edge_index_drug_protein[[1,0]]), 1)
        torch.save((edge_index_drug_protein), os.path.join(self.root, self.type_data, file_name + "_edge_index_drug_protein.pt"))

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]    

    @staticmethod
    def get_raw_data(path: str) -> tuple:
        """
            Get raw data: ligands, proteins, affinity, k_folds, val_index 
        """
        ligands = json.load(open(os.path.join(path, "ligands_can.txt")), object_pairs_hook=OrderedDict)
        proteins = json.load(open(os.path.join(path, "proteins.txt")), object_pairs_hook=OrderedDict)    

        affinity = pickle.load(open(os.path.join(path, "Y"), "rb"), encoding='latin1')
        
        k_folds = json.load(open(os.path.join(path, "folds/train_fold_setting.txt")))
        test_index = json.load(open(os.path.join(path, "folds/test_fold_setting.txt")))
        
        return ligands, proteins, affinity, k_folds, test_index

    @staticmethod
    def write_to_csv(root : str,
                    file_name_test: str, 
                    rows: np.array, 
                    cols: np.array,
                    drugs: list or dict, 
                    proteins_seq: list or dict,
                    proteins_name: list or dict,
                    dataset: str,
                    affinity: np.array) -> None:
        with open(os.path.join(root, file_name_test + ".csv"), 'w') as f:
            f.write('compound_iso_smiles,target_sequence,target_name,affinity,rows,cols,num_drug,num_protein\n')
            for pair_ind in tqdm(range(len(rows)), desc = f"Write to file {file_name_test}.csv"):
                row = []
                row += [drugs[rows[pair_ind]]]
                row += [proteins_seq[cols[pair_ind]]]
                row += [proteins_name[cols[pair_ind]]]
                if dataset == "drugchem":
                    row += [1]
                else:                    
                    row += [affinity[rows[pair_ind], cols[pair_ind]]]
                    row += [rows[pair_ind], cols[pair_ind], affinity.shape[0], affinity.shape[1]]
                f.write(','.join(map(str, row)) + '\n')

    def convert_to_torch_data(self, model: str, df: pd.DataFrame, des: str, dataset : None or str = None) -> list:
        """ 
            Generate list of Graph Data from dataframe 
        """ 

        drugs, proteins, proteins_name, affinity, rows, cols, num_drug, num_protein = \
                                        list(df['compound_iso_smiles']), list(df['target_sequence']),\
                                        list(df['target_name']), list(df['affinity']),\
                                        list(df['rows']), list(df['cols']),\
                                        list(df['num_drug']), list(df['num_protein'])
        drugs, proteins, affinity = np.asarray(drugs), np.asarray(proteins), np.asarray(affinity)

        if model == "dataDTA":
            return self.get_data_dta(proteins, drugs, affinity, rows, cols, num_drug, num_protein, des, dataset)

        elif model == "dgraphDTA":                        
            return self.get_data_dgraphDTA(proteins_name, proteins, drugs, affinity, rows, cols, num_drug, num_protein, des, dataset)

    @staticmethod
    def get_data_dgraphDTA(proteins_name : list, 
                           proteins : np.array, drugs : np.array, affinity : np.array, 
                           rows : list, cols : list, num_drug : list, num_protein : list, 
                           des : str , dataset : None or str = None):
        """ 
            Generate list data for dgraphDTA
        """ 
        data_list_drug = []
        data_list_pro = []
        msa_path = 'data/' + dataset + '/aln'
        contact_path = 'data/' + dataset + '/pconsc4'
        for i in tqdm(range(len(drugs)), desc = des):
            smiles = drugs[i]
            target_seq = proteins[i]
            target_name = proteins_name[i]
            labels = affinity[i]

            if not os.path.isfile(os.path.join(contact_path, target_name + '.npy')):
                raise Exception("Contact map of {} does not exist".format(target_name))

            _, features, edge_index = smile_to_graph(smiles)

            drug_record = gemetric_data.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            drug_record.edge_index_drug_protein = [rows[i], cols[i]+num_drug[i]]
            drug_record.num_drug = num_drug[i]
            drug_record.num_protein = num_protein[i]

            _, tar_features, tar_edge_index = target_to_graph(target_name, target_seq, contact_path, msa_path)

            target_graph = gemetric_data.Data(x=torch.Tensor(tar_features),
                                edge_index=torch.LongTensor(tar_edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))

            data_list_drug.append(drug_record)
            data_list_pro.append(target_graph)
            
        return data_list_drug, data_list_pro

    @staticmethod
    def get_data_dta(proteins : np.array, drugs : np.array, affinity : np.array, 
                    rows : list, cols : list, num_drug : list, num_protein : list, 
                    des : str , dataset : None or str = None):
        """ 
            Generate list data for dta
        """ 

        data_list_drug = []
        data_list_pro = []
        max_len_protein_deep = 1200 if dataset == "davis" else 1000
        proteins_one_hot_deep = [seq_cat(protein, SEQ_DICT, max_len_protein_deep) for protein in proteins]

        max_len_smile_deep = 85 if dataset == "davis" else 100
        smile_one_hot = [seq_cat(drug, CHARISOSMISET, max_len_smile_deep) for drug in drugs]

        proteins_one_hot = [seq_cat(protein) for protein in proteins]
        for i in tqdm(range(len(drugs)), desc = des):
            smiles = drugs[i]
            target = proteins_one_hot[i]
            labels = affinity[i]
            protein_deep = proteins_one_hot_deep[i]
            smile_deep = smile_one_hot[i]

            _ , features, edge_index = smile_to_graph(smiles)

            graph_record = gemetric_data.Data(x=torch.Tensor(features),
                            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            y=torch.FloatTensor([labels]))

            graph_record.smile_deep = torch.LongTensor([smile_deep])
            graph_record.edge_index_drug_protein = [rows[i], cols[i] + num_drug[i]]
            graph_record.num_drug = num_drug[i]
            graph_record.num_protein = num_protein[i]
            protein = gemetric_data.Data()
            protein.target = torch.LongTensor([target])
            protein.target_deep = torch.LongTensor([protein_deep])
            data_list_drug.append(graph_record)
            data_list_pro.append(protein)

        return data_list_drug, data_list_pro

    @staticmethod
    def __is_exist_files(root: str,
                        file_name_test: str,
                        files_name_train: str,
                        format: str) -> bool:
        """ 
            Check that valid input dataset files to run the model 
        """
        file_name_test = os.path.join(root,file_name_test)
        files_name_train = os.path.join(root,files_name_train)

        if os.path.exists(file_name_test + format) and os.path.exists(files_name_train + format):
            return True

        return False