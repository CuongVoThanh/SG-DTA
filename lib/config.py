import torch

from lib.models import graph_embedding
from lib.models import node_embedding
from lib.models import models_protein

from lib.load_data.datasets import Datasets
import lib.models as models

KFOLD = 5

MAP_DRUG_MODEL = {
    "gcn": "DrugGCNNet", 
    "gat": "DrugGATNet", 
    "ginconv": "DrugGINConvNet",
    "gat_gcn": "DrugGAT_GCN",
    "embedding": "EmbeddingSmile"
}

MAP_NODE_EMBEDDING_MODEL = {
    "gcn": "NodeGCNNet", 
    "gat": "NodeGATNet", 
    "ginconv": "NodeGINConvNet",
    "gat_gcn": "NodeGAT_GCN",    
}

MAP_PROTEIN_EMBEDDING_MODEL = {
    "embedding": "EmbeddingProtein", 
    "gcn": "ProteinGCNNet",
    "embeddingdeep": "EmbeddingProteinDeep"
}

MAP_MODEL = {
    "graphdta": "GraphDTA",
    "sgdta": "SGDTA",
    "deepdta": "DeepDTA",
    "dgraphdta": "DGraphDTA"
}

class Config:
    def __init__(self, args, device):
        self.config = {}
        self.load_config(args)
        self.device = device

    def load_config(self, args):
        self.config["datasets"] = {}
        self.config["datasets"]["dataset"] = args.dataset
        self.config["datasets"]["root"] = args.root_data_save
        self.config["datasets"]["kfold"] = KFOLD
        self.config["datasets"]["type_data"] = args.type_data
        if self.config["datasets"]["dataset"] == "davis":
            self.config["num_node"] = 510
        else:
            self.config["num_node"] = 2340

        self.config["model"] = {}
        self.config["model"]["graph_embedding"] = MAP_DRUG_MODEL[args.graph_embedding]
        self.config["model"]["node_embedding"] = MAP_NODE_EMBEDDING_MODEL[args.node_embedding]
        self.config["model"]["protein"] = MAP_PROTEIN_EMBEDDING_MODEL[args.protein_model]
        self.config["model"]["model"] = MAP_MODEL[args.model]

        self.config["optimizer"] = {}
        self.config["optimizer"]["name"] = "Adam"
        self.config["optimizer"]["parameters"] = {}
        self.config["optimizer"]["parameters"]["lr"] = 5e-4

        self.config["loss"] = 'MSELoss'

        self.config["optimizer"]["parameters"]["weight_decay"] = 1e-5
        
        self.config['lr_scheduler'] = {}
        self.config['lr_scheduler']['name'] = "CosineAnnealingWarmRestarts"
        self.config["lr_scheduler"]["parameters"] = {'T_0': 10}

        self.config["exp_name"] = args.exp_name

    def get_dataset(self, **kwargs):
        return Datasets(**kwargs, **self.config["datasets"])

    def get_graph_embedding(self, **kwargs):
        name = self.config["model"]["graph_embedding"]
        parameter_model = {}
        if self.config["datasets"]["dataset"] == 'davis' and name == 'EmbeddingSmile':
            parameter_model['in_channels'] = 85
        elif self.config["datasets"]["dataset"] == 'kiba' and name == 'EmbeddingSmile':
            parameter_model['in_channels'] = 100
        return getattr(graph_embedding, name)(**parameter_model,**kwargs)

    def get_node_embedding(self, **kwargs):
        name = self.config["model"]["node_embedding"]
        return getattr(node_embedding, name)(**kwargs)

    def get_protein_model(self, **kwargs):
        name = self.config["model"]["protein"]
        parameter_model = {}
        if self.config["datasets"]["dataset"] == 'davis' and name == 'EmbeddingProteinDeep':
            parameter_model['in_channels'] = 1200
        elif self.config["datasets"]["dataset"] == 'kiba' and name == 'EmbeddingProteinDeep':
            parameter_model['in_channels'] = 1000

        return getattr(models_protein, name)(**parameter_model,**kwargs)

    def get_model(self, **kwargs):
        name = self.config["model"]["model"]
        parameter_model = {"model_drug": self.get_graph_embedding()}        
        parameter_model["embedding_protein"] = self.get_protein_model()

        if name == "SGDTA":
            parameter_model['device'] = self.device
            parameter_model['node_embedding'] = self.get_node_embedding()                        
            parameter_model['num_node'] = self.config["num_node"]

        return getattr(models, name)(**parameter_model, **kwargs)

    def get_optimizer(self, model_parameters):
        return getattr(torch.optim, self.config['optimizer']['name'])(model_parameters,
                                                                      **self.config['optimizer']['parameters'])

    def get_loss_function(self, **kwargs):
        return getattr(torch.nn, self.config['loss'])(**kwargs)  

    def get_lr_scheduler(self, optimizer):
        return getattr(torch.optim.lr_scheduler,
                       self.config['lr_scheduler']['name'])(optimizer, **self.config['lr_scheduler']['parameters'])
                       
    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config