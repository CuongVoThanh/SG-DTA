import logging
import argparse
import torch

from lib.experiment import Experiment
from lib.config import Config
from lib.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description="Stacked Graph Drug Target Interaction")
    parser.add_argument("mode", choices=["train", "test"], help="Train or test")
    parser.add_argument("--dataset", choices=["davis", "drugchem", "kiba"], help="Choose dataset", required=True)
    parser.add_argument("--drug_embedding", choices=["gcn", "gat", "ginconv", "gat_gcn",'embedding'], help="Choose drug_embedding for drug", required=True)    
    parser.add_argument("--protein_embedding", default="embedding", choices=["embedding","gcn",'embeddingdeep'], help="Choose embedding for protein target")
    parser.add_argument("--model", default="graphdta", choices=["graphdta", "sgdta",'deepdta','dgraphdta'], help="Choose model", required=True)
    parser.add_argument("--network_embedding", default="gcn", choices=["gcn", "gat", "ginconv", "gat_gcn"], help="Choose embedding for drug-target network")
    parser.add_argument("--type_data", default="dataDTA", choices=["dataDTA","dgraphDTA"], help="Choose model for dataset", required=True)
    
    parser.add_argument("--exp_name", help="Experiment name", required=True)    

    parser.add_argument("--train_fold", type=int, default=0, choices=range(1,7), help="Fold to trainning (6: full-fold, 1-5: k-fold, 0: train k-fold)")
    
    parser.add_argument("--resume_fold", type=int, default=0, choices=range(1,7), help="Fold to resume (6: full-fold, 1-5: k-fold, 0: non resume training) (Default: 0)")
    parser.add_argument("--root_data_save", type=str, default="./data", help="Root for saving processed data (Default: ./data)")
    
    parser.add_argument("--train_epochs", type=int, default=1000, help="Epochs to train the model (Default: 1000)")
    parser.add_argument("--train_batch", type=int, default=512, help="Traning batch size (Default: 512)")
    parser.add_argument("--test_batch", type=int, default=512, help="Testing batch size (Default: 512)")
    parser.add_argument("--val_on_epoch", type=int, default=100, help="Validation on epoch (Default: 100)")
    
    parser.add_argument("--test_on_fold", type=int, default=1, choices=range(1,7), help="Test on fold (6: full-fold, 1-5: k-fold) (Default: 1)")

    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--deterministic",
                        action="store_true",
                        help="set cudnn.deterministic = True and cudnn.benchmark = False")

    args = parser.parse_args()
    if args.resume_fold != 0  and args.mode == "test":
        raise Exception("args.resume_fold = 0 is set on `test` mode: can't resume testing")
    if args.train_fold != 0 and args.resume_fold != 0:
        raise Exception("Please choices one mode: resume or train_on_fold")

    return args

def main():
    args = parse_args()
    device = torch.device('cpu') if not torch.cuda.is_available() or args.cpu else torch.device('cuda')
    
    exp = Experiment(args.exp_name, mode=args.mode, model_checkpoint_interval=args.val_on_epoch)
    cfg = Config(args, device)
    runner = Runner(cfg, exp, device, resume=args.resume_fold, epochs=args.train_epochs, val_on_epoch=args.val_on_epoch,
                        train_batch_size=args.train_batch, test_batch_size=args.test_batch, deterministic=args.deterministic)
    
    if args.mode == 'train':
        try:
            if args.train_fold !=0 :
                runner.train(args.train_fold) # mode 6: train with full-fold
            elif args.resume_fold != 0:
                runner.train(args.resume_fold)
            else:
                runner.train_per_kfold()    
        except KeyboardInterrupt:
            logging.info('Training interrupted.')
    else:
        runner.eval(args.test_on_fold, epoch=exp.get_last_checkpoint_epoch())
    return

if __name__ == "__main__":
    main()