import argparse
import logging
import random
import os
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_scatter import scatter
from datasets_abide.abideDataset_hypergraph import ABIDEDataset
from datasets_abide.tu_dataset import TUEvaluator
from unsupervised.embedding_evaluation import EmbeddingEvaluation
from unsupervised.encoder.tu_encoder_hypergraphGCN import TUEncoder
from unsupervised.learning.GInfoMinMax import GInfoMinMax
from unsupervised.view_learner import ViewLearner
from unsupervised.utils import set_tu_dataset_y_shape


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cuda >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.cuda}')
    else:
        device = torch.device('cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    setup_seed(args.seed)


    my_transforms = Compose([set_tu_dataset_y_shape]) 
    dataset = ABIDEDataset(args.path, args.name, transform=my_transforms)   
    dataset.data.y = dataset.data.y.squeeze()

    evaluator = TUEvaluator()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = GInfoMinMax(
        TUEncoder(num_dataset_features=3, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers,
                  drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
        args.emb_dim,
        ).to(device)
    
    
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    view_learner = ViewLearner(TUEncoder(num_dataset_features=3, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers,
                                         drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                               mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)


    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)

    if args.downstream_classifier == "linear":
        ee = EmbeddingEvaluation(
            MLPClassifier(random_state=1, max_iter=300),
            evaluator,
            dataset.task_type,
            dataset.num_tasks,
            device,
            save_dir=args.save_dir,  
            param_search=False
        )


    else:
        ee = EmbeddingEvaluation(SVC(), evaluator, dataset.task_type,
                                 dataset.num_tasks,
                                 device, 
                                 save_dir=args.save_dir, param_search=True)

    model.eval()
    if args.kf_cv:
        train_score, val_score = ee.kf_before_embedding_evaluation(model.encoder, dataset)
    elif args.train_ratio is not None:
        train_score, val_score = ee.split_before_embedding_evaluation(
            model.encoder, dataset, train_ratio=args.train_ratio, batch_size=args.batch_size, flag=True
        )
    else:
        raise ValueError("Please set --kf_cv or provide --train_ratio.")
    
    logging.info(
        "Before training Embedding Eval Scores: Train: {} Val: {}".format(train_score, val_score))
 

    model_losses = []

    view_losses = []
    view_regs = []

    valid_curve = []
    train_curve = []

    valid_std_curve = []
    train_std_curve = []

    valid_f1_curve = []
    train_f1_curve = []

    valid_f1_std_curve = []
    train_f1_std_curve = []

    valid_sen_curve = []
    train_sen_curve = []

    valid_sen_std_curve = []
    train_sen_std_curve = []

    valid_spe_curve = []
    train_spe_curve = []

    valid_spe_std_curve = []
    train_spe_std_curve = []

    train_pre_curve=[]
    valid_pre_curve=[]

    train_pre_std_curve=[]
    valid_pre_std_curve=[]

    train_auc_curve=[]
    valid_auc_curve=[]

    train_auc_std_curve=[]
    valid_auc_std_curve=[]

    for epoch in range(1, args.epochs + 1):
        model_loss_all = 0
        view_loss_all = 0
        reg_all = 0
        for batch in dataloader:
            # set up
            batch = batch.to(device)
            #batch.edge_weight=383530；batch.edge_index=[2,383530]    116*32
            # train view to maximize contrastive loss
            view_learner.train()
            view_learner.zero_grad()
            model.eval()
        
            x, _ = model(batch.batch, batch.x, batch.edge_index, None, batch.edge_weight)
            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None, batch.edge_weight)
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems  116*32
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / args.temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()#[383530]

            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight)
            # regularization
            row, col = batch.edge_index #[2,38350] [[0,0,,,,,3711 ],[0,0,,,,,3711 ]
            edge_batch = batch.batch[row]#[38350]【0，1，2，...31]  
            edge_drop_out_prob = 1 - batch_aug_edge_weight  #[38350]
            uni, edge_batch_num = edge_batch.unique(return_counts=True)
            sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")  
            reg = []
            for b_id in range(args.batch_size):
                if b_id in uni:
                    num_edges = edge_batch_num[uni.tolist().index(b_id)]
                    reg.append(sum_pe[b_id] / num_edges)
                else:
                    # means no edges in that graph. So don't include.
                    pass
            num_graph_with_edges = len(reg)
            reg = torch.stack(reg)
            reg = reg.mean()


            view_loss = model.calc_loss(x, x_aug) - (args.reg_lambda * reg)
            view_loss_all += view_loss.item() * batch.num_graphs
            reg_all += reg.item()
            # gradient ascent formulation
            (-view_loss).backward()
            view_optimizer.step()

            # train (model) to minimize contrastive loss
            model.train()
            view_learner.eval()
            model.zero_grad()
            
            x, _ = model(batch.batch, batch.x, batch.edge_index, None, batch.edge_weight)
            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None, batch.edge_weight)

            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / args.temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()

            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight)

            model_loss = model.calc_loss(x, x_aug)
            model_loss_all += model_loss.item() * batch.num_graphs
            # standard gradient descent formulation
            model_loss.backward()
            model_optimizer.step()


        fin_model_loss = model_loss_all / len(dataloader)
        fin_view_loss = view_loss_all / len(dataloader)
        fin_reg = reg_all / len(dataloader)

        logging.info('Epoch {}, Model Loss {}, View Loss {}, Reg {}'.format(epoch, fin_model_loss, fin_view_loss, fin_reg))
        model_losses.append(fin_model_loss)
        view_losses.append(fin_view_loss)
        view_regs.append(fin_reg)

        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(args.save_dir, f'unsupervised_{args.name}_classifier_epoch{epoch}.pth')
        )
        torch.save(
            view_learner.state_dict(),
            os.path.join(args.save_dir, f'view_learner_{args.name}_classifier_epoch{epoch}.pth')
        )

        if epoch % args.eval_interval == 0:
            model.eval()
            if args.kf_cv:
                train_score, val_score = ee.kf_embedding_evaluation(
                    epoch, model.encoder, dataset, flag=True
                )
            elif args.train_ratio is not None:
                train_score, val_score = ee.split_embedding_evaluation(
                    epoch, model.encoder, dataset,
                    train_ratio=args.train_ratio,
                    batch_size=args.batch_size,
                    flag=True
                )


            logging.info(
                "Metric: {} Train_mean: {} Val_mean: {}".format(evaluator.eval_metric, train_score[0], val_score[0]
                                                                                        ))
            logging.info(
                "Metric: {} Train_std: {} Val_std: {}".format(evaluator.eval_metric, train_score[1], val_score[1]
                                                                                       ))
            logging.info(
                "Metric: f1 Train_mean: {} Val_mean: {}".format(train_score[2], val_score[2]))

            logging.info(
                "Metric: f1 Train_std: {} Val_std: {}".format(train_score[3], val_score[3]))

            logging.info(
                "Metric: sen Train_mean: {} Val_mean: {}".format(train_score[4], val_score[4]))

            logging.info(
                "Metric: sen Train_std: {} Val_std: {} ".format(train_score[5], val_score[5]))

            logging.info(
                "Metric: spe Train_mean: {} Val_mean: {}".format(train_score[6], val_score[6]))

            logging.info(
                "Metric: spe Train_std: {} Val_std: {}".format(train_score[7], val_score[7]))

            logging.info(
                "Metric: precision Train_mean: {} Val_mean: {}".format(train_score[8], val_score[8]))
            
            logging.info(
                "Metric: precision Train_std: {} Val_std: {}".format(train_score[9], val_score[9]))

            logging.info(
                "Metric: auc Train_mean: {} Val_mean: {}".format(train_score[10], val_score[10]))

            logging.info(
                "Metric: auc Train_std: {} Val_std: {}".format(train_score[11], val_score[11]))

            train_f1_curve.append(train_score[2])
            valid_f1_curve.append(val_score[2])

            train_f1_std_curve.append(train_score[3])
            valid_f1_std_curve.append(val_score[3])

            train_sen_curve.append(train_score[4])
            valid_sen_curve.append(val_score[4])

            train_sen_std_curve.append(train_score[5])
            valid_sen_std_curve.append(val_score[5])

            train_spe_curve.append(train_score[6])
            valid_spe_curve.append(val_score[6])
            
            train_spe_std_curve.append(train_score[7])
            valid_spe_std_curve.append(val_score[7])
            
            train_pre_curve.append(train_score[8])
            valid_pre_curve.append(val_score[8])

            train_pre_std_curve.append(train_score[9])
            valid_pre_std_curve.append(val_score[9])
        
            train_auc_curve.append(train_score[10])
            valid_auc_curve.append(val_score[10])

            train_auc_std_curve.append(train_score[11])
            valid_auc_std_curve.append(val_score[11])    

            train_curve.append(train_score[0])
            valid_curve.append(val_score[0])

            train_std_curve.append(train_score[1])
            valid_std_curve.append(val_score[1])
  

    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)
    best_train_epoch = np.argmax(np.array(train_curve))
   
    best_f1_train_epoch = np.argmax(np.array(train_f1_curve))
    best_f1_valid_epoch = np.argmax(np.array(valid_f1_curve))

    best_sen_train_epoch = np.argmax(np.array(train_sen_curve))
    best_sen_valid_epoch = np.argmax(np.array(valid_sen_curve))

    best_spe_train_epoch = np.argmax(np.array(train_spe_curve))
    best_spe_valid_epoch = np.argmax(np.array(valid_spe_curve))

    best_pre_train_epoch = np.argmax(np.array(train_pre_curve))
    best_pre_valid_epoch = np.argmax(np.array(valid_pre_curve))

    best_auc_train_epoch = np.argmax(np.array(train_auc_curve))
    best_auc_valid_epoch = np.argmax(np.array(valid_auc_curve))

    logging.info('FinishedTraining!')
    logging.info('BestEpoch: {}'.format(best_val_epoch))
    logging.info('BestTrainScore: acc_mean: {} acc_std: {} f1_mean: {} f1_std: {} sen_mean: {} sen_std: {} spe_mean: {} spe_std: {} pre_mean: {} pre_std: {} auc_mean: {} auc_std: {}'.format(
                        best_train, train_std_curve[best_train_epoch],
                        train_f1_curve[best_f1_train_epoch], train_f1_std_curve[best_f1_train_epoch],
                        train_sen_curve[best_sen_train_epoch], train_sen_std_curve[best_sen_train_epoch],
                        train_spe_curve[best_spe_train_epoch], train_spe_std_curve[best_spe_train_epoch],
                        train_pre_curve[best_pre_train_epoch], train_pre_std_curve[best_pre_train_epoch],
                        train_auc_curve[best_auc_train_epoch], train_auc_std_curve[best_auc_train_epoch]
                        ))
    logging.info('BestValidationScore: acc_mean: {} acc_std: {} f1_mean: {} f1_std: {} sen_mean: {} sen_std: {} spe_mean: {} spe_std: {} pre_mean: {} pre_std: {} auc_mean: {} auc_std: {}'.format(
                        valid_curve[best_val_epoch], valid_std_curve[best_val_epoch],
                        valid_f1_curve[best_f1_valid_epoch], valid_f1_std_curve[best_f1_valid_epoch],
                        valid_sen_curve[best_sen_valid_epoch], valid_sen_std_curve[best_sen_valid_epoch],
                        valid_spe_curve[best_spe_valid_epoch], valid_spe_std_curve[best_spe_valid_epoch],
                        valid_pre_curve[best_pre_valid_epoch], valid_pre_std_curve[best_pre_valid_epoch],
                        valid_auc_curve[best_auc_valid_epoch], valid_auc_std_curve[best_auc_valid_epoch]
                        ))

    return valid_curve[best_val_epoch]

def arg_parse():
    parser = argparse.ArgumentParser(description='A-GCL ABIDEI')
    
    parser.add_argument('--name', type=str, default='ABIDEI',
                        help='dataset.')
    parser.add_argument('--path', type=str, default='./data/ABIDEI/',
                        help='path of dataset.')
    parser.add_argument(
        '--cuda',
        type=int,
        default=0,
        help='CUDA device id (e.g., 0,1,2). Use -1 for CPU.'
    )
    parser.add_argument('--kf_cv', action='store_true', 
                        help='use cross validation')
    parser.add_argument('--train_ratio', type=float, default=None,
                        help='train_ratio.')
    parser.add_argument('--save_dir', type=str, default='./weights',
                    help='directory to save model weights')
    parser.add_argument('--template', type=int, default=116,
                        help='dataset template.')
    parser.add_argument('--model_lr', type=float, default=0.0005,
                        help='Model Learning rate.')
    parser.add_argument('--view_lr', type=float, default=0.0005,
                        help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=2,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.3,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Train Epochs')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature')
    parser.add_argument('--reg_lambda', type=float, default=2.0,
                        help='View Learner Edge Perturb Regularization Strength')
    parser.add_argument('--eval_interval', type=int, default=5, 
                        help="eval epochs interval")
    parser.add_argument('--downstream_classifier', type=str, default="linear",
                        help="Downstream classifier is linear or non-linear")
    parser.add_argument('--seed', type=int, default=123)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    run(args)