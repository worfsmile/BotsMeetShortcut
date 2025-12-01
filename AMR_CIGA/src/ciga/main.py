import argparse
import os
import os.path as osp
from copy import deepcopy
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from models.gnn_ib import GIB
from models.ciga import GNNERM, CIGA, GNNPooling
from models.losses import get_contrast_loss, get_irm_loss
from utils.logger import Logger
from utils.util import args_print, set_seed
from processed_data_for_CIGA.mydataloader import get_dataloader
from eval_model import eval_model

def main(args, record):
    print(args.dataset, args.text, args.feature, args.ood_type)
    file_path = f't-ai9/data/ood2/{args.dataset}/{args.text}/{args.feature}/{args.ood_type}/train_idx.pt'
    if not os.path.exists(file_path):
        record[args.dataset][args.text][args.feature][args.ood_type] = f'none'
        return record
    erm_model = None  # used to obtain pesudo labels for CNC sampling in contrastive loss
    args.seed = eval(str(args.seed))
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    def ce_loss(a, b, reduction='mean'):
        return F.cross_entropy(a, b, reduction=reduction)
    criterion = ce_loss
    eval_metric = 'acc' if len(args.eval_metric) == 0 else args.eval_metric
    edge_dim = -1.
    ### automatic dataloading and splitting

    train_loader, valid_loader, test_loader, input_dim, num_classes, edge_dim = get_dataloader(args)
    
    print("loading data done")
    evaluator = Evaluator('ogbg-molhiv')
    eval_metric = 'rocauc'
    # log
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_info = {
        'test_acc': [],
        'train_acc': [],
        'val_acc': [],
        'epochs' : []
    }
    experiment_name = f'{args.dataset}-{args.bias}_{args.ginv_opt}_erm{args.erm}_dir{args.dir}_coes{args.contrast}-{args.spu_coe}_seed{args.seed}_{datetime_now}'
    experiment_name = f'{datetime_now[4::]}'
    exp_dir = os.path.join('./logs/', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger.init_logger(filename=exp_dir + '/log.log')
    args_print(args, logger)
    logger.info(f"Using criterion {criterion}")
    logger.info(f"# Train: {len(train_loader.dataset)}  #Val: {len(valid_loader.dataset)} #Test: {len(test_loader.dataset)} ")
    best_weights = None
    for seed in args.seed:
        set_seed(seed)
        # models and optimizers
        model = CIGA(ratio=args.r,
                        input_dim=input_dim,
                        edge_dim=edge_dim,
                        out_dim=num_classes,
                        gnn_type=args.model,
                        num_layers=args.num_layers,
                        emb_dim=args.emb_dim,
                        drop_ratio=args.dropout,
                        graph_pooling=args.pooling,
                        virtual_node=args.virtual_node,
                        c_dim=args.classifier_emb_dim,
                        c_in=args.classifier_input_feat,
                        c_rep=args.contrast_rep,
                        c_pool=args.contrast_pooling,
                        s_rep=args.spurious_rep).to(device)
        model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        print(model)
                
        model_optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
        
        last_train_acc, last_test_acc, last_val_acc = 0, 0, 0
        cnt = 0
        epoch_info = {
            'train_acc': [],
            'test_acc': [],
            'val_acc': [],
        }
        # generate environment partitions
        if args.num_envs > 1:
            env_idx = (torch.sigmoid(torch.randn(len(train_loader.dataset))) > 0.5).long()
            print(f"num env 0: {sum(env_idx == 0)} num env 1: {sum(env_idx == 1)}")

            for epoch in range(args.epoch):
                if epoch == 45:
                    print()
                # for epoch in tqdm(range(args.epoch)):
                all_loss, n_bw = 0, 0
                all_losses = {}
                contrast_loss, all_contrast_loss = torch.zeros(1).to(device), 0.
                spu_pred_loss = torch.zeros(1).to(device)
                model.train()
                torch.autograd.set_detect_anomaly(True)
                num_batch = (len(train_loader.dataset) // args.batch_size) + int(
                    (len(train_loader.dataset) % args.batch_size) > 0)
                for step, graph in tqdm(enumerate(train_loader), total=num_batch, desc=f"Epoch [{epoch}] >>  ", disable=args.no_tqdm, ncols=60):
                    n_bw += 1
                    graph.to(device)
                    is_labeled = graph.y == graph.y
                    is_labeled = is_labeled.to('cpu')

                    if args.dir > 0:    #方向损失
                        # obtain dir losses
                        dir_loss, causal_pred, spu_pred, causal_rep = model.get_dir_loss(graph,
                                                                                        graph.y,
                                                                                        criterion,
                                                                                        is_labeled=is_labeled,
                                                                                        return_data='rep')
                        spu_loss = criterion(spu_pred[is_labeled], graph.y[is_labeled])
                        pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled])
                        pred_loss = pred_loss + spu_loss + args.dir * (epoch ** 1.6) * dir_loss
                        all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                        all_losses['dir'] = (all_losses.get('dir', 0) * (n_bw - 1) + dir_loss.item()) / n_bw
                        all_losses['spu'] = (all_losses.get('spu', 0) * (n_bw - 1) + spu_loss.item()) / n_bw
                    elif args.ginv_opt.lower() == 'gib':
                        # obtain gib loss
                        pred_loss, causal_rep = model.get_ib_loss(graph, return_data="rep")
                        all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                    else:
                        # obtain ciga I(G_S;Y) losses   #################
                        if args.spu_coe > 0 and not args.erm:
                            if args.contrast_rep.lower() == "feat":
                                (causal_pred, spu_pred), causal_rep = model(graph, return_data="feat", return_spu=True)
                            else:
                                (causal_pred, spu_pred), causal_rep = model(graph, return_data="rep", return_spu=True)

                            spu_pred_loss = criterion(spu_pred[is_labeled], graph.y[is_labeled], reduction='none')
                            pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled], reduction='none')
                            assert spu_pred_loss.size() == pred_loss.size()
                            # hinge loss
                            spu_loss_weight = torch.zeros(spu_pred_loss.size()).to(device)
                            spu_loss_weight[spu_pred_loss > pred_loss] = 1.0
                            spu_pred_loss = spu_pred_loss.dot(spu_loss_weight) / (sum(spu_pred_loss > pred_loss) + 1e-6)
                            pred_loss = pred_loss.mean()
                            all_losses['spu'] = (all_losses.get('spu', 0) * (n_bw - 1) + spu_pred_loss.item()) / n_bw
                            all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                        else:
                            if args.contrast_rep.lower() == "feat":
                                causal_pred, causal_rep = model(graph, return_data="feat")
                            else:
                                causal_pred, causal_rep = model(graph, return_data="rep")
                            pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled])
                            all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                    contrast_loss = 0
                    contrast_coe = args.contrast

                    if args.contrast > 0:
                        # obtain contrast loss
                        if args.contrast_sampling.lower() in ['cnc', 'cncp']:
                            # cncp referes to only contrastig the positive examples in cnc
                            if erm_model == None:
                                model_path = os.path.join('erm_model', f'{args.dataset}{args.num_text}') + ".pt"
                                erm_model = GNNERM(input_dim=input_dim,
                                                edge_dim=edge_dim,
                                                out_dim=num_classes,
                                                gnn_type=args.model,
                                                num_layers=args.num_layers,
                                                emb_dim=args.emb_dim,
                                                drop_ratio=args.dropout,
                                                graph_pooling=args.pooling,
                                                virtual_node=args.virtual_node).to(device)
                                erm_model.load_state_dict(torch.load(model_path, map_location=device))
                                print("Loaded model from ", model_path)
                            # obtain the erm predictions to sampling pos/neg pairs in cnc
                            erm_model.eval()
                            with torch.no_grad():
                                erm_y_pred = erm_model(graph)
                            erm_y_pred = erm_y_pred.argmax(-1)
                        else:
                            erm_y_pred = None
                        contrast_loss = get_contrast_loss(causal_rep,
                                                        graph.y.view(-1),
                                                        norm=F.normalize if not args.not_norm else None,
                                                        contrast_t=args.contrast_t,
                                                        sampling=args.contrast_sampling,
                                                        y_pred=erm_y_pred)
                        all_losses['contrast'] = (all_losses.get('contrast', 0) * (n_bw - 1) + contrast_loss.item()) / n_bw
                        all_contrast_loss += contrast_loss.item()

                    if args.num_envs > 1:
                        # indicate invariant learning
                        batch_env_idx = env_idx[step * args.batch_size:step * args.batch_size + graph.y.size(0)]
                        if 'molhiv' in args.dataset.lower():
                            batch_env_idx = batch_env_idx.view(graph.y.shape)
                        causal_pred, labels, batch_env_idx = causal_pred[is_labeled], graph.y[is_labeled], batch_env_idx[
                            is_labeled]
                        if args.irm_opt.lower() == 'eiil':
                            dummy_w = torch.tensor(1.).to(device).requires_grad_()
                            loss = F.nll_loss(causal_pred * dummy_w, labels, reduction='none')
                            env_w = torch.randn(batch_env_idx.size(0)).cuda().requires_grad_()
                            optimizer = torch.optim.Adam([env_w], lr=1e-3)
                            for i in range(20):
                                # penalty for env a
                                lossa = (loss.squeeze() * env_w.sigmoid()).mean()
                                grada = torch.autograd.grad(lossa, [dummy_w], create_graph=True)[0]
                                penaltya = torch.sum(grada ** 2)
                                # penalty for env b
                                lossb = (loss.squeeze() * (1 - env_w.sigmoid())).mean()
                                gradb = torch.autograd.grad(lossb, [dummy_w], create_graph=True)[0]
                                penaltyb = torch.sum(gradb ** 2)
                                # negate
                                npenalty = -torch.stack([penaltya, penaltyb]).mean()
                                # step
                                optimizer.zero_grad()
                                npenalty.backward(retain_graph=True)
                                optimizer.step()
                            new_batch_env_idx = (env_w.sigmoid() > 0.5).long()
                            env_idx[step * args.batch_size:step * args.batch_size +
                                                        graph.y.size(0)][labels] = new_batch_env_idx.to(env_idx.device)
                            irm_loss = get_irm_loss(causal_pred, labels, new_batch_env_idx, criterion=criterion)
                        elif args.irm_opt.lower() == 'ib-irm':
                            ib_penalty = causal_rep.var(dim=0).mean()
                            irm_loss = get_irm_loss(causal_pred, labels, batch_env_idx,
                                                    criterion=criterion) + ib_penalty / args.irm_p
                            all_losses['ib'] = (all_losses.get('ib', 0) * (n_bw - 1) + ib_penalty.item()) / n_bw
                        elif args.irm_opt.lower() == 'vrex':
                            loss_0 = criterion(causal_pred[batch_env_idx == 0], labels[batch_env_idx == 0])
                            loss_1 = criterion(causal_pred[batch_env_idx == 1], labels[batch_env_idx == 1])
                            irm_loss = torch.var(torch.FloatTensor([loss_0, loss_1]).to(device))
                        else:
                            irm_loss = get_irm_loss(causal_pred, labels, batch_env_idx, criterion=criterion)
                        all_losses['irm'] = (all_losses.get('irm', 0) * (n_bw - 1) + irm_loss.item()) / n_bw
                        pred_loss += irm_loss * args.irm_p

                    # compile losses
                    batch_loss = pred_loss + contrast_coe * contrast_loss + args.spu_coe * spu_pred_loss
                    model_optimizer.zero_grad()
                    batch_loss.backward()
                    model_optimizer.step()
                    all_loss += batch_loss.item()

                all_contrast_loss /= n_bw
                all_loss /= n_bw

                model.eval()
                train_acc = eval_model(model, device, train_loader, evaluator, eval_metric=eval_metric)
                val_acc = eval_model(model, device, valid_loader, evaluator, eval_metric=eval_metric)
                test_acc = eval_model(model,
                                    device,
                                    test_loader,
                                    evaluator,
                                    eval_metric=eval_metric)
                epoch_info['train_acc'].append(train_acc)
                epoch_info['test_acc'].append(test_acc)
                epoch_info['val_acc'].append(val_acc)
                if val_acc <= last_val_acc:
                    cnt += epoch >= args.pretrain
                else:
                    cnt = (cnt + int(epoch >= args.pretrain)) if last_val_acc == 1.0 else 0
                    last_train_acc = train_acc
                    last_val_acc = val_acc
                    last_test_acc = test_acc

                    if args.save_model:
                        best_weights = deepcopy(model.state_dict())
                if epoch >= args.pretrain and cnt >= args.early_stopping:
                    logger.info("Early Stopping")
                    logger.info("+" * 50)
                    logger.info("Last: Test_ACC: {:.3f} Train_ACC:{:.3f} Val_ACC:{:.3f} ".format(
                        last_test_acc, last_train_acc, last_val_acc))
                    all_info['epochs'].append(epoch)
                    all_info['test_acc'].append(last_test_acc)
                    all_info['train_acc'].append(last_train_acc)
                    all_info['val_acc'].append(last_val_acc)
                    break

                print("      [{:3d}/{:d}]".format(epoch, args.epoch) +
                            "\n       train_ACC: {:.4f} / {:.4f}"
                            "\n       valid_ACC: {:.4f} / {:.4f}"
                            "\n       tests_ACC: {:.4f} / {:.4f}\n".format(
                                # train_acc, torch.tensor(epoch_info['train_acc']).max(),
                                # val_acc, torch.tensor(epoch_info['val_acc']).max(),
                                # test_acc, torch.tensor(epoch_info['test_acc']).max()))
                                train_acc, last_train_acc,
                                val_acc, last_val_acc,
                                test_acc, last_test_acc))
            logger.info("=" * 50)
            all_info['test_acc'].append(last_test_acc)
            all_info['train_acc'].append(last_train_acc)
            all_info['val_acc'].append(last_val_acc)
            all_info['epochs'].append(epoch)
            

    if len(all_info['epochs']) == 0:
        print("No training is done.")
        return 0
    
    print("epoch: {:.0f}\n \
        Test ACC:{:.4f}-+-{:.4f}\n \
        Train ACC:{:.4f}-+-{:.4f}\n \
        Val ACC:{:.4f}-+-{:.4f} ".format(
        sum(all_info['epochs']) / len(all_info['epochs']),
        torch.tensor(all_info['test_acc']).mean(),
        torch.tensor(all_info['test_acc']).std(),
        torch.tensor(all_info['train_acc']).mean(),
        torch.tensor(all_info['train_acc']).std(),
        torch.tensor(all_info['val_acc']).mean(),
        torch.tensor(all_info['val_acc']).std()))

    if args.save_model:
        print("Saving best weights..")
        model_path = os.path.join('erm_model', f'{args.dataset}{args.num_text}') + ".pt"
        os.makedirs('erm_model', exist_ok=True)
        for k, v in best_weights.items():
            best_weights[k] = v.cpu()
        torch.save(best_weights, model_path)
        print("Done..")

    record_acc = torch.tensor(all_info['test_acc']).mean().item()
    record_std = torch.tensor(all_info['test_acc']).std().item()
    record[args.dataset][args.text][args.feature][args.ood_type] = f'{record_acc}+-{record_std}'
    print("\n\n\n")
    torch.cuda.empty_cache()
    return record
