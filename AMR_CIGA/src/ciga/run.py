import argparse
from main import main


def run():
    parser = argparse.ArgumentParser(description='Causality Inspired Invariant Graph LeArning')
   
    parser.add_argument('--ood_type', default='00', type=str, help='00,01,10,11')
    parser.add_argument('--text', default='description', type=str, help='text type')
    parser.add_argument('--dataset', default='csi17', type=str)
     # training config
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--epoch', default=2, type=int, help='training iterations')
    
    parser.add_argument('--device', default=1, type=int, help='cuda device')
    parser.add_argument('--root', default='./t-ai9', type=str, help='directory for datasets.')
    parser.add_argument('--bias', default='0.33', type=str, help='select bias extend')
    parser.add_argument('--feature', type=str, default="sentiments", help='full feature or simple feature')

    parser.add_argument('--valid_ratio', default=0.1, type=int, help='val split ratio')
   
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for the predictor')
    parser.add_argument('--seed', nargs='?', default='[42, 31415926, 114514]', help='random seed')
    parser.add_argument('--pretrain', default=20, type=int, help='pretrain epoch before early stopping')

    # model config
    parser.add_argument('--emb_dim', default=64, type=int)
    parser.add_argument('--r', default=0.25, type=float, help='selected ratio')
    
    # emb dim of classifier
    parser.add_argument('-c_dim', '--classifier_emb_dim', default=32, type=int)
    # inputs of classifier
    # raw:  raw feat
    # feat: hidden feat from featurizer
    parser.add_argument('-c_in', '--classifier_input_feat', default='raw', type=str)
    parser.add_argument('--model', default='gin', type=str)
    parser.add_argument('--pooling', default='mean', type=str)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--early_stopping', default=50, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--virtual_node', action='store_true')
    parser.add_argument('--eval_metric',
                        default='',
                        type=str,
                        help='specify a particular eval metric, e.g., mat for MatthewsCoef')

    # Invariant Learning baselines config
    parser.add_argument('--num_envs', default=1, type=int, help='num of envs need to be partitioned')
    parser.add_argument('--irm_p', default=1, type=float, help='penalty weight')
    parser.add_argument('--irm_opt', default='irm', type=str, help='algorithms to use')

    # Invariant Graph Learning config
    parser.add_argument('--erm', action='store_true')  # whether to use normal GNN arch
    parser.add_argument('--ginv_opt', default='ginv', type=str)  # which interpretable GNN archs to use
    parser.add_argument('--dir', default=0, type=float)
    parser.add_argument('--contrast_t', default=1.0, type=float, help='temperature prameter in contrast loss')
    # strength of the contrastive reg, \alpha in the paper
    parser.add_argument('--contrast', default=4, type=float)
    parser.add_argument('--not_norm', action='store_true')  # whether not using normalization for the constrast loss
    parser.add_argument('-c_sam', '--contrast_sampling', default='mul', type=str)
    # contrasting summary from the classifier or featurizer
    # rep:  classifier rep
    # feat: featurizer rep
    # conv: featurizer rep + 1L GNNConv
    parser.add_argument('-c_rep', '--contrast_rep', default='rep', type=str)
    # pooling method for the last two options in c_rep
    parser.add_argument('-c_pool', '--contrast_pooling', default='add', type=str)
    # spurious rep for maximizing I(G_S;Y)
    # rep:  classifier rep
    # conv: featurizer rep + 1L GNNConv
    parser.add_argument('-s_rep', '--spurious_rep', default='rep', type=str)
    # strength of the hinge reg, \beta in the paper
    parser.add_argument('--spu_coe', default=0, type=float)
    # misc
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--commit', default='', type=str, help='experiment name')
    # parser.add_argument('--save_model', action='store_true')  # save pred to ./pred if not empty
    parser.add_argument('--save_model', default=False, type=bool)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = run()
    record = {}
    ood_types = ['00']
    rank = 8
    datasets = ["cresci-2017-data"]

    texts = ['description']
    features = ['values']
    for dataset in datasets:
        record[dataset] = {}
        for text in texts:
            record[dataset][text] = {}
            for feature in features:
                record[dataset][text][feature] = {}
                for ood_type in ood_types:
                    args.dataset = dataset
                    args.text = text
                    args.seed = [42]
                    args.epoch = 500
                    args.device = rank%8
                    args.feature = feature
                    args.ood_type = ood_type
                    args.num_envs = 2
                    args.early_stopping = 50

                    args.batch_size = 32    
                    record = main(args, record)
    

