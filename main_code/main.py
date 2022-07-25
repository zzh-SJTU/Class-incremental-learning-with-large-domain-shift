import os
import time
import torch
import argparse
import importlib
import numpy as np
from functools import reduce
import random
import approach
from datasets.data_loader import get_loaders
from datasets.dataset_config import dataset_config
from networks import tvmodels, allmodels, set_tvmodel_head_var
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib import pyplot
def plot_confusion(df_cm, save_dir='heat_map_show.pdf', labels=None,cmap='YlGnBu',name = 'Acc for each task'): 
    print(save_dir)
    fig = pyplot.figure(figsize = (15.5,15))
    sn.set(font_scale=2.0)
    mask = np.zeros_like(df_cm)
    mask[np.triu_indices_from(mask)] = True
    for i in range(mask.shape[0]):
        mask[i][i]=0
    with sn.axes_style("white"):
        hm=sn.heatmap(df_cm, annot=True,xticklabels=['(0,1)','(2,3)','(4,5)','(6,7)','(8,9)'],yticklabels=['(0,1)','(2,3)','(4,5)','(6,7)','(8,9)'],mask=mask,vmin=0, vmax=1, fmt=".2%", annot_kws={"size":28},cmap=cmap)
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=30, va="center")
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=30)
    hm.xaxis.set_label('task')
    hm.yaxis.set_label('Acc')
    fig.savefig(save_dir, format='pdf', dpi=300, bbox_inches = 'tight',pad_inches = 0)
    matplotlib.rc_file_defaults()

def main(argv=None):
    seed = 1998
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tstart = time.time()
    parser = argparse.ArgumentParser(description='Continual learning')
    parser.add_argument('--datasets', default=['mnist'], type=str, choices=list(dataset_config.keys()),
                         nargs='+', metavar="DATASET")
    parser.add_argument('--batch-size', default=64, type=int, required=False,)
    parser.add_argument('--num-tasks', default=4, type=int, required=False)
    parser.add_argument('--network', default='resnet18', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=0.1, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    args, canshu2 = parser.parse_known_args(argv)
    canshu1 = dict(nepochs=args.nepochs, lr=args.lr)
    # GPU选择
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = 'cuda'
    else:
        device = 'cpu'
    # 网络参数
    from networks.network import LLL_Net
    tvnet = getattr(importlib.import_module(name='torchvision.models'), args.network)
    init_model = tvnet(pretrained=False)
    set_tvmodel_head_var(init_model)
    # 方法选择
    from approach.incremental_learning import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    appr_args, extra_args = Appr.extra_parser(canshu2)
    # iCARL的中replay数据的相关参数
    from datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks,
                                                              args.batch_size)
    max_task = len(taskcla) 
    net = LLL_Net(init_model, remove_existing_head=False)
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    appr_kwargs = {**canshu1, **dict( **appr_args.__dict__)}
    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(transform, class_indices, **appr_exemplars_dataset_args.__dict__)
    appr = Appr(net, device, **appr_kwargs)
    print('数据类别如下：每个task用括号表示，括号里是该task包含的类别')
    print(taskcla)
    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))
    for t, (_, ncla) in enumerate(taskcla):
        if t >= max_task:
            continue
        net.add_head(taskcla[t][1])
        net.to(device)
        #训练
        appr.train(t, trn_loader[t], val_loader[t])
        print('#' * 110)
        #测试
        for u in range(t + 1):
            test_loss, acc_taw[t, u], acc_tag[t, u] = appr.eval(u, tst_loader[u])
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print('task {:2d}上的结果为 : loss={:.3f} '
                  '|  准确率={:5.1f}%, 遗忘率为={:5.1f}% '.format(u, test_loss,
                                                                 100 * acc_tag[t, u], 100 * forg_tag[t, u]))
    print('训练完毕，准确率如下')
    print(acc_tag)
    print('遗忘程度如下')
    print(forg_tag)
    #将准确率与遗忘程度以热度图的形式保存
    plot_confusion(acc_tag)
    plot_confusion(forg_tag,save_dir='fog.pdf',cmap='PuRd',name='forget')
    return acc_taw, acc_tag, forg_taw, forg_tag
if __name__ == '__main__':
    main()
