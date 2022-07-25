import torch
from copy import deepcopy
from argparse import ArgumentParser
import itertools
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
import torch.nn.functional as F
import numpy as np

class Appr(Inc_Learning_Appr):
    # 我的方法
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                exemplars_dataset=None, lamb=1, T=2, alpha = 0.5, fi_sampling_type='max_pred',lamb2=5000,fi_num_samples=-1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train,
                                   exemplars_dataset)
        self.model_old = None
        self.model_old_2 = None
        self.model_old_origin = None
        self.model_list_history =[]
        self.lamb = lamb
        self.lamb2 = lamb2
        self.alpha = alpha
        self.last_epoch_feature = None
        self.sampling_type = fi_sampling_type
        self.num_samples = fi_num_samples
        self.last_layer_feature = None
        self.last_layer_feature_old = None
        self.T = T
        self.feature_list =[]
        feat_ext = self.model.model
        self.older_params = {n: p.clone().detach() for n, p in feat_ext.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters()
                       if p.requires_grad}

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        parser.add_argument('--lamb', default=1, type=float, required=False,)
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        parser.add_argument('--alpha', default=0.5, type=float, required=False,)
        parser.add_argument('--fi-sampling-type', default='max_pred', type=str, required=False,
                            choices=['true', 'max_pred', 'multinomial'],)
        parser.add_argument('--fi-num-samples', default=-1, type=int, required=False,)
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
    def compute_fisher_matrix_diag(self, trn_loader):
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.model.named_parameters()
                  if p.requires_grad}
        n_samples_batches = (self.num_samples // trn_loader.batch_size + 1) if self.num_samples > 0 \
            else (len(trn_loader.dataset) // trn_loader.batch_size)
        self.model.train()
        for images, targets in itertools.islice(trn_loader, n_samples_batches):
            outputs = self.model.forward(images.to(self.device))

            if self.sampling_type == 'true':
                preds = targets.to(self.device)
            elif self.sampling_type == 'max_pred':
                preds = torch.cat(outputs, dim=1).argmax(1).flatten()
            elif self.sampling_type == 'multinomial':
                probs = torch.nn.functional.softmax(torch.cat(outputs, dim=1), dim=1)
                preds = torch.multinomial(probs, len(targets)).flatten()

            loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), preds)
            self.optimizer.zero_grad()
            loss.backward()
            for n, p in self.model.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * len(targets)
        n_samples = n_samples_batches * trn_loader.batch_size
        fisher = {n: (p / n_samples) for n, p in fisher.items()}
        return fisher


    def train_loop(self, t, trn_loader, val_loader,val_loader_1 ):
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
        super().train_loop(t, trn_loader, val_loader,val_loader_1)
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        self.model_list_history.append(deepcopy(self.model))
        self.model_list_history[t].eval()
        self.model_list_history[t].freeze_all()
        self.older_params = {n: p.clone().detach() for n, p in self.model.model.named_parameters() if p.requires_grad}
        curr_fisher = self.compute_fisher_matrix_diag(trn_loader)
        for n in self.fisher.keys():
            if self.alpha == -1:
                alpha = (sum(self.model.task_cls[:t]) / sum(self.model.task_cls)).to(self.device)
                self.fisher[n] = alpha * self.fisher[n] + (1 - alpha) * curr_fisher[n]
            else:
                self.fisher[n] = (self.alpha * self.fisher[n] + (1 - self.alpha) * curr_fisher[n])
        '''
        if t == 0:
            self.model_old_origin = deepcopy(self.model)
            self.model_old_origin.eval()
            self.model_old_origin.freeze_all()
        if t >0:
            self.model_old_2 = deepcopy(self.model_old)
            self.model_old_2.eval()
            self.model_old_2.freeze_all()
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()
        '''

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        counter=0
        list_feature = []
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            '''
            targets_old = None
            targets_old_2 = None
            if t > 1:
                targets_old = self.model_list_history[t-2](images.to(self.device))
            if t == 4:
                targets_old_2 = self.model_list_history[0](images.to(self.device))
            '''
            output_list = []
            for i in range(t):
                output_list.append(self.model_list_history[i](images.to(self.device)))
            outputs = self.model(images.to(self.device))
            self.last_layer_feature = self.model.model.last_feature
            if self.last_epoch_feature == None:
                self.last_epoch_feature = self.last_layer_feature
            if (self.last_layer_feature.size(0)==128):
                list_feature.append(self.last_layer_feature)
            counter+=1
            loss = self.criterion(t, outputs, targets.to(self.device), output_list)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        feature_sum = sum(list_feature)
        average_feature = feature_sum/len(list_feature)
        self.last_epoch_feature = average_feature
        self.feature_list.append(average_feature)
        '''
        mean_feature = torch.mean(torch.tensor(list_feature))
        std_feature = np.std(list_feature)
        print(mean_feature)
        print(std_feature)
        '''

    def eval(self, t, val_loader, val_loader_1):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            if t%2 ==0:
                self.model.eval()
                for images, targets in val_loader:
                    output_list = []
                    for i in range(t):
                        output_list.append(self.model_list_history[i](images.to(self.device)))
                    outputs = self.model(images.to(self.device))
                    loss = self.criterion(t, outputs, targets.to(self.device), output_list)
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                    total_loss += loss.data.cpu().numpy().item() * len(targets)
                    total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                    total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                    total_num += len(targets)
            else:
                self.model.eval()
                for images, targets in val_loader_1:
                    output_list = []
                    for i in range(t):
                        output_list.append(self.model_list_history[i](images.to(self.device)))
                    outputs = self.model(images.to(self.device))
                    loss = self.criterion(t, outputs, targets.to(self.device), output_list)
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                    total_loss += loss.data.cpu().numpy().item() * len(targets)
                    total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                    total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                    total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, outputs, targets, output_history):
        loss = 0
        #复现lwf的结果时需要将214-219行的注释取消，同时将220-232注释掉。
        '''
        if t>0:
            for i in range(t):
                loss += self.lamb * self.cross_entropy(outputs[i],
                                                    output_history[t-1][i], exp=1.0 / self.T)
        '''
        if t>1:
            flag = t%2
            for i in range(t):
                if i%2 == flag:
                    loss += self.lamb * self.cross_entropy(outputs[i],
                                                    output_history[i][i], exp=1.0 / self.T)
        if t > 0 :
            loss_reg = 0
            for n, p in self.model.model.named_parameters():
                if n in self.fisher.keys():
                    loss_reg += torch.sum(self.fisher[n] * (p - self.older_params[n]).pow(2)) / 2
            loss += self.lamb2 * loss_reg
        
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
