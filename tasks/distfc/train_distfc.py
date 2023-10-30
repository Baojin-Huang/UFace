import os,math
import numpy as np
import sys
import logging
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

from torchkit.util.utils import AverageMeter, Timer
from torchkit.util.utils import adjust_learning_rate, warm_up_lr
from torchkit.util.utils import accuracy_dist
from torchkit.util.distributed_functions import AllGather
from torchkit.loss import get_loss
from torchkit.task.base_task import BaseTask
from torchkit.util.utils import l2_norm

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')


class TrainTask(BaseTask):
    """ TrainTask in distfc mode, which means classifier shards into multi workers
    """
    def __init__(self, cfg_file):
        super(TrainTask, self).__init__(cfg_file)

    def _loop_step(self, train_loaders, backbone, heads, criterion, tl, opt,
                   scaler, epoch, class_splits):
        """ load_data --> extract feature --> calculate loss and apply grad --> summary
        """
        backbone.train()  # set to training mode
        for head in heads:
            head.train()

        batch_sizes = self.batch_sizes

        am_losses = [AverageMeter() for _ in batch_sizes]
        am_top1s = [AverageMeter() for _ in batch_sizes]
        am_top5s = [AverageMeter() for _ in batch_sizes]
        t = Timer()
        for batch, samples in enumerate(zip(*train_loaders)):
            global_batch = epoch * self.step_per_epoch + batch
            if global_batch <= self.warmup_step:
                warm_up_lr(global_batch, self.warmup_step, self.cfg['LR'], opt)
            if batch >= self.step_per_epoch:
                break

            inputs = torch.cat([x[0] for x in samples], dim=0)
            inputs = inputs.cuda(non_blocking=True)
            labels = torch.cat([x[1] for x in samples], dim=0)
            labels = labels.cuda(non_blocking=True)

            if self.cfg['AMP']:
                with amp.autocast():
                    features = backbone(inputs)
                features = features.float()
            else:
                features = backbone(inputs)

            # gather features
            _features_gather = [torch.zeros_like(features) for _ in range(self.world_size)]
            features_gather = AllGather(features, *_features_gather)
            features_gather = [torch.split(x, batch_sizes) for x in features_gather]
            all_features = []
            for i in range(len(batch_sizes)):
                all_features.append(torch.cat([x[i] for x in features_gather], dim=0).cuda())

            # gather labels
            labels_gather = [torch.zeros_like(labels) for _ in range(self.world_size)]
            dist.all_gather(labels_gather, labels)
            labels_gather = [torch.split(x, batch_sizes) for x in labels_gather]
            all_labels = []
            for i in range(len(batch_sizes)):
                all_labels.append(torch.cat([x[i] for x in labels_gather], dim=0).cuda())

            step_losses = []
            step_original_outputs = []
            tl_loss1 = 1*torch.ones(1).cuda()
            tl_loss2 = -1*torch.ones(1).cuda()
            for i in range(len(batch_sizes)):
                # print(all_labels[i])
                outputs, part_labels, original_outputs = heads[i](all_features[i], all_labels[i])
                # print(part_labels)
                index_masked = torch.where(part_labels == -1)[0]
                index_normal = torch.where(part_labels >-1)[0]


                # Loss1
                eps = 0.00001
                if index_masked.shape[0] > 0:
                    target_samples_mask = all_features[i][index_masked]
                    target_samples_up_mask = target_samples_mask[:,128:256+128]
                    up_len_mask = torch.norm(target_samples_up_mask, p=2, dim=1).mean()
                    target_samples_down_mask = target_samples_mask[:,256+128:512]
                    down_len_mask = torch.norm(target_samples_down_mask, p=2, dim=1).mean()
                    # target_samples_all_mask = target_samples_mask[:,:128]
                    # all_len_mask = torch.norm(target_samples_all_mask, p=2, dim=1).mean()
                    tl_loss1 = torch.exp((down_len_mask)/(up_len_mask+down_len_mask+eps))-((down_len_mask)/(up_len_mask+down_len_mask+eps))
                target_samples = all_features[i][index_normal]
                target_samples_up = target_samples[:,128:256+128]
                up_len = torch.norm(target_samples_up, p=2, dim=1).mean()
                target_samples_down = target_samples[:,256+128:512]
                down_len = torch.norm(target_samples_down, p=2, dim=1).mean()    
                # target_samples_all = target_samples[:,:128]
                # all_len = torch.norm(target_samples_all, p=2, dim=1).mean() 
                tl_loss2 = torch.log(down_len/(up_len+eps))-(down_len/(up_len+eps))
                
                step_original_outputs.append(original_outputs)
                loss = criterion(outputs, part_labels) * self.branch_weights[i]

                step_losses.append(loss+ 0.3*tl_loss1 - 0.2*tl_loss2)



            # print("masked loss:",tl_loss1.item(),"normal loss:",tl_loss2.item())
            index_normal_all = torch.where(all_labels[i] >=0)[0]
            total_loss = sum(step_losses)
            # compute gradient and do SGD step
            opt.zero_grad()
            # Automatic Mixed Precision setting
            if self.cfg['AMP']:
                scaler.scale(total_loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                total_loss.backward()
                opt.step()

            for i in range(len(batch_sizes)):
                # measure accuracy and record loss
                prec1, prec5 = accuracy_dist(self.cfg,
                                             step_original_outputs[i][index_normal_all].data,
                                             all_labels[i][index_normal_all],
                                             class_splits[i],
                                             topk=(1, 5))

                am_losses[i].update(step_losses[i].data.item(),
                                    all_features[i][index_normal_all].size(0))
                am_top1s[i].update(prec1.data.item(), all_features[i][index_normal_all].size(0))
                am_top5s[i].update(prec5.data.item(), all_features[i][index_normal_all].size(0))
                # wirte loss and acc to tensorboard
                summarys = {
                    'train/loss_%d' % i: am_losses[i].val,
                    'train/top1_%d' % i: am_top1s[i].val,
                    'train/top5_%d' % i: am_top5s[i].val
                }
                self._writer_summarys(summarys, batch, epoch)

            duration = t.get_duration()
            self._log_tensor(batch, epoch, duration, am_losses, am_top1s, am_top5s)

    def _prepare(self):
        """ common prepare task for training
        """
        train_loaders, class_nums = self._make_inputs()
        backbone, heads, class_splits = self._make_model(class_nums)
        self._load_pretrain_model(backbone, self.cfg['BACKBONE_RESUME'], heads, self.cfg['HEAD_RESUME'])
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[self.local_rank])
        loss = get_loss('DistCrossEntropy').cuda()
        opt = self._get_optimizer(backbone, heads)
        scaler = amp.GradScaler()
        self._load_meta(opt, scaler, self.cfg['META_RESUME'])
        return train_loaders, backbone, heads, class_splits, loss, opt, scaler

    def train(self):
        """ make_inputs --> make_model --> load_pretrain --> build DDP -->
            build optimizer --> loop step
        """
        train_loaders, backbone, heads, class_splits, loss, opt, scaler = self._prepare()
        tl = torch.nn.NLLLoss()
        self._create_writer()
        for epoch in range(self.start_epoch, self.epoch_num):
            adjust_learning_rate(opt, epoch, self.cfg)
            self._loop_step(train_loaders, backbone, heads, loss, tl, opt, scaler, epoch, class_splits)
            if epoch >14:
                self._save_ckpt(epoch, backbone, heads, opt, scaler)


def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    task = TrainTask(os.path.join(task_dir, 'train_config.yaml'))
    task.init_env()
    task.train()


if __name__ == '__main__':
    main()
