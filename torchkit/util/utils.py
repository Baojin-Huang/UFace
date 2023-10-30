import os
import logging
import time
import yaml
import torch
import torch.distributed as dist


def l2_norm(input, axis=1):
    """ l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


@torch.no_grad()
def all_gather_tensor(input_tensor, dim=0):
    """ allgather tensor from all workers
    """
    world_size = dist.get_world_size()
    tensor_size = torch.tensor([input_tensor.shape[0]], dtype=torch.int64).cuda()
    tensor_size_list = [torch.ones_like(tensor_size) for _ in range(world_size)]
    dist.all_gather(tensor_list=tensor_size_list, tensor=tensor_size, async_op=False)

    max_size = torch.cat(tensor_size_list, dim=0).max()
    padded = torch.empty(max_size.item(), *input_tensor.shape[1:], dtype=input_tensor.dtype).cuda()
    padded[:input_tensor.shape[0]] = input_tensor
    padded_list = [torch.ones_like(padded) for _ in range(world_size)]
    dist.all_gather(tensor_list=padded_list, tensor=padded, async_op=False)

    slices = []
    for ts, t in zip(tensor_size_list, padded_list):
        slices.append(t[:ts.item()])
    return torch.cat(slices, dim=0)


def get_class_split(num_classes, num_gpus):
    """ split the num of classes by num of gpus
    """
    class_split = []
    for i in range(num_gpus):
        _class_num = num_classes // num_gpus
        if i < (num_classes % num_gpus):
            _class_num += 1
        class_split.append(_class_num)
    return class_split


def separate_irse_bn_paras(modules):
    """ sepeated bn params and wo-bn params
    """
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
    """ sepeated bn params and wo-bn params
    """
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, param in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(param)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id,
                              all_parameters))

    return paras_only_bn, paras_wo_bn


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    """ warm up learning rate when batch step below warmup steps
    """
    if batch % 500 == 0:
        print('current batch {} learning rate {}'.format(
            batch,
            batch * init_lr / num_batch_warm_up))
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up


def adjust_learning_rate(optimizer, epoch, cfg):
    """ Decay the learning rate based on schedule
    """
    lr = cfg['LR']
    for milestone in cfg['STAGES']:
        lr *= 0.1 if epoch >= milestone else 1.
    print('current epoch {} learning rate {}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """ Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """ Timer for count duration
    """
    def __init__(self):
        self.start_time = time.time()

    def get_duration(self):
        duration = time.time() - self.start_time
        self.start_time = time.time()
        return duration


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def accuracy_dist(cfg, outputs, labels, class_split, topk=(1,)):
    """ Computes the precision@k for the specified values of k in parallel
    """
    assert cfg['WORLD_SIZE'] == len(class_split), \
        "world size should equal to the number of class split"
    base = sum(class_split[:cfg['RANK']])
    maxk = max(topk)

    # add each gpu part max index by base
    scores, preds = outputs.topk(maxk, 1, True, True)
    preds += base

    batch_size = labels.size(0)

    # all_gather
    scores_gather = [torch.zeros_like(scores)
                     for _ in range(cfg['WORLD_SIZE'])]
    dist.all_gather(scores_gather, scores)
    preds_gather = [torch.zeros_like(preds) for _ in range(cfg['WORLD_SIZE'])]
    dist.all_gather(preds_gather, preds)
    # stack
    _scores = torch.cat(scores_gather, dim=1)
    _preds = torch.cat(preds_gather, dim=1)

    _, idx = _scores.topk(maxk, 1, True, True)
    pred = torch.gather(_preds, dim=1, index=idx)
    pred = pred.t()

    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def load_config(config_file):
    with open(config_file, 'r') as ifs:
        config = yaml.safe_load(ifs)
    return config


def load_pretrain_backbone(backbone, backbone_resume):
    """ load pretrain backbone checkpoint
    """
    if os.path.exists(backbone_resume) and os.path.isfile(backbone_resume):
        logging.info("Loading Backbone Checkpoint '{}'".format(backbone_resume))
        backbone.load_state_dict(torch.load(backbone_resume))
    else:
        logging.info(("No Backbone Found at '{}'"
                      "Please Have a Check or Continue to Train from Scratch").format(backbone_resume))


def load_pretrain_head(heads, head_names, head_resume, dist_fc=False, rank=0):
    """ load pretrain head checkpoint
    """
    if dist_fc:
        head_resume = '%s_Split_%d_checkpoint.pth' % (head_resume, rank)
    else:
        head_resume = head_resume
    if os.path.exists(head_resume) and os.path.isfile(head_resume):
        logging.info("Loading Head Checkpoint '{}'".format(head_resume))
        pretrain_heads = torch.load(head_resume)
        for index, head in enumerate(heads):
            head_name = head_names[index]
            pretrain_head = pretrain_heads[head_name]
            head.load_state_dict(pretrain_head)
    else:
        logging.info(("No Head Found at '{}'"
                      "Please Have a Check or Continue to Train from Scratch").format(head_resume))
