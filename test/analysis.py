import os,sys
import argparse
import numpy as np
from tqdm import tqdm
import torch
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from torchkit.backbone import get_model
from torchkit.util.utils import l2_norm
import cv2
import matplotlib.pyplot as plt

features = []
def hook(module, input, output):
    features.append(input)
    return None

def g1(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    input_size = [112, 112]
    net = get_model(args.network)(input_size)
    net.load_state_dict(torch.load(weight))
    net.eval()

    

    # # 确定想要提取出的中间层名字
    # for (name, module) in net.named_modules():
    #     if name == 'body1.7':
    #         module.register_forward_hook(hook)
    
    feat = net(img)
    all_len = torch.norm(feat[:,:128], p=2, dim=1).sum()  
    up_len = torch.norm(feat[:,128:128+256], p=2, dim=1).sum()
    down_len = torch.norm(feat[:,128+256:512], p=2, dim=1).sum()  
    # print(feat)
    print(all_len,up_len,down_len)

    # fig, ax = plt.subplots(figsize=(10, 7))
    # ax.bar(x=np.arange(1,513), height=abs(feat[0]))

    # fig.savefig("normal.png")

    # print(feat_in.numpy().shape)
    # print(np.transpose(features[0][0][0][0:3], (1, 2, 0)).shape)
    # cv2.imwrite('12.jpg',g1(np.transpose(features[0][0][0][3:6], (1, 2, 0)).numpy())*255)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='IR_50', help='backbone network')
    parser.add_argument('--weight', type=str, default='/sd/ckpt/webface+mask_arcface_UNet5+ULoss11/Backbone_Epoch_25_checkpoint.pth')
    parser.add_argument('--img', type=str, default='data/11.jpg')
    args = parser.parse_args()
    inference(args.weight, args.network, args.img)