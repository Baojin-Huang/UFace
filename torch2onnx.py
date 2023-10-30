import numpy as np
import onnx
import torch
import zipfile
from torchkit.backbone import get_model


def convert_onnx(net, path_module, output, opset=11):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    weight = torch.load(path_module,map_location='cpu')
    net.load_state_dict(weight)
    net.eval()
    torch.onnx.export(net, img, output, keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
    model = onnx.load(output)
    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    onnx.save(model, output)

data_dir = 'ms1m+mask1.0_cur_UNet5+ULoss'
backbone_name = 'IR_50'
backbone_model = get_model(backbone_name)
backbone = backbone_model([112,112])
for ii in range(19,30):
    pth_path = 'Backbone_Epoch_'+ str(ii) + '_checkpoint'
    f = zipfile.ZipFile('/sd/zips/' + pth_path + '_' + data_dir + '.zip','w',zipfile.ZIP_DEFLATED)
    
    print(pth_path," success!")
    convert_onnx(backbone, '/sd/ckpt/' + data_dir + '/' + pth_path +'.pth','/sd/onnx/' + pth_path + '_' + data_dir + '.onnx')
    f.write('/sd/onnx/' + pth_path + '_' + data_dir + '.onnx')
    f.close()
