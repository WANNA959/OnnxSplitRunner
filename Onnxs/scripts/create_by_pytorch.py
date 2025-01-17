# create onnx-model file to $ProjectFold/Onnxs/$model_name/$model_name.onnx

from config import Config
import torch
import torch.onnx
import torchvision
import sys
from SplitToChilds.support import SupportedModels


print(sys.path)

mymodels = {
    "googlenet": torchvision.models.googlenet(pretrained=True),
    "resnet50": torchvision.models.resnet50(pretrained=True),
    # "vgg19": torchvision.models.vgg19(pretrained=True),
    # "squeezenetv1": torchvision.models.squeezenet1_0(pretrained=True),
}

for model_name in mymodels:
    if model_name in SupportedModels:
        model = mymodels[model_name]
        model.eval()

        torch.onnx._export(model, torch.rand(*SupportedModels[model_name]["input_shape"]), Config.ModelSavePathName(
            model_name), opset_version=12,  export_params=True, input_names=[SupportedModels[model_name]["input_name"]], output_names=["output"])
