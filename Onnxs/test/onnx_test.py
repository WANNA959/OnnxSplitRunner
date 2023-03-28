import torch
import onnx
import onnx.utils


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.convs1 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3))
        self.convs2 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3))
        self.convs3 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3))
        self.convs4 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3),
                                          torch.nn.Conv2d(3, 3, 3))

    def forward(self, x):
        x = self.convs1(x)
        x1 = self.convs2(x)
        x2 = self.convs3(x)
        x = x1 + x2
        x = self.convs4(x)
        return x


def splitWhole():
    model = Model()
    input = torch.randn(1, 3, 20, 20)

    # PyTorch 自动生成输入和输出的张量序号
    torch.onnx.export(
        model, input, "/root/py_project/OnnxSplitRunner/whole_model.onnx")

    onnx.utils.extract_model(
        "/root/py_project/OnnxSplitRunner/whole_model.onnx", "/root/py_project/OnnxSplitRunner/partial_model.onnx", ['23'], ['24'])

    model = onnx.load("/root/py_project/OnnxSplitRunner/partial_model.onnx")
    print(model.graph.input)
    print(model.graph.output)


if __name__ == '__main__':
    onnx.utils.extract_model(
        "/root/py_project/OnnxSplitRunner/Onnxs/resnet50/resnet50.onnx", "/root/py_project/OnnxSplitRunner/test-manual.onnx", ['459'], ['471'])
