import netron

if __name__ == '__main__':
    modelPath = "/root/py_project/OnnxSplitRunner/Onnxs/resnet50/childs/4/resnet50-4.onnx"
    netron.start(modelPath)
