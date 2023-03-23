from SplitToChilds.moduleOperate import ModelAnalyzer


def SplitModel(model_name, onnx_path=None):
    modelAnalyzer = ModelAnalyzer(model_name, onnx_path)
    modelAnalyzer.SplitAndStoreChilds(modelAnalyzer.GetAllNodes())


if __name__ == "__main__":
    model_name = "resnet50"
    SplitModel(model_name)
