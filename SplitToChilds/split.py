from SplitToChilds.moduleOperate import ModelAnalyzer


def SplitModel(model_name, onnx_path=None):
    modelAnalyzer = ModelAnalyzer(model_name, onnx_path)
    # nodes = modelAnalyzer.GetConvergeNodes()
    # nodes = [node.name for node in nodes]
    modelAnalyzer.BuildDependencies()
    # node = modelAnalyzer.node_dict["Add_14"]
    # modelAnalyzer.checkConverageNode(node)
    converageNodes = modelAnalyzer.GetConverageNodes()
    converageNodes = [node.name for node in converageNodes]
    print(converageNodes)
    # modelAnalyzer.SplitAndStoreChilds(modelAnalyzer.GetAllNodes())


if __name__ == "__main__":
    model_name = "resnet50"
    SplitModel(model_name)
