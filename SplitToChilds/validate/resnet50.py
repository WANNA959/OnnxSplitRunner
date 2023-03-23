from SplitToChilds import runmodule
import numpy as np
from config import Config
import time

default_batchsize = 15


def Print(result: dict):
    for k, v in result.items():
        print(k, ":", v[0][:10])
    print()


driver = ['CUDAExecutionProvider']
test_count = 1

if __name__ == "__main__":
    print("run validate:")
    model_name = "resnet50"
    model_params = Config.LoadModelParamsDictById(model_name)

    print("\n==>start to validate model:", model_name)

    input_dict = {}
    for value in model_params["input"]["data"]:
        shape = [v if v >= 0 else default_batchsize for v in value["shape"]]
        input_dict[value["name"]] = np.array(
            np.random.randn(*shape), dtype=value["type"])

    print("raw")
    start = time.time()
    for _ in range(test_count):
        output = runmodule.RunWholeOnnxModel(model_name, input_dict, driver)
    print((time.time()-start)/test_count)     # 0.365s 1105.875 MB
    Print(output)

    print("child")
    start = time.time()
    for _ in range(test_count):
        output = runmodule.RunChildOnnxModelSequentially(
            model_name, input_dict, driver)
    print((time.time()-start)/test_count)       # 1.215s/2.7s 856.875 MB
    Print(output)

'''
==>start to validate model: resnet50
raw
0.9638564586639404
output : [-0.9327309   0.18960615 -0.11414948 -0.785846    0.3101012  -1.3566552
  0.01524135 -0.6205009   0.16976798 -1.4140887 ]

child
1496.941861629486
output : [-0.93273175  0.18960646 -0.114149   -0.78584605  0.3101013  -1.3566563
  0.01524213 -0.6205      0.16976905 -1.4140893 ]
'''