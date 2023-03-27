from rknn.api import RKNN
import os
import json
import numpy as np


def gen_random_inputs(jsonpath: str) -> list():
    with open(jsonpath, "r", encoding="utf-8") as f:
        content = json.load(f)
        shapes = [item['shape'] for item in content['input']['data']]
        print(shapes)
        inputs = list()
        for shape in shapes:
            shape[1], shape[3] = shape[3], shape[1]
            input = np.random.rand(*shape)
            input = input.astype(np.float32)
            inputs.append(input)
            print(input.dtype)
            print(input.shape)
        return inputs


def run_batch_rknn(rknnDir: str):
    dirs = os.listdir(rknnDir)
    dirs = [dir for dir in dirs if os.path.isdir(os.path.join(rknnDir, dir))]
    for idx in range(150, len(dirs)):
        subRknnDir = os.path.join(rknnDir, str(idx))
        run_single_rknn(subRknnDir)


def run_single_rknn(rknnDir: str):

    print("start validate {}".format(rknnDir))
    files = [item for item in os.listdir(rknnDir) if "json" in item]
    jsonfile = os.path.join(rknnDir, files[0])
    files = [item for item in os.listdir(rknnDir) if "rknn" in item]
    rknnfile = os.path.join(rknnDir, files[0])

    inputs = gen_random_inputs(jsonfile)
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Load model
    print('--> Loading model')
    ret = rknn.load_rknn(path=rknnfile)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target="RK3588")
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=inputs)
    x = outputs[0]
    output = np.exp(x)/np.sum(np.exp(x))
    outputs = [output]
    print('done')

    rknn.release()


if __name__ == '__main__':
    # 147 149 150
    # run_single_rknn(
    # "/root/py_project/OnnxSplitRunner/atransform/rknn/models/resnet50/147")

    run_batch_rknn(
        "/root/py_project/OnnxSplitRunner/atransform/rknn/models/resnet50")
