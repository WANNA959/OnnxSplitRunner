import numpy as np
import cv2
import os
import sys
import json
from rknn.api import RKNN
import shutil

model_name = "resnet50"
current_path = sys.path[0]


def get_input_list(jsonfile):
    input_list = list()
    input_size_list = list()
    output_list = list()
    with open(jsonfile, 'r', encoding='utf-8') as fd:
        info = json.load(fd)
        data = info['input']['data']
        input_list = [item['name'] for item in data]
        input_size_list = [item['shape'] for item in data]
        output_list = [item['name'] for item in info['output']['data']]
    return input_list, input_size_list, output_list


def onnx2rknn(onnx_model, input_list, input_size_list, output_list, rknn_model):
    # Create RKNN object
    rknn = RKNN(verbose=True)
    # pre-process config
    print('--> config model')
    # rknn.config(mean_values=[123.675, 116.28, 103.53],
    #             std_values=[58.82, 58.82, 58.82], target_platform="rk3588")
    rknn.config(target_platform="rk3588")
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=onnx_model, inputs=input_list,
                         input_size_list=input_size_list, outputs=output_list)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    # ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(rknn_model)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # # Set inputs
    # img = cv2.imread('./dog_224x224.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # # Init runtime environment
    # print('--> Init runtime environment')
    # ret = rknn.init_runtime()
    # if ret != 0:
    #     print('Init runtime environment failed!')
    #     exit(ret)
    # print('done')

    # # Inference
    # print('--> Running model')
    # outputs = rknn.inference(inputs=[img])
    # x = outputs[0]
    # output = np.exp(x)/np.sum(np.exp(x))
    # outputs = [output]
    # print('done')

    rknn.release()


def trans_batch(onnxDir, rknnDir):
    dirs = os.listdir(onnxDir)
    for dir in dirs:
        dirpath = os.path.join(onnxDir, dir)
        if os.path.isfile(dirpath):
            continue
        tans_single(dirpath, rknnDir)


def tans_single(dirpath, rknnDir):
    files = [item for item in os.listdir(dirpath) if "json" in item]
    jsonfile = os.path.join(dirpath, files[0])
    files = [item for item in os.listdir(dirpath) if "onnx" in item]
    onnxfile = os.path.join(dirpath, files[0])
    input_list, input_size_list, output_list = get_input_list(jsonfile)
    print(input_list)
    print(input_size_list)
    print(output_list)
    rknnfile = os.path.join(rknnDir, files[0].replace(".onnx", ".rknn"))
    print(onnxfile)
    print(rknnfile)
    onnx2rknn(onnxfile, input_list, input_size_list, output_list, rknnfile)


if __name__ == '__main__':

    onnxDir = os.path.join(current_path, "../../Onnxs", model_name, "childs")
    print(onnxDir)

    rknnDir = os.path.join(current_path, "models")
    if os.path.exists(rknnDir):
        shutil.rmtree(rknnDir)
    os.makedirs(rknnDir)

    # trans_batch(onnxDir, rknnDir)
    tans_single(
        "/root/py_project/OnnxSplitRunner/Onnxs/resnet50/childs/0", rknnDir)
    # tans_single(
    #     "/root/py_project/OnnxSplitRunner/Onnxs/resnet50/childs/38", rknnDir)
