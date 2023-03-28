import torch
import onnx
import onnxruntime as rt
import numpy as np

if __name__ == '__main__':
    sess = rt.InferenceSession(
        "/root/py_project/OnnxSplitRunner/Onnxs/resnet50/childs/147/resnet50-147.onnx", None)

    out_names = [name for name in sess.get_outputs()[0].name]

    inputs = dict()
    # 打印输入节点的名字，以及输入节点的shape
    for i in range(len(sess.get_inputs())):
        print(sess.get_inputs()[i].name, sess.get_inputs()[i].shape)
        input = np.random.rand(
            *sess.get_inputs()[i].shape).astype(np.float32)
        print(input.dtype)
        inputs[sess.get_inputs()[i].name] = input

    print("----------------")
    # 打印输出节点的名字，以及输出节点的shape
    for i in range(len(sess.get_outputs())):
        print(sess.get_outputs()[i].name, sess.get_outputs()[i].shape)

    pred_onx = sess.run(None, inputs)
