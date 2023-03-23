#!/bin/bash

currentDir=$(cd $(dirname $0); pwd)
modelName=resnet50
childrenPath=$currentDir/../../Onnxs/$modelName/childs

ncnnModelDir=$currentDir/models/$modelName

rm -rf $ncnnModelDir
mkdir -p $ncnnModelDir

for dir in ${childrenPath}/*
do
    if [ -d $dir ];then
        fileName=`ls $dir | grep onnx | sed -e "s/.onnx//g"`
        onnx2ncnn $dir/$fileName.onnx $ncnnModelDir/$fileName.param $ncnnModelDir/$fileName.bin
        rm $ncnnModelDir/$fileName.bin
    fi
done
