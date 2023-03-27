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
        num=${dir##*/}
        mkdir -p $ncnnModelDir/$num
        onnx2ncnn $dir/$fileName.onnx $ncnnModelDir/$num/$fileName.param $ncnnModelDir/$num/$fileName.bin
        cp $dir/$fileName-params.json $ncnnModelDir/$num/$fileName-params.json
        # rm $ncnnModelDir/$num/$fileName.bin
    fi
done
