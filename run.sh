#!/bin/bash

for model in resnet20 
do
    echo "python3 -u trainer.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model"
    python3 -u trainer.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model
done