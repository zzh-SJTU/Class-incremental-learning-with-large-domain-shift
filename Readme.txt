本目录下为base实验即每次实验的数据集均相同的class incremental learning
如果要进行有不同数据集的实验需要先进入Dif_dataset目录根据该目录下Readme进行实验
终端运行以下命令即可：
python3 -u main_code/main.py --datasets mnist --num-tasks 5 --network resnet50 --nepochs 10 --batch-size 32 --approach lwf
approach 参数可以换成[lwf,ewc,finetuning]中的任意一种，其他参数也可以相应改变
进行icarl的实验，输入下列命令
python3 -u main_code/main.py --datasets mnist --num-tasks 5 --network resnet50 --nepochs 10 --batch-size 32 --approach icarl --num-exemplars 2000 --exemplar-selection herding
每次实验之后，均会在本目录下产生fog.pdf和heat_map_show.pdf连个文件，可以清晰地显示训练完每个任务之后地准确率和遗忘程度
实验具体设置，完整结果和相关参考，见补充材料