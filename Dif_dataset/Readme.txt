该目录下为存在不同数据集的场景实验，如果要使用my_algorithm进行，则输入以下命令
python3 main_code/main.py --datasets mnist --num-tasks 5 --network resnet32 --seed 12342 --nepochs 10 --batch-size 128  --approach lwf --gpu 2
如想要进行learning without forgetting的实验，需要在/main_code/approach/lwf.py的最后一个函数中按照注释进行操作，然后同样运行上一行命令
每次实验之后，均会在本目录下产生fog.pdf和heat_map_show.pdf连个文件，可以清晰地显示训练完每个任务之后地准确率和遗忘程度