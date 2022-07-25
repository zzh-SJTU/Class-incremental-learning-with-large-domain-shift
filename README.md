# Class-incremental-learning-with-large-domain-shift
Exploring class-incremental learning with large domain shift for image classification.  
Run the following command to reproduce the same results on the report

    python3 -u main_code/main.py --datasets mnist --num-tasks 5 --network resnet50 --nepochs 10 --batch-size 32 --approach lwf
