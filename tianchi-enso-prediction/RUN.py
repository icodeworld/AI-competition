import sys, os, time
import shutil
# generate data
if not os.path.isdir("data_48"):
    os.mkdir("data_48")
else:
    pass
os.system("python CMIP_LABEL.py")
os.system("python CMIP_TRAIN.py")
os.system("python SODA_LABEL.py")
os.system("python SODA_TRAIN.py")

# train
for cnn_dimmm in [35]:
    for in_month in [12]:
        for l1_rate in [0.001]:
            for aug in ['none']:
                for drop_rate in [0.4]:
                    os.system("python main.py --in_monthes " + str(in_month) + " --use_align --cnn_dim " + str(cnn_dimmm) +
                              " --lr1 " + str(l1_rate) + " --lr2 0.0003" + " --drop_rate " + str(drop_rate) +
                              " --augmentation " + aug)

# test
if os.path.isdir("result/"):
    shutil.rmtree("result/")
else:
    pass

os.system("python test_48.py --in_monthes " + str(in_month) + " --use_align --cnn_dim " + str(cnn_dimmm) + " --lr1 " +
          str(l1_rate) + " --lr2 0.0003" + " --drop_rate " + str(drop_rate) + " --seed 42 " + " --augmentation " + aug +
          " --soda")
# os.system(
#     "python test_48.py --in_monthes 12  --use_align --cnn_dim 30 --drop_rate 0.4 --lr1 0.001 --lr2 0.00009 --seed 42 --soda")
