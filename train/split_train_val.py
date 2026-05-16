# coding:utf-8

import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xml_path', default='Annotation', type=str, help='input xml label path')
parser.add_argument('--txt_path', default='ImageSets/Main', type=str, help='output txt label path')
opt = parser.parse_args()

# Total split ratios: 70% train, 20% val, 10% test
test_percent = 0.1  # 10% for test
val_percent = 0.2   # 20% for validation (of remaining after test)
# The remaining 70% will be for training

xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)
list_index = range(num)

# First split: 90% (train+val) vs 10% (test)
tv_num = int(num * (1 - test_percent))  # train + val count
test_num = num - tv_num

# Then split the 90% into train and val (7:2 ratio of total, which is ~77.8:22.2 of the 90%)
train_num = int(tv_num * (7/9))  # 7/9 of 90% = 70% of total
val_num = tv_num - train_num     # remaining 20% of total

# Randomly sample the indices
trainval_test = random.sample(list_index, tv_num)  # first get train+val (90%)
train = random.sample(trainval_test, train_num)    # then split that into train (70%)
# The remaining in trainval_test not in train are val (20%)

test = [i for i in list_index if i not in trainval_test]  # test is the remaining 10%

file_trainval = open(txtsavepath + '/trainval.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')

for i in list_index:
    name = total_xml[i][:-4] + '\n'
    if i in trainval_test:  # train or val
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:  # test
        file_test.write(name)

file_trainval.close()
file_train.close()
file_val.close()
file_test.close()