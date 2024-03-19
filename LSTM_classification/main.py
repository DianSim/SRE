import sys
from preprocessing import Preprocessing
from network import My_Model
from train import Train
from test import Test


if len(sys.argv) == 2 and sys.argv[1] == '-t': 
    train_bool = False
else:
    train_bool = True

prep = Preprocessing()
train_dataset, val_dataset, test_dataset = prep.create_iterators() 
network = My_Model()  

if train_bool:
    train_instance = Train(network, train_dataset, val_dataset)  
    train_instance.train()
else:
    print('test.......')
    print('TEST data evaluation')
    test_obj = Test(test_dataset)
    test_obj.test()
    print('VAL data evaluation')
    test_obj = Test(val_dataset)
    test_obj.test() 
    print('TRAIN data evaluation')
    test_obj = Test(train_dataset)
    test_obj.test()
