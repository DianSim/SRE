
import sys
from preprocessing import Preprocessing
from network import MatchboxNet
from train import Train
from test import Test
from feature import FeatureMappings


if len(sys.argv) == 2 and sys.argv[1] == '-t': 
    train_bool = False
else:
    train_bool = True


prep = Preprocessing()
feature_instance = FeatureMappings(prep)

train_dataset, val_dataset, test_dataset = feature_instance.create_features()

network = MatchboxNet(B=3, R=2, C=64)

if train_bool:
    # runs Train.py
    train_instance = Train(network, train_dataset, val_dataset)  
    train_instance.train()  #trains the model
else:
    print('test.......')
    test_obj = Test(network, test_dataset)
    test_obj.test() #tests the model
