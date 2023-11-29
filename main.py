import pickle
from core.engine import ClassifierTrainer
from torch.utils.data import DataLoader
import argparse
import os

os.makedirs('weights', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--model', 
    choices=["piece_selector", "move_prediction"], required=True, type=str)
parser.add_argument('--mvpred_index', type=int, choices=[1, 2, 3, 4, 5, 6])
parser.add_argument('--run_name', type=str, required=True)

args = parser.parse_args()

if args.model == 'piece_selector':
    X_train, X_test = 'data_pkl/X_train_30000.pkl', 'data_pkl/X_test_1000.pkl'
    y_train, y_test = 'data_pkl/y_train_30000.pkl', 'data_pkl/y_test_1000.pkl'
else:
    index = args.mvpred_index
    X_train, X_test = f'data_pkl/p{index}_X_30000.pkl', f'data_pkl/p{index}_test_X_1000.pkl'
    y_train, y_test = f'data_pkl/p{index}_y_30000.pkl', f'data_pkl/p{index}_test_y_1000.pkl'

print(X_train, X_test)
print(y_train, y_test)

with open(X_train, 'rb') as pkl_file:
    input_data = pickle.load(pkl_file)
with open(y_train, 'rb') as pkl_file:
    label_data = pickle.load(pkl_file)

train_data = list(zip(input_data, label_data))
train_loader = DataLoader(train_data, 250, shuffle=True) 

with open(X_test, 'rb') as pkl_file:
    test_input_data = pickle.load(pkl_file)
with open(y_test, 'rb') as pkl_file:
    test_label_data = pickle.load(pkl_file)

test_data = list(zip(test_input_data, test_label_data))
test_loader = DataLoader(test_data, 250) 

trainer = ClassifierTrainer()

trainer.train(run_name= args.run_name, train_dataloader=train_loader, val_dataloader=test_loader)