import pickle
from core.engine import ClassifierTrainer
from torch.utils.data import DataLoader

with open('data_pkl/X_train_3000.pkl', 'rb') as pkl_file:
    input_data = pickle.load(pkl_file)
with open('data_pkl/y_train_3000.pkl', 'rb') as pkl_file:
    label_data = pickle.load(pkl_file)

train_data = list(zip(input_data, label_data))
train_loader = DataLoader(train_data, 100, shuffle=True) 

with open('data_pkl/X_test_100.pkl', 'rb') as pkl_file:
    test_input_data = pickle.load(pkl_file)
with open('data_pkl/y_test_100.pkl', 'rb') as pkl_file:
    test_label_data = pickle.load(pkl_file)

test_data = list(zip(test_input_data, test_label_data))
test_loader = DataLoader(test_data, 100) 

traner = ClassifierTrainer()

traner.train(train_dataloader=train_loader, val_dataloader=test_loader)