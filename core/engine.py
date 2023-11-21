import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from core.model import CNN
from tqdm import tqdm

class ClassifierTrainer(object):
  def __init__(self):
    self.step_cache = {} 
  def train(self, train_dataloader, 
            reg=0.0, dropout=1.0,
            learning_rate=1e-2, momentum=0, learning_rate_decay=0.95,
            num_epochs=30, batch_size=100, acc_frequency=None,
            augment_fn=None, predict_fn=None,
            verbose=False):
    
    best_val_acc = 0.0
    best_model = {}
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    loss_function = CrossEntropyLoss()
    model = CNN(64)
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for index,data in tqdm(enumerate(train_dataloader)):
        inputs, labels = data
        print(inputs)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()


    #return best_model, loss_history, train_acc_history, val_acc_history
