import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, RMSprop, Adam
from core.model import CNN
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
import torch
from torchmetrics import Accuracy

class ClassifierTrainer(object):
  def __init__(self):
    self.step_cache = {} 
  def train(self, run_name, train_dataloader, val_dataloader,
            learning_rate=0.0015, momentum=0, learning_rate_decay=0.95,
            num_epochs=30):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
    loss_function = CrossEntropyLoss()
    model = CNN(64).to(device)
    accuracy = Accuracy(task="multiclass", num_classes=64)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    lr_scheduler = OneCycleLR(optimizer, learning_rate, epochs=num_epochs, steps_per_epoch=len(train_dataloader))
    for epoch in range(num_epochs):
      loss_epoch = 0.0
      model.train()
      for index,data in tqdm(enumerate(train_dataloader)):
        inputs, labels = data
        labels = labels.long().to(device)
        inputs = inputs.float().to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss_epoch += loss.detach().cpu().item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
      
      ## Eval
      model.eval()
      all_preds = []
      all_labels = []
      for data in tqdm(val_dataloader):
        inputs, labels = data
        labels = labels.long().to(device)
        inputs = inputs.float().to(device)
        with torch.no_grad():
          outputs = model(inputs)
        preds = outputs.detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())

      all_preds = torch.tensor(all_preds, dtype=float)
      all_labels = torch.tensor(all_labels, dtype=int)

      torch.save(model.state_dict(), f'weights/{run_name}.pth')
      
      print(f"Accuracy val at epoch {epoch}: {accuracy(all_preds, all_labels)}")
      print(f"Epoch {epoch}| Loss: {loss_epoch / len(train_dataloader)}")