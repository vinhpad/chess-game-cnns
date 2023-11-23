import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from core.model import CNN
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
import torch
from torchmetrics import Accuracy

class ClassifierTrainer(object):
  def __init__(self):
    self.step_cache = {} 
  def train(self, train_dataloader, val_dataloader,
            learning_rate=1e-2, momentum=0, learning_rate_decay=0.95,
            num_epochs=30):
    
    loss_function = CrossEntropyLoss()
    model = CNN(64)
    accuracy = Accuracy(task="multiclass", num_classes=64)
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    lr_scheduler = OneCycleLR(optimizer, learning_rate, epochs=num_epochs, steps_per_epoch=len(train_dataloader))
    for epoch in range(num_epochs):
      loss_epoch = 0.0
      for index,data in enumerate(train_dataloader):
        inputs, labels = data
        labels = labels.long()
        inputs = inputs.float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss_epoch += loss.detach().cpu().item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
      
      ## Eval
      all_preds = []
      all_labels = []
      for data in tqdm(val_dataloader):
        inputs, labels = data
        labels = labels.long()
        inputs = inputs.float()
        with torch.no_grad():
          outputs = model(inputs)
        preds = outputs.detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

      all_preds = torch.tensor(all_preds, dtype=float)
      all_labels = torch.tensor(all_labels, dtype=int)
      
      print(f"Accuracy val at epoch {epoch}: {accuracy(all_preds, all_labels)}")
      print(f"Epoch {epoch}| Loss: {loss / len(train_dataloader)}")