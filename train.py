import random, argparse
from types import SimpleNamespace

import wandb
import timm
import torchvision as tv
import torchvision.transforms as T

from fastprogress import progress_bar

from torchmetrics import Accuracy

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_device(t, device):
    if isinstance(t, (tuple, list)):
        return [_t.to(device) for _t in t]
    elif isinstance(t, torch.Tensor):
        return t.to(device)
    else:
        raise("Not a Tensor or list of Tensors")
    return t    
        
WANDB_PROJECT = "fmnist_bench"
WANDB_ENTITY = "capecape"
        
config = SimpleNamespace(
    epochs=20, 
    model_name="resnet10t", 
    bs=256,
    device="cuda",
    seed=42,
    lr=1e-3,
    wd=1e-3)

def parse_args(config):
    parser = argparse.ArgumentParser(description='Run training baseline')
    parser.add_argument('--model_name', type=str, default=config.model_name, help='A model from timm')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--bs', type=int, default=config.bs, help='batch size')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning_rate')
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)

train_tfms = T.Compose([
    T.RandomCrop(28, padding=4), 
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,)),
])

val_tfms = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,)),
])

tfms ={"train": train_tfms, "valid":val_tfms}

 

class ImageModel:
    def __init__(self, data_path=".", tfms=tfms, model_name="resnet10t", device="cuda"):
        
        self.device = device
        self.config = SimpleNamespace(model_name=model_name, device=device)
        
        self.model = timm.create_model(model_name, pretrained=False, num_classes=10, in_chans=1).to(device)
        
        self.train_ds = tv.datasets.FashionMNIST(data_path, download=True, 
                                                 transform=tfms["train"])
        self.valid_ds = tv.datasets.FashionMNIST(data_path, download=True, train=False, 
                                                 transform=tfms["valid"])
        
        self.train_acc = Accuracy(task="multiclass", num_classes=10).to(device)
        self.valid_acc = Accuracy(task="multiclass", num_classes=10).to(device)
        
        self.do_validation = True
        
        self.dataloaders()
                 
        
    
    def dataloaders(self, bs=256, num_workers=8):
        self.config.bs = bs
        self.num_workers = num_workers
        self.train_dataloader = DataLoader(self.train_ds, batch_size=bs, shuffle=True, 
                                   pin_memory=True, num_workers=num_workers)
        self.valid_dataloader = DataLoader(self.valid_ds, batch_size=bs*2, shuffle=False, 
                                   num_workers=num_workers)

    def compile(self, epochs=5, lr=2e-3, wd=0.01, num_workers=8):
        self.config.epochs = epochs
        self.config.lr = lr
        self.config.wd = wd
        
        self.optim = AdamW(self.model.parameters(), weight_decay=wd)
        self.loss_func = nn.CrossEntropyLoss()
        self.schedule = OneCycleLR(self.optim, max_lr=lr, 
                                   steps_per_epoch=len(self.train_dataloader), 
                                   epochs=epochs)
    
    def reset(self):
        self.train_acc.reset()
        self.valid_acc.reset()
        
    def train_step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.schedule.step()
        return loss
        
    def one_epoch(self, train=True, use_wandb=False):
        avg_loss = 0.
        if train: 
            self.model.train()
            dl = self.train_dataloader
        else: 
            self.model.eval()
            dl = self.valid_dataloader
        pbar = progress_bar(dl, leave=False)
        for i, b in enumerate(pbar):
            with (torch.inference_mode() if not train else torch.enable_grad()):
                images, labels = to_device(b, self.device)
                with torch.autocast("cuda"):
                    preds = self.model(images)
                    loss = self.loss_func(preds, labels)
                avg_loss += loss
                if train:
                    self.train_step(loss)
                    acc = self.train_acc(preds, labels)
                    if use_wandb: 
                        wandb.log({"train_loss": loss.item(),
                                   "train_acc": acc,
                                   "learning_rate": self.schedule.get_last_lr()[0]})
                else:
                    acc = self.valid_acc(preds, labels)
            pbar.comment = f"train_loss={loss.item():2.3f}, train_acc={acc:2.3f}"      
            
        return avg_loss.mean().item(), acc
    
    def print_metrics(epoch, train_loss, val_loss):
        print(f"epoch: {epoch}, train_loss: {train_loss}, train_acc: {self.train_acc.compute()} | val_loss: {val_loss}, val_acc: {self.valid_acc.compute()}")
    
    def fit(self, use_wandb=False):
        if use_wandb:
            run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=self.config)
            
        for epoch in progress_bar(range(self.config.epochs), total=self.config.epochs, leave=True):
            train_loss, _ = self.one_epoch(train=True, use_wandb=use_wandb)
            
            if use_wandb:
                wandb.log({"epoch":epoch+1})
                
            ## validation
            if self.do_validation:
                val_loss, _ = self.one_epoch(train=False, use_wandb=use_wandb)
                if use_wandb:
                    wandb.log({"val_loss": val_loss,
                               "val_acc": self.valid_acc.compute()})
            self.print_metrics(epoch, train_loss, val_loss)
            self.reset()
        if use_wandb:
            wandb.finish()

            
if __name__=="__main__":
    parse_args(config)
    
    set_seed(config.seed)
    
    model = ImageModel(model_name=config.model_name)
    model.compile(epochs=20, lr=config.lr, wd=config.wd)
    model.fit(use_wandb=True)