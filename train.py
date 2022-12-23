import random, argparse
from types import SimpleNamespace

import wandb
import timm
import torchvision as tv
import torchvision.transforms as T

from fastprogress import progress_bar

from torcheval.metrics import MulticlassAccuracy, Mean

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from preds_logger import PredsLogger


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
    bs=512,
    device="cuda",
    seed=42,
    lr=1e-2,
    wd=0.)

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

        
mean, std = (0.28, 0.35)
        
train_tfms = T.Compose([
    T.RandomCrop(28, padding=1), 
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean, std),
    T.RandomErasing(scale=(0.02, 0.25), value="random"),
])

val_tfms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std),
])

tfms = {"train": train_tfms, "valid":val_tfms}

class ImageModel:
    def __init__(self, model, data_path=".", tfms=tfms, device="cuda", bs=256, use_wandb=False):
        
        self.device = device
        self.config = SimpleNamespace(device=device)
        
        self.model = model.to(device)
        
        self.train_ds = tv.datasets.FashionMNIST(data_path, download=True, 
                                                 transform=tfms["train"])
        self.valid_ds = tv.datasets.FashionMNIST(data_path, download=True, train=False, 
                                                 transform=tfms["valid"])
        
        self.train_acc = MulticlassAccuracy(device=device)
        self.valid_acc = MulticlassAccuracy(device=device)
        self.loss = Mean()
        
        self.do_validation = True
        
        self.dataloaders(bs=bs)
        self.config.tfms = tfms
        
        self.use_wandb = use_wandb
        
        # get validation data reference
        if use_wandb:
            self.preds_logger = PredsLogger(ds=self.valid_ds) 
    
    @classmethod
    def from_timm(cls, model_name, data_path=".", tfms=tfms, device="cuda", bs=256 ):
        model = timm.create_model(model_name, pretrained=False, num_classes=10, in_chans=1)
        image_model = cls(model, data_path, tfms, device, bs)
        image_model.config.model_name = model_name
        return image_model
            
    def dataloaders(self, bs=128, num_workers=8):
        self.config.bs = bs
        self.num_workers = num_workers
        self.train_dataloader = DataLoader(self.train_ds, batch_size=bs, shuffle=True, 
                                   pin_memory=True, num_workers=num_workers)
        self.valid_dataloader = DataLoader(self.valid_ds, batch_size=bs*2, shuffle=False, 
                                   num_workers=num_workers)
    
    def log(self, d):
        if self.use_wandb and (wandb.run is not None):
            wandb.log(d)
    
    def compile(self, epochs=5, lr=2e-3, wd=0.01):
        self.config.epochs = epochs
        self.config.lr = lr
        self.config.wd = wd
        
        self.optim = AdamW(self.model.parameters(), weight_decay=wd)
        self.loss_func = nn.CrossEntropyLoss()
        self.schedule = OneCycleLR(self.optim, 
                                   max_lr=lr, 
                                   total_steps=epochs*len(self.train_dataloader))
    
    def reset_metrics(self):
        self.train_acc.reset()
        self.valid_acc.reset()
        self.loss.reset()
        
    def train_step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.schedule.step()
        return loss
        
    def one_epoch(self, train=True):
        if train: 
            self.model.train()
            dl = self.train_dataloader
        else: 
            self.model.eval()
            dl = self.valid_dataloader
        pbar = progress_bar(dl, leave=False)
        preds = []
        for i, b in enumerate(pbar):
            with (torch.inference_mode() if not train else torch.enable_grad()):
                images, labels = to_device(b, self.device)
                # with torch.autocast("cuda"):
                preds_b = self.model(images)
                loss = self.loss_func(preds_b, labels)
                self.loss.update(loss.detach().cpu(), weight=len(images))
                preds.append(preds_b)
                if train:
                    self.train_step(loss)
                    acc = self.train_acc.update(preds_b, labels)
                    self.log({"train_loss": loss.item(),
                              "train_acc": acc.compute(),
                              "learning_rate": self.schedule.get_last_lr()[0]})
                else:
                    acc = self.valid_acc.update(preds_b, labels)
            pbar.comment = f"train_loss={loss.item():2.3f}, train_acc={acc.compute():2.3f}"      
            
        return torch.cat(preds, dim=0), self.loss.compute()
    
    def log_preds(self, preds):
        if self.use_wandb:
            print("Logging model predictions on validation data")
            preds, _ = self.get_model_preds()
            self.preds_logger.log(preds=preds)
    
    def get_model_preds(self, with_inputs=False):
        preds, loss = self.one_epoch(train=False)
        if with_inputs:
            images, labels = self.get_data_tensors()
            return images, labels, preds, loss
        else:
            return preds, loss
            
    def print_metrics(self, epoch, train_loss, val_loss):
        print(f"epoch: {epoch:3}, train_loss: {train_loss:10.3f}, train_acc: {self.train_acc.compute():3.3f}   ||   val_loss: {val_loss:10.3f}, val_acc: {self.valid_acc.compute():3.3f}")
    
    def fit(self, log_preds=False):         
        for epoch in progress_bar(range(self.config.epochs), total=self.config.epochs, leave=True):
            _, train_loss = self.one_epoch(train=True)
            
            self.log({"epoch":epoch})
                
            ## validation
            if self.do_validation:
                _, val_loss = self.one_epoch(train=False)
                self.log({"val_loss": val_loss,
                          "val_acc": self.valid_acc.compute()})
            self.print_metrics(epoch, train_loss, val_loss)
            self.reset_metrics()
        if log_preds:
            self.log_preds(preds)

            
if __name__=="__main__":
    parse_args(config)
    
    set_seed(config.seed)
    
    if config.model_name == "jeremy":
        print("Using custom Jeremy's Model")
        from jeremy_resnet import resnet
        
        leak = 0.1
        sub = 0.4
        
        model = ImageModel(resnet(leak, sub), bs=512)
        model.config.model_name = {"model_name":"jeremy", "leak":leak, "sub":sub}
    else:
        model = ImageModel.from_timm(model_name=config.model_name)
    
    
    model.compile(epochs=config.epochs, lr=config.lr, wd=config.wd)
    
    # train
    with wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=model.config):
        model.fit()