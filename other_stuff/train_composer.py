import random, argparse
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from torchvision import datasets
import torchvision.transforms as T

import timm

import composer
from composer.models import ComposerClassifier
from composer import Trainer

from composer.loggers import WandBLogger

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

tfms = {"train": train_tfms, "valid":val_tfms}
        

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



def load_data(config):
    train_dataset = datasets.FashionMNIST(".", download=True, train=True, transform=tfms["train"])
    eval_dataset = datasets.FashionMNIST(".", download=True, train=False, transform=tfms["valid"])
    train_dataloader = DataLoader(train_dataset, batch_size=config.bs, num_workers=8, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.bs, num_workers=8)
    return train_dataloader, eval_dataloader




if __name__=="__main__":
    parse_args(config)

    wandb_logger = WandBLogger(project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=["composer"])
    set_seed(config.seed)
    
    timm_model = timm.create_model(config.model_name, pretrained=False, num_classes=10, in_chans=1)
    model = ComposerClassifier(timm_model)
    
    train_dataloader, eval_dataloader = load_data(config)
    
    optimizer = AdamW(model.parameters(), weight_decay=config.wd)
    scheduler = OneCycleLR(optimizer, max_lr=config.lr, 
                           steps_per_epoch=len(train_dataloader), 
                           epochs=config.epochs)
    train_epochs = f"{config.epochs}ep" # Train for 3 epochs because we're assuming Colab environment and hardware
    device = "gpu" if torch.cuda.is_available() else "cpu" # select the device

    trainer = composer.trainer.Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration=train_epochs,
        optimizers=optimizer,
        schedulers=scheduler,
        device=device,
        precision='amp',
        loggers=wandb_logger,
    )
    
    trainer.fit()