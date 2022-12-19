import random, argparse
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms

import timm

import composer
from composer.models import ComposerClassifier
from composer import Trainer
from composer.algorithms import ChannelsLast, CutMix, LabelSmoothing, BlurPool, RandAugment, MixUp, EMA
from composer.models import mnist_model
from composer.algorithms.randaugment import RandAugmentTransform

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



config = SimpleNamespace(
    epochs=20, 
    model_name="resnet10t", 
    bs=256,
    device="gpu",
    seed=42,
    lr=2e-2,
    #algorithms
    use_randaug=True,
    randaug_depth=1,
    use_smoothing=True,
    labelsmoothing=0.08,
    use_mixup=True,
    mixup=0.15,
    use_cutmix=True,
    use_ema=True,
    use_blur=True,
    #optim
    warmup=2,
    optim="adam",
    scheduler="onecycle",
    )

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
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(".", download=True, train=True, transform=transform)
    eval_dataset = datasets.FashionMNIST(".", download=True, train=False, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=config.bs, num_workers=8, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.bs,num_workers=8)
    return train_dataloader, eval_dataloader




if __name__=="__main__":
    parse_args(config)

    wandb_logger = WandBLogger(project="fmnist", entity="fastai", tags=["composer"])
    set_seed(config.seed)
    
    timm_model = timm.create_model(config.model_name, pretrained=False, num_classes=10, in_chans=1)
    model = ComposerClassifier(timm_model)
    
    train_dataloader, eval_dataloader = load_data(config)
    
    if config.optim.lower() == "sgd":
        optimizer = composer.optim.DecoupledSGDW(
            model.parameters(), # Model parameters to update
            lr=config.lr, # Peak learning rate
            momentum=0.9,
            weight_decay=2.0e-3 # If this looks large, it's because its not scaled by the LR as in non-decoupled weight decay
        )
    elif config.optim.lower() == "adam":
        optimizer = composer.optim.DecoupledAdamW(
            model.parameters(), 
            lr=config.lr, 
            betas=(0.9, 0.95), 
            eps=1e-08, 
            weight_decay=1e-05, 
            amsgrad=False
        )
    elif config.optim.lower() == "amsgrad":
        optimizer = composer.optim.DecoupledAdamW(
            model.parameters(), 
            lr=config.lr, 
            betas=(0.9, 0.95), 
            eps=1e-08, 
            weight_decay=1e-05, 
            amsgrad=True
        )
    else:
        raise Error("Not optimizer selcted")
    
    if config.scheduler == "cosine":
        lr_scheduler = composer.optim.CosineAnnealingWithWarmupScheduler(
            t_warmup=f'{config.warmup}ep', 
            t_max='1dur'
        )
        step_schedulers_every_batch = False
    elif config.scheduler == "onecycle":
        lr_scheduler = OneCycleLR(
            optimizer, max_lr=config.lr, 
            steps_per_epoch=len(train_dataloader), 
            epochs=config.epochs
        )
        step_schedulers_every_batch = True
    else:
        lr_scheduler = composer.optim.LinearWithWarmupScheduler(
            t_warmup="1ep", # Warm up over 1 epoch
            alpha_i=1.0, # Flat LR schedule achieved by having alpha_i == alpha_f
            alpha_f=1.0
        )
        step_schedulers_every_batch = False
        
    algorithms=[]
    if config.use_mixup:
        algorithms.append(MixUp(alpha=config.mixup)),
    if config.use_smoothing:
        algorithms.append(LabelSmoothing(smoothing=config.labelsmoothing))
    if config.use_randaug:
        algorithms.append(RandAugment(depth=config.randaug_depth, augmentation_set="safe"))
    if config.use_blur:
        algorithms.append(BlurPool(replace_convs=True, replace_maxpools=True))
    if config.use_cutmix:
        algorithms.append(CutMix(alpha=.9))
    if config.use_ema:
        algorithms.append(EMA(half_life='100ba'))                       
            
    trainer = composer.trainer.Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration=f"{config.epochs}ep",
        optimizers=optimizer,
        schedulers=lr_scheduler,
        step_schedulers_every_batch=step_schedulers_every_batch,
        device=config.device,
        precision='amp',
        algorithms=algorithms,
        loggers=wandb_logger,
    )
    trainer.fit()