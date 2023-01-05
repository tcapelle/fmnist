import wandb
from fastprogress import progress_bar

VAL_DATA_AT = 'fastai/fmnist_pt/validation_data:latest'

class PredsLogger:
    def __init__(self, ds, image_mode="L", n=None, val_data_at=VAL_DATA_AT):
        self.ds = ds
        self.image_mode = image_mode
        self.n = n
        self.val_data_at = val_data_at
        
    def _get_reference_table(self):
        artifact = wandb.use_artifact(self.val_data_at, type='data')
        self.val_table = artifact.get("val_table")
        return True
    
    def _init_preds_table(self, num_classes=10):
        "Create predictions table"
        self.preds_table = wandb.Table(columns=["image", "label", "preds"]+[f"prob_{i}" for i in range(num_classes)])
        
    def create_preds_table(self, preds, n=None):
        if self.val_table is None:
            print("No val table reference found")
            return
        table_idxs = self.val_table.get_index()
        
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
                 
        for idx in progress_bar(table_idxs[:n], leave=False):
            pred = preds[idx]
            self.preds_table.add_data(
                self.val_table.data[idx][1],
                self.val_table.data[idx][2],
                pred.argmax(),
                *pred
            )
    
    def log(self, preds, n=None, table_name="preds_table", aliases=None):
        # get the validation data from the reference
        self._get_reference_table()
            
        # create the Predictions Table 
        self._init_preds_table(num_classes=preds.shape[-1])
        
        # Populate the Table with the model predictions
        self.create_preds_table(preds, n=n)
        
        # Log to W&B
        assert wandb.run is not None
        pred_artifact = wandb.Artifact(f"run_{wandb.run.id}_preds", type="evaluation")
        pred_artifact.add(self.preds_table, table_name)
        wandb.log_artifact(pred_artifact, aliases=aliases or ["latest"])
        
        wandb.log({"preds_table":self.preds_table})
        