import warnings
warnings.filterwarnings("ignore")

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from config import args
from dataset import STForecastingDataset
from model import STID

class LightningSTID(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = STID(**args.model)
        self.loss_fn = args.train.loss

    def forward(self, seq_x, seq_x_mark):
        output = self.model(seq_x, seq_x_mark)
        return output

    def training_step(self, batch, batch_idx):
        seq_x, seq_y, seq_x_mark, seq_y_mark = batch
        output = self(seq_x, seq_x_mark)
        loss = self.loss_fn(output, seq_y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq_x, seq_y, seq_x_mark, seq_y_mark = batch
        output = self(seq_x, seq_x_mark)
        loss = self.loss_fn(output, seq_y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return output
    
    def test_step(self, batch, batch_idx):
        seq_x, seq_y, seq_x_mark, seq_y_mark = batch
        output = self(seq_x, seq_x_mark)
        loss = self.loss_fn(output, seq_y)
        self.log("test_loss", loss, on_epoch=True, logger=True)
        return output
        
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.args.train.lr)
        return optim
    
    
L.seed_everything(args.train.seed)

train_set = STForecastingDataset(args.train.root_path,flag='train')
val_set = STForecastingDataset(args.train.root_path,flag='val')
test_set = STForecastingDataset(args.train.root_path, flag='test')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.train.batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.train.batch_size)
model = LightningSTID(args)

trainer = L.Trainer(max_steps=1000)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(dataloaders=test_loader, ckpt_path='best')
