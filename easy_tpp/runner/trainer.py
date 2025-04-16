from easy_tpp.models import BaseModel
from easy_tpp.preprocess import TPPDataModule
from easy_tpp.config_factory import RunnerConfig
from easy_tpp.utils import logger

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
import os
import random


class Trainer :
    
    def __init__(self, config : RunnerConfig, **kwargs):
        
        """_summary__.
        Args:
            config (RunnerConfig): Configuration object containing all the necessary parameters for training.
            **kwargs: Additional keyword arguments that can be used to override specific configurations.
        """
        
        #Initialize your configs
        data_config = config.data_config
        model_config = config.model_config
        trainer_config = config.trainer_config
        
        # Initialize your model
        self.max_epochs = trainer_config.max_epochs
        
        self.model = BaseModel.generate_model_from_config(model_config=model_config)
        
        self.model_id = model_config.model_id
        
        
        #Intialize Dataloaders
        self.datamodule = TPPDataModule(data_config)
        
        
        #Initialize Train params
        self.log_freq = trainer_config.log_freq
        self.checkpoints_freq = trainer_config.checkpoints_freq
        self.patience = trainer_config.patience_max
        self.devices = trainer_config.devices
        self.logger_config = trainer_config.get('logger_config')
        self.val_freq = trainer_config.val_freq
        self.use_precision_16 = trainer_config.use_precision_16
        
        # Use the dirpath directly from the trainer_config
        self.dirpath = trainer_config.save_model_dir
        
        try:
            self.logger = trainer_config.get_logger()
        except Exception as e:
            self.logger = False
            logger.critical(f"Logging is disabled for this run. Error: {str(e)}")
        
    @property
    def callbacks(self):
        
        
        checkpoint_callback = ModelCheckpoint(
            monitor='avg_val_loss',
            dirpath= self.dirpath,
            filename = f'{self.model_id}'+'-{epoch:02d}-{val_loss:.2f}',
            save_top_k=2,
            mode = 'min',
            every_n_epochs = self.checkpoints_freq
        )
        
        early_stop_callback = EarlyStopping(
            monitor='avg_val_loss',
            patience= self.patience,
            verbose=False,
            mode='min'
        )
        
        return [checkpoint_callback, early_stop_callback]
    
    @property
    def trainer(self) -> pl.Trainer:
        # If devices is a number > 0, use GPU(s) if available
        # If devices is "auto", let PyTorch Lightning decide
        if isinstance(self.devices, int) and self.devices > 0 and torch.cuda.is_available():
            devices = self.devices
            accelerator = 'gpu'
            strategy = 'ddp' if self.devices > 1 else "auto"
        else:
            devices = self.devices  # Could be "auto" or another value
            accelerator = 'auto'
            strategy = "auto"
        
        #Wheter to use float16 or float32
        if self.use_precision_16 :
            precision = '16-mixed'
        else :
            precision = '32-true'
            
        
        trainer = pl.Trainer(
            max_epochs = self.max_epochs,
            devices = devices,
            accelerator = accelerator,
            strategy = strategy,
            logger = self.logger,
            log_every_n_steps = self.log_freq,
            callbacks = self.callbacks,
            enable_progress_bar = True,
            enable_model_summary = True,
            check_val_every_n_epoch = self.val_freq,
            precision = precision,
        )
        
        return trainer
    
    def train(self) -> None:
        
        trainer = self.trainer
        # Train the model
        self.datamodule.setup(stage='fit')
        
        train_dataloader = self.datamodule.train_dataloader()
        val_dataloader = self.datamodule.val_dataloader()
        
        trainer.fit(
            model = self.model,
            train_dataloaders = train_dataloader,
            val_dataloaders = val_dataloader,
            )
        
    def test(self) -> None:
        """
        Test the model with optional custom parameters for the test_step method.
        """
        trainer = self.trainer
        self.datamodule.setup(stage='test')
        
        test_dataloader = self.datamodule.test_dataloader()
        
        results = trainer.test(
            model = self.model,
            dataloaders = test_dataloader
        )
                
        return results