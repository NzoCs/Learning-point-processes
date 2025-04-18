from easy_tpp.models import BaseModel
from easy_tpp.preprocess import TPPDataModule
from easy_tpp.config_factory import RunnerConfig
from easy_tpp.utils import logger

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
import os


class Trainer :
    
    def __init__(self, config : RunnerConfig, **kwargs):
        
        """_summary__.
        Args:
            config (RunnerConfig): Configuration object containing all the necessary parameters for training.
            **kwargs: Additional keyword arguments that can be used to override specific configurations.
        """
        
        # Set matmul precision for Tensor Cores (if available)
        # Recommended for A100 GPUs as per logs
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7:
             # Check if the major capability is >= 7 (Volta, Ampere, Hopper, etc.)
             try:
                 torch.set_float32_matmul_precision('medium')
                 logger.info("Set torch.set_float32_matmul_precision('medium') for Tensor Cores.")
             except Exception as e:
                 logger.warning(f"Could not set matmul precision: {e}")

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
        self.patience = trainer_config.patience
        self.devices = trainer_config.devices
        self.logger_config = trainer_config.get('logger_config')
        self.val_freq = trainer_config.val_freq
        self.use_precision_16 = trainer_config.use_precision_16
        self.accumulate_grad_batches = trainer_config.accumulate_grad_batches
        
        # Use the dirpath directly from the trainer_config
        self.dirpath = trainer_config.save_model_dir
        
        dataset_id = data_config.dataset_id
        logger.info(f"--- Starting Training/Testing for Model : {self.model_id} on dataset : {dataset_id} ---")

        try:
            self.logger = trainer_config.get_logger()
        except Exception as e:
            self.logger = False
            logger.critical(f"Logging is disabled for this run. Error: {str(e)}")
        
    @property
    def callbacks(self):
        
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath= self.dirpath,
            filename = f'{self.model_id}'+'-{epoch:02d}-{val_loss:.2f}',
            save_top_k=2,
            mode = 'min',
            every_n_epochs = self.checkpoints_freq
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
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
            
        # Check if distributed training is requested
        if self.devices > 1:
            # Use DDPStrategy with find_unused_parameters=True
            strategy = DDPStrategy(find_unused_parameters=True)
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                devices=self.devices,
                accelerator='gpu',
                strategy=strategy,
                logger=self.logger,
                log_every_n_steps=self.log_freq,
                callbacks=self.callbacks,
                enable_progress_bar=True,
                enable_model_summary=True,
                check_val_every_n_epoch=self.val_freq,
                precision='16-mixed' if self.use_precision_16 else '32-true',
                accumulate_grad_batches=self.accumulate_grad_batches
            )
            
        else:
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
                accumulate_grad_batches=self.accumulate_grad_batches
            )
        
        return trainer
    
    def train(self) -> None:
        """Training a model."""
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
        Results are saved to a JSON file in the model directory.
        """
        trainer = self.trainer
        self.datamodule.setup(stage='test')
        
        test_dataloader = self.datamodule.test_dataloader()
        
        results = trainer.test(
            model = self.model,
            dataloaders = test_dataloader
        )
        
        # Save test results
        if results and len(results) > 0:
            import json
            
            results_file = os.path.join(self.dirpath, f'test_results.json')
            
            with open(results_file, 'w') as f:
                json.dump(results[0], f, indent=4)
            
            logger.info(f"Test results saved to {results_file}")
                
        return results
