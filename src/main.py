import pl_train as PLT
import pytorch_lightning as pl
from configs.config_v1 import CONFIGS




if __name__ == '__main__':


    
    dm = PLT.PLYOLOV1DataModule(
        configs=CONFIGS,
        data_dir=CONFIGS.DATASET_DIR,
        val_split=0.2,
        num_workers = 4,
        normalize=None,
        seed=CONFIGS.SEED,
        batchsize=CONFIGS.BATCH_SIZE
    )
    
    model = PLT.PLYOLOV1Trainer(
        hparams={
            'batch_size': CONFIGS.BATCH_SIZE,
            'auto_scale_batch_size': True,
            'auto_lr_find': True,
            'learning_rate': CONFIGS.LEARNING_RATE,
            'reload_dataloaders_every_epoch': False,
            'hidden_dim': 32,
            'configs': CONFIGS
        }
    )

    trainer = pl.Trainer(
        max_steps=-1,
        max_epochs=80, 
        accelerator="gpu", 
        devices=2,
        enable_checkpointing=True,
        weights_save_path='/data/ylw/code/git_yolo/pytorch-YOLO-v1/pl_yolo_v1/runs',
        limit_train_batches=1.0,
        log_every_n_steps=10,
        precision=16,
        val_check_interval=1.0,
        # enable_model_summary=True
    )

    from src.data.mnist_datamodule import MNISTDataModule
    from PyUtils.PLModuleInterface import MnistInterface
    
    # dm = MNISTDataModule(batch_size=128)
    # model = MnistInterface()

    dm.setup(CONFIGS.MODE)
    if CONFIGS.MODE == 'train':
        trainer.fit(model, datamodule=dm)
    elif CONFIGS.MODE == 'test':
        # model = PLYOLOV1Trainer.load_from_checkpoint(CONFIGS.PRETRAINED)
        model = model.load_from_checkpoint(CONFIGS.PRETRAINED)
        trainer.testing = True
        trainer.test(model, datamodule=dm)
