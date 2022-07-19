import os, sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from PyUtils.argus.image_argus import *
import PyUtils.viz.cv_draw as CVDraw

from configs.config_v1 import CONFIGS
import pl_train as PLT
from src.data.dataset import yoloDataset
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
        devices=1,
        enable_checkpointing=True,
        weights_save_path='/data/ylw/code/git_yolo/pytorch-YOLO-v1/pl_yolo_v1/runs',
        limit_train_batches=1.0,
        log_every_n_steps=10,
        precision=16,
        val_check_interval=1.0,
        # enable_model_summary=True
    )

    dm.setup(CONFIGS.MODE)
    if CONFIGS.MODE == 'train':
        trainer.fit(model, datamodule=dm)
    elif CONFIGS.MODE == 'test':
        # model = PLYOLOV1Trainer.load_from_checkpoint(CONFIGS.PRETRAINED)
        model = model.load_from_checkpoint(
            CONFIGS.PRETRAINED, 
            hparams_file=f'/data/ylw/code/git_yolo/pytorch-YOLO-v1/pl_yolo_v1/lightning_logs/version_90/hparams.yaml'
        )
        model.eval()
        trainer.testing = True
        trainer.test(model, datamodule=dm)
    elif CONFIGS.MODE == 'predict':
        
        val_dataset = yoloDataset(
                list_file='voc2007test.txt',
                train=False,
                transform=[transforms.ToTensor()],
                config=CONFIGS
            )

        val_loader = DataLoader(
                val_dataset,
                batch_size=CONFIGS.BATCH_SIZE,
                shuffle=False,
                drop_last=True,
                pin_memory=False,
        )

        map_location = {'cpu': 'cuda:0'}  # 好像不起作用
        model = PLT.PLYOLOV1Trainer.load_from_checkpoint(
            CONFIGS.PRETRAINED, 
            map_location=map_location,
            hparams_file=f'/data/ylw/code/git_yolo/pytorch-YOLO-v1/pl_yolo_v1/lightning_logs/version_90/hparams.yaml'
        )
        print(f'{model.device=}')
        model.eval()
        model.freeze()
        print(f'{model.device=}')
        model.to('cuda:0')
        print(f'{model.device=}')

        trainer = pl.Trainer(
            accelerator="gpu", 
            devices=1,
            precision=16,
        )
        
        trainer.predict(model=model, dataloaders=val_loader)
