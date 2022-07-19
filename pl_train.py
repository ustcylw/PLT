import os, sys
from cv2 import log
from torchinfo import summary as TISummary
import torch
import torchvision as TV
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor, RichModelSummary
from pytorch_lightning.core.datamodule import LightningDataModule
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl


import PyUtils.PLModuleInterface as PLMI
from PyUtils.argus.image_argus import *
from PyUtils.bbox import BBoxes
import PyUtils.viz.cv_draw as CVDraw


from src.net.resnet_yolo import resnet50
from src.loss.yoloLoss import yoloLoss
from src.data.dataset import yoloDataset
from configs.config_v1 import CONFIGS
from src.data.encode_decode import encoder, decoder



class PLYOLOV1DataModule(LightningDataModule):
    name = 'pl-yolo-v1'
    def __init__(
        self, 
        configs,
        data_dir: str = CONFIGS.DATASET_DIR,
        val_split: float = 0.2,
        num_workers = 4,
        normalize=None,
        seed=7,
        batchsize=128,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.configs = configs
        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.batchsize = batchsize
        self.args = args
        self.kwargs = kwargs

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        ...

    def setup(self, stage=None):
        """Split the train and valid dataset."""
        # extra = dict(transform=self.default_transforms) if self.default_transforms else {}
        # dataset = MNIST(self.data_dir, train=True, download=False, **extra)
        # train_length = len(dataset)
        # self.dataset_train, self.dataset_val = random_split(dataset, [train_length - self.val_split, self.val_split])
        ...

    def train_dataloader(self):
        self.train_dataset = yoloDataset(
            list_file=[
                'voc2012.txt',
                'voc2007.txt'
            ],
            train=True,
            transform=[transforms.ToTensor()],
            config=self.configs
         )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.configs.BATCH_SIZE,
            shuffle=True,
            # num_workers=self.configs.NUM_WORKERS
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_dataset = yoloDataset(
            list_file='voc2007test.txt',
            train=False,
            transform=[transforms.ToTensor()],
            config=self.configs
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.configs.BATCH_SIZE,
            shuffle=False,
            # num_workers=self.configs.NUM_WORKERS,
            drop_last=True,
            pin_memory=False,
        )

        return self.val_loader

    def test_dataloader(self):
        self.test_dataset = yoloDataset(
            list_file='voc2007test.txt',
            train=False,
            transform=[transforms.ToTensor()],
            config=self.configs
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.configs.BATCH_SIZE,
            shuffle=False,
            # num_workers=self.configs.NUM_WORKERS,
            drop_last=True,
            pin_memory=False,
        )

        return self.test_loader


class PLYOLOV1Trainer(PLMI.PLMInterface):
    
    def __init__(self, hparams=...):
        super().__init__(hparams)

        self.model = self.create_model()
        self.criterion = yoloLoss(7,2,5,0.5)
        
        print(f'='*80)
        TISummary(self.model)
        print(f'-'*80)

    def create_model(self):
        
        resnet = TV.models.resnet50(pretrained=True)
        new_state_dict = resnet.state_dict()

        model = resnet50()
        dd = model.state_dict()
        for k in new_state_dict.keys():
            # print(k)
            if k in dd.keys() and not k.startswith('fc'):
                # print('yes')
                dd[k] = new_state_dict[k]
        model.load_state_dict(dd)
        
        # if self.hparams.configs.PRETRAINED is not None:
        #     print(f'loading pretrained-model [{self.hparams.configs.PRETRAINED}] ...')
        #     model = self.load_from_checkpoint(self.hparams.configs.PRETRAINED)
        #     print(f'loading pretrained-model [{self.hparams.configs.PRETRAINED}] complete.')
    
        return model

    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # 此处相当与对变量cross_entropy进行一次注册，如果不注册，tensorboard不显示。
        # 不建议放在这里，放到end_step比较好
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_acc', value=accuracy(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        # we currently return the accuracy as the validation_step/test_step is run on the IPU devices.
        # Outputs from the step functions are sent to the host device, where we calculate the metrics in
        # validation_epoch_end and test_epoch_end for the test_step.
        loss = self.criterion(probs, y)
        # acc = self.accuracy(probs, y)
        # 不建议放在这里，放到end_step比较好
        # self.log("val_acc", accuracy(probs, y), prog_bar=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):

        x,  y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        
        return preds, batch, batch_idx, loss

    def accuracy(self, logits, y):
        # currently IPU poptorch doesn't implicit convert bools to tensor
        # hence we use an explicit calculation for accuracy here. Once fixed in poptorch
        # we can use the accuracy metric.
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc

    def training_step_end(self, step_output):
        self.log('train_loss', step_output, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('lr', self.lr_schedulers().get_lr(), step_output, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        # since the training step/validation step and test step are run on the IPU device
        # we must log the average loss outside the step functions.
        self.log("val_loss", torch.stack(outputs).mean(), prog_bar=True, logger=True)

    def test_step_end(self, *args, **kwargs):
        preds, batch, batch_idx, loss = args[0]

        # 釋放顯存        
        preds = preds.cpu()
        images = batch[0].cpu().numpy()
        batch[1] = batch[1].cpu()
        loss = loss.cpu()
        
        h, w, _ = self.hparams.configs.INPUT_SHAPE

        for i in range(preds.shape[0]):
            
            pred = preds[i, :, :, :]
            bboxes,cls_indexs,probs = decoder(pred)

            image = images[i, :, :, :]
            image = image.transpose((1, 2 ,0)) * 255

            result = []
            clses = []
            for i,box in enumerate(bboxes):
                x1 = int(box[0]*w)
                x2 = int(box[2]*w)
                y1 = int(box[1]*h)
                y2 = int(box[3]*h)
                cls_index = cls_indexs[i]
                cls_index = int(cls_index) # convert LongTensor to int
                prob = probs[i]
                prob = float(prob)
                result.append([x1, y1, x2, y2, cls_index,prob])
                clses.append(self.hparams.configs.VOC_CLASSES[cls_index])

            torch.cuda.empty_cache()
            
            rets = BBoxes(result, mode='xyxycs')
            if len(rets.shape) < 2:
                continue
            if rets.shape[0] <= 0:
                continue
            
            labels = [f'{self.hparams.configs.VOC_CLASSES[int(cls)]} {score}' for cls, score in zip(rets[:, 4], rets[:, 5])]
            image = CVDraw.rectangles(image=cv2.UMat(image.astype(np.uint8)), bboxes=rets, labels=labels)
            if self.hparams.configs.TEST_SHOW:
                CVDraw.Show(name='cv-image-1').show(image=image, wait_time=100)
            if self.hparams.configs.TEST_SAVE:
                image_file = os.path.join(self.hparams.configs.TEST_SAVE_DIR, f'{batch_idx}-{i}.jpg')
                CVDraw.save_image(image=image, image_file=image_file)

    def test_epoch_end(self, outputs):
        ...

    def predict_step(self, batch, batch_idx):
        
        preds = self(batch[0])
        preds = preds.cpu()

        images = batch[0].cpu().numpy()
        batch[1] = batch[1].cpu()
        
        h, w, _ = CONFIGS.INPUT_SHAPE

        for i in range(preds.shape[0]):
            
            pred = preds[i, :, :, :]
            bboxes, cls_indexs, probs = decoder(pred)

            image = images[i, :, :, :]
            image = image.transpose((1, 2 ,0)) * 255

            result = []
            clses = []
            for i,box in enumerate(bboxes):
                x1 = int(box[0]*w)
                x2 = int(box[2]*w)
                y1 = int(box[1]*h)
                y2 = int(box[3]*h)
                cls_index = cls_indexs[i]
                cls_index = int(cls_index) # convert LongTensor to int
                prob = probs[i]
                prob = float(prob)
                result.append([x1, y1, x2, y2, cls_index,prob])
                clses.append(CONFIGS.VOC_CLASSES[cls_index])

            torch.cuda.empty_cache()
            
            rets = BBoxes(result, mode='xyxycs')
            if len(rets.shape) < 2:
                continue
            if rets.shape[0] <= 0:
                continue
            
            labels = [f'{CONFIGS.VOC_CLASSES[int(cls)]} {score}' for cls, score in zip(rets[:, 4], rets[:, 5])]
            image = CVDraw.rectangles(image=cv2.UMat(image.astype(np.uint8)), bboxes=rets, labels=labels)
            if CONFIGS.TEST_SHOW:
                CVDraw.Show(name='cv-image-1').show(image=image, wait_time=100)
            if CONFIGS.TEST_SAVE:
                image_file = os.path.join(CONFIGS.TEST_SAVE_DIR, f'{batch_idx}-{i}.jpg')
                print(f'{image_file=}  {self.device}')
                CVDraw.save_image(image=image, image_file=image_file)

    def configure_optimizers(self):
        adam = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=1e-4
        )
        # optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

        return {
            "optimizer": adam,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    adam, 
                    factor=0.6, 
                    patience=2, 
                    verbose=True, 
                    mode="min", 
                    threshold=1e-3, 
                    min_lr=1e-8, 
                    eps=1e-8
                ),
                "monitor": "val_loss",
                "frequency": 1  # "indicates how often the metric is updated"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            }
        }
    
    def configure_callbacks(self):
        log_dir = os.path.join(self.hparams.configs.ROOT_DIR, f'checkpoints/{self.hparams.configs.PRE_SYMBOL}_{self.hparams.configs.POST_SYMBOL}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        model_checkpoint = PLMI.ModelCheckpoint(
            monitor='val_loss',
            dirpath=log_dir,
            filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
            save_top_k=-1,
            save_last=True,
            every_n_epochs=1
        )
        # model_summary = RichModelSummary(max_depth=-1)  # ModelSummary(max_depth=-1)
        # device_stats = DeviceStatsMonitor()
        # lr_monitor = LearningRateMonitor(logging_interval='step')
        log_dir = os.path.join(self.hparams.configs.ROOT_DIR, f'runs/{self.hparams.configs.PRE_SYMBOL}_{self.hparams.configs.POST_SYMBOL}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        graph_callback = PLMI.GraphCallback(
            interval=10, 
            epoch_step=True, 
            log_dir=log_dir, 
            dummy_input=torch.rand(
                1, 
                self.hparams.configs.INPUT_SHAPE[2], 
                self.hparams.configs.INPUT_SHAPE[0], 
                self.hparams.configs.INPUT_SHAPE[1]
            )
        )
        # return [model_checkpoint, model_summary, lr_monitor, device_stats, graph_callback]
        return [model_checkpoint, graph_callback]


def predict():
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

    map_location = {'cpu': 'cuda:0'}
    model = PLYOLOV1Trainer.load_from_checkpoint(
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

    for batch_idx, batch in enumerate(val_loader):

        batch[0] = batch[0].cuda(0)
        
        preds = model(batch[0]).cpu()
        
        images = batch[0].cpu().numpy()
        batch[1] = batch[1].cpu()
        
        h, w, _ = CONFIGS.INPUT_SHAPE

        for i in range(preds.shape[0]):
            
            pred = preds[i, :, :, :]
            bboxes,cls_indexs,probs = decoder(pred)

            image = images[i, :, :, :]
            image = image.transpose((1, 2 ,0)) * 255

            result = []
            clses = []
            for i,box in enumerate(bboxes):
                x1 = int(box[0]*w)
                x2 = int(box[2]*w)
                y1 = int(box[1]*h)
                y2 = int(box[3]*h)
                cls_index = cls_indexs[i]
                cls_index = int(cls_index) # convert LongTensor to int
                prob = probs[i]
                prob = float(prob)
                result.append([x1, y1, x2, y2, cls_index,prob])
                clses.append(CONFIGS.VOC_CLASSES[cls_index])

            torch.cuda.empty_cache()
            
            rets = BBoxes(result, mode='xyxycs')
            if len(rets.shape) < 2:
                continue
            if rets.shape[0] <= 0:
                continue
            
            labels = [f'{CONFIGS.VOC_CLASSES[int(cls)]} {score}' for cls, score in zip(rets[:, 4], rets[:, 5])]
            image = CVDraw.rectangles(image=cv2.UMat(image.astype(np.uint8)), bboxes=rets, labels=labels)
            if CONFIGS.TEST_SHOW:
                CVDraw.Show(name='cv-image-1').show(image=image, wait_time=100)
            if CONFIGS.TEST_SAVE:
                image_file = os.path.join(CONFIGS.TEST_SAVE_DIR, f'{batch_idx}-{i}.jpg')
                CVDraw.save_image(image=image, image_file=image_file)





if __name__ == '__main__':

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
    model = PLYOLOV1Trainer.load_from_checkpoint(
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
    
    trainer.predict(model=model, dataloaders=val_loader)
