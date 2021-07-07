import pytorch_lightning as pl
from Dataset_Code.FashionMNIST import FashionMNISTDataModule
import torch.optim
from pytorch_lightning.callbacks import EarlyStopping


class NetworkTrainer:
    """
    This class is useful for conventional trainings with labeled train, validation and test set. If your implementation
    is any different e.g: Autoencoder with unlabeled data. Your best bet is to use pytorch_lightning.trainer as it is
    """
    def __init__(self, hparams, data_module_class, model_class, model_name, restore_model=False):
        self.gpus = 1
        self.hparams = hparams
        self.model_name = model_name
        self.restore_model = restore_model
        self.model_class, self.model = model_class, None
        self.data_module_class, self.data_module = data_module_class, None
        self.trainer = None
        self.setup()

    def setup(self):
        self.__setup_datamodule()
        if self.restore_model:
            self.__load_model()
        else:
            self.__setup_model()
        self.__setup_trainer()

    def __setup_trainer(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.hparams['patience'], mode='min')
        self.trainer = pl.Trainer(
            weights_summary=None,
            max_epochs=self.hparams["max_epochs"],
            gpus=self.gpus,
            callbacks=[early_stopping]
        )

    def __setup_datamodule(self):
        self.data_module = self.data_module_class(self.hparams["batch_size"])
        self.data_module.prepare_data()
        self.data_module.setup()

    def __setup_model(self):
        self.model = self.model_class(self.hparams)

    def __load_model(self):
        self.model = self.model_class.load_from_checkpoint(
            checkpoint_path="SavedModels/{}".format(self.model_name + ".ckpt"))

    def train_validate(self, save_model=False):
        self.trainer.fit(self.model, train_dataloader=self.data_module.train_dataloader(),
                         val_dataloaders=self.data_module.val_dataloader())
        if save_model:
            self.trainer.save_checkpoint("SavedModels/{}".format(self.model_name + ".ckpt"))

    def test_model(self):
        self.trainer.test(model=self.model, test_dataloaders=self.data_module.test_dataloader())








