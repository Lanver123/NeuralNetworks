from Networks.Autoencoder.Autoencoder import Autoencoder
from Dataset_Code.UnlabeledMNIST import UnlabeledMNIST
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch


if __name__ == '__main__':
    hparams = {
        "max_epochs": 1,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "n_hidden": (512,),
        "optimizer": torch.optim.Adam,
        "patience": 10,
        "ae_mode": "decoder",
        'embedding_dim': 5
    }

    mnist_data = UnlabeledMNIST(hparams['batch_size'])
    mnist_data.prepare_data()
    mnist_data.setup()

    autoencoder = Autoencoder(hparams)

    early_stopping = EarlyStopping(monitor='val_classification_loss', patience=hparams['patience'], mode='min')
    trainer = pl.Trainer(
        weights_summary=None,
        max_epochs=hparams["max_epochs"],
        gpus=1,
        callbacks=[early_stopping]
    )

    trainer.fit(model=autoencoder, train_dataloader=mnist_data.unlabeled_dataloader())
    autoencoder.save_encoder()
    autoencoder.switch_mode('classification')

    trainer.fit(model=autoencoder, train_dataloader=mnist_data.labeled_train_dataloader(),
                val_dataloaders=mnist_data.labeled_val_dataloader())

    trainer.test(model=autoencoder, test_dataloaders=mnist_data.test_dataloader())

    #self.trainer.save_checkpoint("SavedModels/{}".format(self.model_name + ".ckpt"))