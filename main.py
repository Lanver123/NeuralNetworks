from Trainer.NetworkTrainer import NetworkTrainer
import torch.optim
from Dataset_Code.FashionMNIST import FashionMNISTDataModule
from Networks.FashionMNISTNet import FashionMNISTNet


if __name__ == '__main__':

    hparams = {
        "max_epochs": 1,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "n_hidden": (512,),
        "optimizer": torch.optim.Adam,
        "patience": 10
    }

    trainer = NetworkTrainer(hparams=hparams, model_name="fashion_mnist_{}".format(str(hparams)),
                             model_class=FashionMNISTNet, data_module_class=FashionMNISTDataModule, restore_model=False)
    trainer.setup()
    trainer.train_validate(save_model=True)
    #trainer.test_model()
