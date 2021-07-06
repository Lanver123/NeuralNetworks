import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class FashionMNISTNet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = nn.Sequential(
            nn.Linear(1 * 28 * 28, self.hparams['n_hidden'][0]),
            nn.Sigmoid(),
            nn.Linear(self.hparams['n_hidden'][0], 10)
        )

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, x):
        # flatten the image  before sending as input to the model
        n = x.shape[0]
        x = x.view(n, -1)

        y = self.model(x)

        return y

    def general_step(self, batch, batch_idx, mode: str):
        images, targets = batch

        # Perform a forward pass on the network with inputs
        out = self.forward(images)

        # calculate the loss with the network predictions and ground truth targets
        loss = F.cross_entropy(out, targets)

        # Find the predicted class from probabilities of the image belonging to each of the classes
        # from the network output
        _, preds = torch.max(out, 1)

        # Calculate the accuracy of predictions
        acc = preds.eq(targets).sum().float() / targets.size(0)

        # Log the accuracy and loss values to the tensorboard
        self.log('loss_{}'.format(mode), loss)
        self.log('acc_{}'.format(mode), acc)

        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "training")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "validation")

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    def validation_epoch_end(self, outputs):
        # Average the loss over the entire validation data from it's mini-batches
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        # Log the validation accuracy and loss values to the tensorboard
        self.log('val_loss', avg_loss)
        self.log('val_acc', avg_acc)

    def configure_optimizers(self):
        optimizer = self.hparams['optimizer'](self.model.parameters(), lr=self.hparams['learning_rate'])

        return optimizer

    def visualize_predictions(self, images, preds, targets):

        # Helper function to help us visualize the predictions of the
        # validation data by the model

        class_names = ['t-shirts', 'trouser', 'pullover', 'dress',
                       'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

        # determine size of the grid based for the given batch size
        num_rows = torch.tensor(len(images)).float().sqrt().floor()

        fig = plt.figure(figsize=(10, 10))
        for i in range(len(images)):
            plt.subplot(num_rows, len(images) // num_rows + 1, i+1)
            plt.imshow(images[i].cpu().numpy().squeeze(0))
            plt.title(class_names[torch.argmax(preds, axis=-1)
                                  [i]] + f'\n[{class_names[targets[i]]}]')
            plt.axis('off')

        self.logger.experiment.add_figure(
            'predictions', fig, global_step=self.global_step)
