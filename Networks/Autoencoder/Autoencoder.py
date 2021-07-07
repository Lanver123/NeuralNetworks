import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class Autoencoder(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.encoder = nn.Sequential(
            nn.Linear(1 * 28 * 28, self.hparams['n_hidden'][0]),
            nn.ReLU(),
            nn.Linear(self.hparams['n_hidden'][0], self.hparams['embedding_dim'])
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams['embedding_dim'], self.hparams['n_hidden'][0]),
            nn.ReLU(),
            nn.Linear(self.hparams['n_hidden'][0], 1 * 28 * 28)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hparams['embedding_dim'], self.hparams['n_hidden'][0]),
            nn.ReLU(),
            nn.Linear(self.hparams['n_hidden'][0], 10))

    def switch_mode(self, mode):
        self.hparams['ae_mode'] = mode

    def freeze_encoder(self):
        self.encoder.freeze()

    def save_encoder(self):
        torch.save(self.encoder, 'pretrained_encoder.pt')

    def load_pretrained_encoder(self):
        self.encoder = torch.load('pretrained_encoder.pt')

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, x):
        # flatten the image  before sending as input to the model
        n = x.shape[0]
        x = x.view(n, -1)

        embedding = self.encoder(x)

        return embedding

    def general_step(self, batch, batch_idx, mode: str):
        images, targets = batch
        n = images.shape[0]
        images = images.view(n, -1)

        reconstruction, classification_scores, loss, predictions = None, None, None, None

        # Perform a forward pass on the network with inputs
        embedding = self.forward(images)
        if self.hparams['ae_mode'] == 'decoder':
            reconstruction = self.decoder(embedding)
            loss = F.mse_loss(reconstruction, images)
            self.log('{}_decoder_loss'.format(mode), loss)
        elif self.hparams['ae_mode'] == 'classification':
            classification_scores = self.classifier(embedding)
            loss = F.cross_entropy(classification_scores, targets)
            _, predictions = torch.max(classification_scores, 1)
            acc = predictions.eq(targets).sum().float() / targets.size(0)
            self.log('{}_classification_loss'.format(mode), loss)
            self.log('{}_classification_acc'.format(mode), acc)

        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "training")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = None
        if self.hparams['ae_mode'] == 'decoder':
            optimizer = self.hparams['optimizer'](list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                                  lr=self.hparams['learning_rate'])
        if self.hparams['ae_mode'] == 'classification':
            optimizer = self.hparams['optimizer'](self.classifier.parameters(),
                                                  lr=self.hparams['learning_rate'])

        return optimizer
