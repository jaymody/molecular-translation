import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)

from layers import Encoder, DecoderWithAttention
from utils import get_score


# TODO: right now this setup will not work with ReduceLROnPLateau with the pl module
def get_scheduler(optimizer):
    return CosineAnnealingLR(optimizer, T_max=4, eta_min=1e-6, last_epoch=-1)


class ImageCaptioner(pl.LightningModule):
    def __init__(
        self,
        tokenizer,
        valid_labels,
        model_name,
        encoder_lr,
        decoder_lr,
        weight_decay,
        attention_dim,
        embed_dim,
        decoder_dim,
        dropout,
        max_len,
        gradient_accumulation_steps,
        max_grad_norm,
        device,
        **kwargs,
    ):
        super().__init__()

        # only save params of interest and kwargs
        self.save_hyperparameters(
            "model_name",
            "encoder_lr",
            "decoder_lr",
            "weight_decay",
            "attention_dim",
            "embed_dim",
            "decoder_dim",
            "dropout",
            "max_len",
            "gradient_accumulation_steps",
            "max_grad_norm",
            *kwargs.keys(),
        )

        self.tokenizer = tokenizer
        self.valid_labels = valid_labels
        self.to(device)

        self.encoder = Encoder(model_name, pretrained=True)
        self.encoder.to(device)
        self.decoder = DecoderWithAttention(
            attention_dim=attention_dim,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            vocab_size=len(tokenizer),
            dropout=dropout,
            device=self.device,
        )

        self.critereon = nn.CrossEntropyLoss(ignore_index=self.tokenizer.stoi["<pad>"])
        self.automatic_optimization = False

    def configure_optimizers(self):
        encoder_optimizer = Adam(
            self.encoder.parameters(),
            lr=self.hparams.encoder_lr,
            weight_decay=self.hparams.weight_decay,
            amsgrad=False,
        )

        decoder_optimizer = Adam(
            self.decoder.parameters(),
            lr=self.hparams.decoder_lr,
            weight_decay=self.hparams.weight_decay,
            amsgrad=False,
        )
        encoder_scheduler = get_scheduler(encoder_optimizer)
        decoder_scheduler = get_scheduler(decoder_optimizer)
        return [encoder_optimizer, decoder_optimizer], [
            encoder_scheduler,
            decoder_scheduler,
        ]

    def predict(self, images):
        with torch.no_grad():
            features = self.encoder(images)
            predictions = self.decoder.predict(
                features, self.hparams.max_len, self.tokenizer
            )

        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        text_preds = self.tokenizer.predict_captions(predicted_sequence)
        text_preds = [f"InChI=1S/{text}" for text in text_preds]

        return text_preds

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, labels, label_lengths = batch

        # forward pass
        features = self.encoder(images)
        preds, caps_sorted, decode_lengths, _, _ = self.decoder(
            features, labels, label_lengths
        )
        targets = caps_sorted[:, 1:]
        preds = pack_padded_sequence(preds, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = self.critereon(preds, targets)
        self.log("train_loss", loss, prog_bar=True)

        # normalize loss for gradient accumulation backwards pass
        self.manual_backward(loss / self.hparams.gradient_accumulation_steps)

        # run optimization
        if batch_idx % self.hparams.gradient_accumulation_steps == 0:
            # get optimizers
            encoder_optimizer, decoder_optimizer = self.optimizers()

            # clip gradients
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), self.hparams.max_grad_norm
            )
            decoder_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.decoder.parameters(), self.hparams.max_grad_norm
            )

            # perform optimizer step
            encoder_optimizer.step()
            decoder_optimizer.step()

            # clear gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch

        features = self.encoder(images)
        predictions = self.decoder.predict(
            features, self.hparams.max_len, self.tokenizer
        )

        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        text_preds = self.tokenizer.predict_captions(predicted_sequence)

        return text_preds

    def validation_epoch_end(self, outputs):
        outputs = np.concatenate(outputs)
        outputs = [f"InChI=1S/{text}" for text in outputs]
        score = get_score(self.valid_labels, outputs)
        self.log("val_score", score, prog_bar=True)
