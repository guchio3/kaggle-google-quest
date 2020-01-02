import os
import sys
from logging import getLogger

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

from datasets import QUESTDataset
from metrics import compute_spearmanr, soft_binary_cross_entropy
from models.bert_model_binary_multilabel_classifier import \
    BertModelBinaryMultiLabelClassifier
from transformers import BertForMaskedLM, BertModel, BertTokenizer
from utils import (dec_timer, logInit, parse_args, load_checkpoint, save_checkpoint,
                   sel_log, send_line_notification)

EXP_ID = 'e001'
MNT_DIR = '../mnt'
DEVICE = 'cuda'
PRETRAIN = 'bert-base-uncased'
BATCH_SIZE = 24
MAX_EPOCH = 10


def train_one_epoch(model, fobj, optimizer, loader):
    model.train()

    running_loss = 0
    for (qa_id, input_ids, attention_mask,
         token_type_ids, labels) in tqdm(loader):
        # send them to DEVICE
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        # forward
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # mean for data parallel
        if len(outputs) > 1:
            loss = outputs[0].mean()
        else:
            loss = outputs[0]

        # backword and update
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # store loss to culc epoch mean
        running_loss += loss

    loss_mean = running_loss / len(loader)

    return loss_mean


def test(model, loader, tta=False):
    model.eval()

    with torch.no_grad():
        y_preds, y_trues, qa_ids = [], [], []

        running_loss = 0
        for (qa_id, input_ids, attention_mask,
             token_type_ids, labels) in tqdm(loader):
            # send them to DEVICE
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            # forward
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            # mean for data parallel
            if len(outputs) > 1:
                loss = outputs[0].mean()
            else:
                loss = outputs[0]
            logits = outputs[1]

            running_loss += loss

            y_preds.append(nn.functional.sigmoid(logits))
            y_trues.append(labels)
            qa_ids.append(qa_id)

        loss_mean = running_loss / len(loader)

        y_preds = torch.cat(y_preds).to('cpu').numpy()
        y_trues = torch.cat(y_trues).to('cpu').numpy()
        qa_ids = torch.cat(qa_ids).to('cpu').numpy()

        metric = compute_spearmanr(y_trues, y_preds)

    return loss_mean, metric, y_preds, y_trues, qa_ids


def main(args, logger):
    trn_df = pd.read_csv(f'{MNT_DIR}/inputs/origin/train.csv')

    gkf = GroupKFold(
        n_splits=5).split(
        X=trn_df.question_body,
        groups=trn_df.question_body)

    histories = {
        'trn_loss': [],
        'val_loss': [],
        'val_metric': [],
    }
    loaded_fold = -1
    loaded_epoch = -1
    if args.checkpoint:
        histories, loaded_fold, loaded_epoch = load_checkpoint(args.checkpoint)
    for fold, (trn_idx, val_idx) in enumerate(gkf):
        if fold < loaded_fold:
            continue
        trn_qa_ids = trn_df.iloc[trn_idx].qa_id
        val_qa_ids = trn_df.iloc[val_idx].qa_id
        if args.debug:
            trn_qa_ids = trn_qa_ids.sample(300, random_state=71)
            val_qa_ids = val_qa_ids.sample(300, random_state=71)

        trn_dataset = QUESTDataset(
            mode='train',
            qa_ids=trn_qa_ids,
            augment=[],
            pretrained_model_name_or_path=PRETRAIN,
            data_path=f'{MNT_DIR}/inputs/origin/')
        trn_sampler = RandomSampler(data_source=trn_dataset)
        trn_loader = DataLoader(trn_dataset,
                                batch_size=BATCH_SIZE,
                                sampler=trn_sampler,
                                num_workers=os.cpu_count(),
                                worker_init_fn=lambda x: np.random.seed(),
                                drop_last=True,
                                pin_memory=True)
        val_dataset = QUESTDataset(
            mode='valid',
            qa_ids=val_qa_ids,
            augment=[],
            pretrained_model_name_or_path='bert-base-uncased',
            data_path=f'{MNT_DIR}/inputs/origin/')
        val_sampler = RandomSampler(data_source=val_dataset)
        val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                sampler=val_sampler,
                                num_workers=os.cpu_count(),
                                worker_init_fn=lambda x: np.random.seed(),
                                drop_last=False,
                                pin_memory=True)

        fobj = soft_binary_cross_entropy
        model = BertModelBinaryMultiLabelClassifier(fobj=fobj,
                                                    num_labels=30,
                                                    pretrained_model_name_or_path=PRETRAIN
                                                    ).to(DEVICE)
        model = torch.nn.DataParallel(model)
        optimizer = optim.Adam(model.parameters(), lr=3e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCH, eta_min=1e-5)

        # load checkpoint model, optim, scheduler
        if args.checkpoint and fold == loaded_fold:
            load_checkpoint(args.checkpoint, model, optimizer, scheduler)

        for epoch in tqdm(list(range(MAX_EPOCH))):
            if fold <= loaded_fold and epoch <= loaded_epoch:
                continue
            trn_loss = train_one_epoch(model, fobj, optimizer, trn_loader)
            val_loss, val_metric, val_y_preds, val_y_trues, val_qa_ids = test(
                model, val_loader)

            scheduler.step()
            histories['trn_loss'].append(trn_loss)
            histories['val_loss'].append(val_loss)
            histories['val_metric'].append(val_metric)
            print(
                f'{trn_loss.detach().to("cpu").numpy()} -- '
                f'{val_loss.detach().to("cpu").numpy()} -- '
                f'{val_metric}',
                logger)
            save_checkpoint(
                f'{MNT_DIR}/checkpoints',
                EXP_ID,
                model,
                optimizer,
                scheduler,
                histories,
                val_y_preds,
                val_y_trues,
                val_qa_ids,
                fold,
                epoch,
                val_loss,
                val_metric)
        send_line_notification(f'finished fold {fold}')


if __name__ == '__main__':
    args = parse_args(None)
    log_file = f'{EXP_ID}.log'
    logger = getLogger(__name__)
    logger = logInit(logger, f'{MNT_DIR}/logs', log_file)
    sel_log(f'args: {sorted(vars(args).items())}', logger)

    main(args, logger)
