import itertools
import os
import sys
from logging import getLogger
from math import floor, ceil
import random

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from torch import nn, optim
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

from transformers import BertForMaskedLM, BertModel, BertTokenizer
from utils import (dec_timer, load_checkpoint, logInit, parse_args,
                   save_and_clean_for_prediction, save_checkpoint, sel_log,
                   send_line_notification)

EXP_ID = os.path.basename(__file__).split('_')[0]
MNT_DIR = './mnt'
DEVICE = 'cuda'
MODEL_PRETRAIN = 'bert-base-uncased'
TOKENIZER_PRETRAIN = 'bert-base-uncased'
BATCH_SIZE = 10
MAX_EPOCH = 6


def seed_everything(seed=71):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()


# --- dataset ---
class QUESTDataset(Dataset):
    def __init__(self, df, mode, tokens, augment,
                 pretrained_model_name_or_path, TBSEP='[TBSEP]',
                 MAX_SEQUENCE_LENGTH=512, logger=None):
        self.mode = mode
        self.augment = augment
        self.len = None
        self.TBSEP = TBSEP
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.logger = logger
        self.cat_dict = {
            'TECHNOLOGY': 0,
            'STACKOVERFLOW': 1,
            'CULTURE': 2,
            'SCIENCE': 3,
            'LIFE_ARTS': 4,
        }

        if mode == "test":
            self.labels = pd.DataFrame([[-1] * 30] * len(df))
        else:  # train or valid
            self.labels = df.iloc[:, 11:]

        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path)
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': [self.TBSEP]})

        tokens = [token.encode('ascii', 'replace').decode()
                  for token in tokens if token != '']
        added_num = self.tokenizer.add_tokens(tokens)
        if logger:
            logger.info(f'additional_tokens : {added_num}')
        else:
            print(f'additional_tokens : {added_num}')
        res = self._preprocess_texts(df)
        self.prep_df = df.merge(pd.DataFrame(res), on='qa_id', how='left')

    def __len__(self):
        return len(self.prep_df)

    def __getitem__(self, idx):
        idx_row = self.prep_df.iloc[idx]
        input_ids = idx_row['input_ids'].squeeze()
        token_type_ids = idx_row['token_type_ids'].squeeze()
        attention_mask = idx_row['attention_mask'].squeeze()
        qa_id = idx_row['qa_id'].squeeze()
        cat_labels = idx_row['cat_label'].squeeze()
        position_ids = torch.arange(self.MAX_SEQUENCE_LENGTH)

        labels = self.labels.iloc[idx].values
        return qa_id, input_ids, attention_mask, \
            token_type_ids, cat_labels, position_ids, labels

    def _trim_input(self, title, question, answer,
                    t_max_len=30, q_max_len=239, a_max_len=239):

        t_len = len(title)
        q_len = len(question)
        a_len = len(answer)

        if (t_len + q_len + a_len + 4) > self.MAX_SEQUENCE_LENGTH:
            if t_max_len > t_len:
                t_new_len = t_len
                a_max_len = a_max_len + floor((t_max_len - t_len) / 2)
                q_max_len = q_max_len + ceil((t_max_len - t_len) / 2)
            else:
                t_new_len = t_max_len

            if a_max_len > a_len:
                a_new_len = a_len
                q_new_len = q_max_len + (a_max_len - a_len)
            elif q_max_len > q_len:
                a_new_len = a_max_len + (q_max_len - q_len)
                q_new_len = q_len
            else:
                a_new_len = a_max_len
                q_new_len = q_max_len

            if t_new_len + a_new_len + q_new_len + 4 != self.MAX_SEQUENCE_LENGTH:
                raise ValueError("New sequence length should be %d, but is %d"
                                 % (self.MAX_SEQUENCE_LENGTH,
                                     (t_new_len + a_new_len + q_new_len + 4)))
            title = title[:t_new_len]
            question = question[:q_new_len]
            answer = answer[:a_new_len]
        return title, question, answer

    def __preprocess_text_row(self, row):
        qa_id = row.qa_id
#        title = self.tokenizer.tokenize(row.question_title)
#        body = self.tokenizer.tokenize(row.question_body)
#        answer = self.tokenizer.tokenize(row.answer.casefold())
#        title = self.tokenizer.tokenize(row.question_title)
#        body = self.tokenizer.tokenize(row.question_body)
#        answer = self.tokenizer.tokenize(row.answer)
        title = row.question_title.casefold()
        body = row.question_body.casefold()
        answer = row.answer.casefold()
        category = row.category

        title, body, answer = self._trim_input(title, body, answer)

#        title_and_body = title + [self.TBSEP] + body
        title_and_body = title + f' {self.TBSEP} ' + body

        encoded_texts_dict = self.tokenizer.encode_plus(
            text=title_and_body,
            text_pair=answer,
            add_special_tokens=True,
            max_length=self.MAX_SEQUENCE_LENGTH,
            pad_to_max_length=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
            return_overflowing_tokens=True,
        )
        encoded_texts_dict['qa_id'] = qa_id
        encoded_texts_dict['cat_label'] = self.cat_dict[category]
        return encoded_texts_dict

    def _preprocess_texts(self, df):
        '''
        could be multi-processed if you need speeding up

        '''
        res = []
        for i, row in tqdm(list(df.iterrows())):
            res.append(self.__preprocess_text_row(row))
        return res


# --- model ---
class BertModelForBinaryMultiLabelClassifier(nn.Module):
    def __init__(self, num_labels,
                 pretrained_model_name_or_path=None, cat_num=0):
        super(BertModelForBinaryMultiLabelClassifier, self).__init__()
        if pretrained_model_name_or_path:
            self.model = BertModel.from_pretrained(
                pretrained_model_name_or_path)
        else:
            raise NotImplementedError
        self.num_labels = num_labels
        if cat_num > 0:
            self.catembedding = nn.Embedding(cat_num, 768)
            self.catdropout = nn.Dropout(0.2)
            self.catactivate = nn.ReLU()

            self.catembeddingOut = nn.Embedding(cat_num, cat_num // 2 + 1)
            self.catactivateOut = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Linear(
                self.model.pooler.dense.out_features + cat_num // 2 + 1, num_labels)
        else:
            self.catembedding = None
            self.catdropout = None
            self.catactivate = None
            self.catembeddingOut = None
            self.catactivateOut = None
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Linear(
                self.model.pooler.dense.out_features, num_labels)
        self.add_module('fc_output', self.classifier)

    def forward(self, input_ids=None, input_cats=None, labels=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None):
        if self.catembedding:
            encoder_hidden_states = self.catembedding(input_cats)
            encoder_hidden_states = self.catdropout(encoder_hidden_states)
            encoder_hidden_states = self.catactivate(encoder_hidden_states)
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)
        # pooled_output = outputs[1]
        pooled_output = torch.mean(outputs[0], dim=1)
        if self.catembeddingOut:
            outcat = self.catembeddingOut(input_cats)
            outcat = self.catactivateOut(outcat)
            pooled_output = torch.cat([pooled_output, outcat], -1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.model.resize_token_embeddings(token_num)


# --- metrics ---
def soft_binary_cross_entropy(pred, soft_targets):
    L = -torch.sum((soft_targets * torch.log(nn.functional.sigmoid(pred)) +
                    (1. - soft_targets) * torch.log(nn.functional.sigmoid(1. - pred))), 1)
    return torch.mean(L)


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(
                col_trues,
                col_pred +
                np.random.normal(
                    0,
                    1e-7,
                    col_pred.shape[0])).correlation)
    return np.mean(rhos)


def train_one_epoch(model, fobj, optimizer, loader):
    model.train()

    running_loss = 0
    for (qa_id, input_ids, attention_mask,
         token_type_ids, cat_labels, position_ids, labels) in tqdm(loader):
        # send them to DEVICE
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
#        cat_labels = cat_labels.to(DEVICE)
        position_ids = position_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        # forward
        outputs = model(
            input_ids=input_ids,
#            input_cats=cat_labels,
            labels=labels,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
#            position_ids=position_ids
        )
        loss = soft_binary_cross_entropy(outputs[0], labels)

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
             token_type_ids, cat_labels, position_ids, labels) in tqdm(loader):
            # send them to DEVICE
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
#            cat_labels = cat_labels.to(DEVICE)
            position_ids = position_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            # forward
            outputs = model(
                input_ids=input_ids,
#                input_cats=cat_labels,
                labels=labels,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
#                position_ids=position_ids
            )
            logits = outputs[0]
            loss = soft_binary_cross_entropy(logits, labels)

            running_loss += loss

            y_preds.append(torch.sigmoid(logits))
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
        fold_trn_df = trn_df.iloc[trn_idx]
        fold_val_df = trn_df.iloc[val_idx]
        if args.debug:
            fold_trn_df = fold_trn_df.sample(100, random_state=71)
            fold_val_df = fold_val_df.sample(100, random_state=71)
        temp = pd.Series(list(itertools.chain.from_iterable(
            fold_trn_df.question_title.apply(lambda x: x.split(' ')) +
            fold_trn_df.question_body.apply(lambda x: x.split(' ')) +
            fold_trn_df.answer.apply(lambda x: x.split(' '))
        ))).value_counts()
        tokens = temp[temp >= 10].index.tolist()
        tokens = []

        trn_dataset = QUESTDataset(
            df=fold_trn_df,
            mode='train',
            tokens=tokens,
            augment=[],
            pretrained_model_name_or_path=TOKENIZER_PRETRAIN,
        )
        # update token
        trn_sampler = RandomSampler(data_source=trn_dataset)
        trn_loader = DataLoader(trn_dataset,
                                batch_size=BATCH_SIZE,
                                sampler=trn_sampler,
                                num_workers=os.cpu_count(),
                                worker_init_fn=lambda x: np.random.seed(),
                                drop_last=True,
                                pin_memory=True)
        val_dataset = QUESTDataset(
            df=fold_val_df,
            mode='valid',
            tokens=tokens,
            augment=[],
            pretrained_model_name_or_path=TOKENIZER_PRETRAIN,
        )
        val_sampler = RandomSampler(data_source=val_dataset)
        val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                sampler=val_sampler,
                                num_workers=os.cpu_count(),
                                worker_init_fn=lambda x: np.random.seed(),
                                drop_last=False,
                                pin_memory=True)

        fobj = BCEWithLogitsLoss()
        model = BertModelForBinaryMultiLabelClassifier(num_labels=30,
                                                       pretrained_model_name_or_path=MODEL_PRETRAIN,
                                                       )
        # model.resize_token_embeddings(len(trn_dataset.tokenizer))
        optimizer = optim.Adam(model.parameters(), lr=3e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCH, eta_min=1e-5)

        # load checkpoint model, optim, scheduler
        if args.checkpoint and fold == loaded_fold:
            load_checkpoint(args.checkpoint, model, optimizer, scheduler)

        for epoch in tqdm(list(range(MAX_EPOCH))):
            model = model.to(DEVICE)
            if fold <= loaded_fold and epoch <= loaded_epoch:
                continue
            trn_loss = train_one_epoch(model, fobj, optimizer, trn_loader)
            val_loss, val_metric, val_y_preds, val_y_trues, val_qa_ids = test(
                model, val_loader)

            scheduler.step()
            histories['trn_loss'].append(trn_loss)
            histories['val_loss'].append(val_loss)
            histories['val_metric'].append(val_metric)
            sel_log(
                f'epoch : {epoch} -- fold : {fold} -- '
                f'trn_loss : {float(trn_loss.detach().to("cpu").numpy()):.4f} -- '
                f'val_loss : {float(val_loss.detach().to("cpu").numpy()):.4f} -- '
                f'val_metric : {float(val_metric):.4f}',
                logger)
            model = model.to('cpu')
            save_checkpoint(
                f'{MNT_DIR}/checkpoints/{EXP_ID}/{fold}',
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
        save_and_clean_for_prediction(
            f'{MNT_DIR}/checkpoints/{EXP_ID}/{fold}',
            trn_dataset.tokenizer)
        del model
    sel_log('now saving best checkpoints...', logger)


if __name__ == '__main__':
    args = parse_args(None)
    log_file = f'{EXP_ID}.log'
    logger = getLogger(__name__)
    logger = logInit(logger, f'{MNT_DIR}/logs/', log_file)
    sel_log(f'args: {sorted(vars(args).items())}', logger)

    main(args, logger)
