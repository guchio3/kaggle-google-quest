import itertools
import os
import random
import sys
from logging import getLogger
from math import ceil, floor

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from torch import nn, optim
from torch.nn import (BCEWithLogitsLoss, DataParallel, MarginRankingLoss,
                      MSELoss)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM, BertModel, BertTokenizer
from transformers.modeling_bert import BertEmbeddings, BertLayer

from utils import (dec_timer, load_checkpoint, logInit, parse_args,
                   save_and_clean_for_prediction, save_checkpoint, sel_log,
                   send_line_notification)

# import nlpaug.augmenter.word as naw


EXP_ID = os.path.basename(__file__).split('_')[0]
MNT_DIR = './mnt'
DEVICE = 'cuda'
MODEL_PRETRAIN = 'bert-base-uncased'
# MODEL_CONFIG = 'bert-base-uncased'
TOKENIZER_PRETRAIN = 'bert-base-uncased'
BATCH_SIZE = 8
MAX_EPOCH = 6


LABEL_COL = [
    'question_asker_intent_understanding', 'question_body_critical',
    'question_conversational', 'question_expect_short_answer',
    'question_fact_seeking', 'question_has_commonly_accepted_answer',
    'question_interestingness_others', 'question_interestingness_self',
    'question_multi_intent', 'question_not_really_a_question',
    'question_opinion_seeking', 'question_type_choice',
    'question_type_compare', 'question_type_consequence',
    'question_type_definition', 'question_type_entity',
    'question_type_instructions', 'question_type_procedure',
    'question_type_reason_explanation', 'question_type_spelling',
    'question_well_written', 'answer_helpful',
    'answer_level_of_information', 'answer_plausible', 'answer_relevance',
    'answer_satisfaction', 'answer_type_instructions',
    'answer_type_procedure', 'answer_type_reason_explanation',
    'answer_well_written'
]


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
                 MAX_SEQUENCE_LENGTH=None, use_category=True, logger=None):
        self.mode = mode
        self.augment = augment
        self.len = len(df)
        self.TBSEP = TBSEP
        if MAX_SEQUENCE_LENGTH:
            self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        else:
            raise NotImplementedError
            # self.MAX_SEQUENCE_LENGTH = -1
            # for i, row in self.prep_df.iterrows():
            #     input_ids = row['input_ids'].squeeze()
            #     if self.MAX_SEQUENCE_LENGTH < len(input_ids):
            #         self.MAX_SEQUENCE_LENGTH = len(input_ids)
            # sel_log(f'calculated seq_len: {self.MAX_SEQUENCE_LENGTH}',
            # logger)
        self.use_category = use_category
        self.logger = logger
        self.cat_dict = {
            'CAT_TECHNOLOGY'.casefold(): 0,
            'CAT_STACKOVERFLOW'.casefold(): 1,
            'CAT_CULTURE'.casefold(): 2,
            'CAT_SCIENCE'.casefold(): 3,
            'CAT_LIFE_ARTS'.casefold(): 4,
        }

        if mode == "test":
            self.labels = pd.DataFrame([[-1] * 30] * len(df))
        else:  # train or valid
            # self.labels = df.iloc[:, 11:]
            self.labels = df[LABEL_COL]

        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path, do_lower_case=True)
        # self.tokenizer.add_special_tokens(
        #     {'additional_special_tokens': [self.TBSEP]})
        self.tokenizer.add_tokens([self.TBSEP])

        tokens = [token.encode('ascii', 'replace').decode()
                  for token in tokens if token != '']
        added_num = self.tokenizer.add_tokens(tokens)
        if logger:
            logger.info(f'additional_tokens : {added_num}')
        else:
            print(f'additional_tokens : {added_num}')
        # change online preprocess or off line preprocess
        self.original_df = df
        # res = self._preprocess_texts(df)
        # self.prep_df = df.merge(pd.DataFrame(res), on='qa_id', how='left')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # change online preprocess or off line preprocess
        # idx_row = self.prep_df.iloc[idx]
        idx_row = self.original_df.iloc[idx].copy()
        # idx_row = self._augment(idx_row)
        idx_row = self.__preprocess_text_row(idx_row,
                                             t_max_len=30,
                                             q_max_len=239,
                                             a_max_len=239)
        # t_max_len=30,
        # q_max_len=239,
        # a_max_len=239)
        # t_max_len=100,
        # q_max_len=700,
        # a_max_len=700)
        input_ids = idx_row['input_ids'].squeeze()
        token_type_ids = idx_row['token_type_ids'].squeeze()
        attention_mask = idx_row['attention_mask'].squeeze()
        qa_id = idx_row['qa_id'].squeeze()
        # cat_labels = idx_row['cat_label'].squeeze()
        cat_labels = -1
        position_ids = torch.arange(self.MAX_SEQUENCE_LENGTH)
        question_body_grp_count = idx_row['question_body_grp_count'].squeeze()

        labels = self.labels.iloc[idx].values
        return qa_id, input_ids, attention_mask, \
            token_type_ids, cat_labels, position_ids, question_body_grp_count, labels

    def _trim_input(self, title, question, answer,
                    t_max_len, q_max_len, a_max_len):

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

            # if t_new_len + a_new_len + q_new_len + 4 != self.MAX_SEQUENCE_LENGTH:
            #     raise ValueError("New sequence length should be %d, but is %d"
            #                      % (self.MAX_SEQUENCE_LENGTH,
            #                          (t_new_len + a_new_len + q_new_len + 4)))
            if len(title) > t_new_len:
                title = title[:t_new_len // 2] + title[-t_new_len // 2:]
            else:
                title = title[:t_new_len]
            if len(question) > q_new_len:
                question = question[:q_new_len // 2] + \
                    question[-q_new_len // 2:]
            else:
                question = question[:q_new_len]
            if len(answer) > a_new_len:
                answer = answer[:a_new_len // 2] + answer[-a_new_len // 2:]
            else:
                answer = answer[:a_new_len]
        return title, question, answer

    def __preprocess_text_row(self, row, t_max_len, q_max_len, a_max_len):
        qa_id = row.qa_id
#        title = self.tokenizer.tokenize(row.question_title)
#        body = self.tokenizer.tokenize(row.question_body)
#        answer = self.tokenizer.tokenize(row.answer.casefold())
        title = self.tokenizer.tokenize(row.question_title)
        body = self.tokenizer.tokenize(row.question_body)
        answer = self.tokenizer.tokenize(row.answer)
#        title = row.question_title.casefold()
#        body = row.question_body.casefold()
#        answer = row.answer.casefold()
#        category = row.category
        category = ('CAT_' + row.category).casefold()

        # category を text として入れてしまう !!!
        if self.use_category:
            title = [category] + title

        title, body, answer = self._trim_input(title, body, answer,
                                               t_max_len=t_max_len,
                                               q_max_len=q_max_len,
                                               a_max_len=a_max_len)

        title_and_body = title + [self.TBSEP] + body
        # title_and_body = title + f' {self.TBSEP} ' + body

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
        encoded_texts_dict['question_body_grp_count'] = row['question_body_grp_count']
        return encoded_texts_dict

    def _preprocess_texts(self, df):
        '''
        could be multi-processed if you need speeding up

        '''
        res = []
        for i, row in tqdm(list(df.iterrows())):
            res.append(self.__preprocess_text_row(row))
        return res

    # def _augment(self, row):
    #     if 'ins_ContextualWordEmbsAug' in self.augment:
    #         aug = naw.ContextualWordEmbsAug(
    #             model_path='bert-base-uncased', action="insert", device='cpu')
    #         row['question_title'] = aug.augment(row['question_title'])
    #         row['question_body'] = aug.augment(row['question_body'])
    #         row['answer'] = aug.augment(row['answer'])
    #     if 'sub_ContextualWordEmbsAug' in self.augment:
    #         aug = naw.ContextualWordEmbsAug(
    #             model_path='bert-base-uncased',
    #             action="substitute",
    #             device='cpu')
    #         row['question_title'] = aug.augment(row['question_title'])
    #         row['question_body'] = aug.augment(row['question_body'])
    #         row['answer'] = aug.augment(row['answer'])
    #     return row


# --- model ---
class BertModelForBinaryMultiLabelClassifier(nn.Module):
    def __init__(self, num_labels, pretrained_model_name_or_path=None,
                 cat_num=0, token_size=None, MAX_SEQUENCE_LENGTH=512):
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
            self._classifier = nn.Linear(
                self.model.pooler.dense.out_features + 1, self.model.pooler.dense.out_features)
            self.classifier = nn.Linear(
                self.model.pooler.dense.out_features, num_labels)
        self.add_module('fc_output', self.classifier)

        # resize
        if token_size:
            self.model.resize_token_embeddings(token_size)

        # define input embedding and transformers
        self.model.embeddings.position_embeddings = self._resize_embeddings(
            self.model.embeddings.position_embeddings, MAX_SEQUENCE_LENGTH)

        # use bertmodel as decoder
        # self.model.config.is_decoder = True

        # add modules
        # self.add_module('my_input_embeddings', self.input_embeddings)
        # self.add_module('my_input_bert_layer', self.input_bert_layer)
        # self.add_module('fc_output', self.classifier)

    def forward(self, input_ids=None, input_cats=None, labels=None, attention_mask=None,
                token_type_ids=None, position_ids=None, question_body_grp_count=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None):
        if self.catembedding:
            raise NotImplementedError
            # encoder_hidden_states = self.catembedding(input_cats)
            # encoder_hidden_states = self.catdropout(encoder_hidden_states)
            # encoder_hidden_states = self.catactivate(encoder_hidden_states)
        # if input_cats or inputs_embeds or encoder_hidden_states or encoder_attention_mask:
        #     raise NotImplementedError

        # embedding_output = self.input_embeddings(
        #     input_ids=input_ids,
        #     position_ids=position_ids,
        #     token_type_ids=token_type_ids,
        #     inputs_embeds=inputs_embeds)
        # layer_output = self.input_bert_layer(embedding_output)
        # inputs_embeds = layer_output[0]  # fit to bertmodel

        # outputs = self.model(input_ids=input_ids_2[:, :512],
        outputs = self.model(input_ids=input_ids,
                             # attention_mask=attention_mask[:, :512],
                             attention_mask=attention_mask,
                             # token_type_ids=token_type_ids[:, :512],
                             token_type_ids=token_type_ids,
                             position_ids=None,
                             head_mask=None,
                             # inputs_embeds=inputs_embeds[:, :512, :],
                             inputs_embeds=None,
                             # encoder_hidden_states=inputs_embeds,
                             encoder_hidden_states=None,
                             encoder_attention_mask=None)
        # pooled_output = outputs[1]
        pooled_output = torch.mean(outputs[0], dim=1)
        if self.catembeddingOut:
            outcat = self.catembeddingOut(input_cats)
            outcat = self.catactivateOut(outcat)
            pooled_output = torch.cat([pooled_output, outcat], -1)

        pooled_output = self.dropout(pooled_output)
        pooled_output = torch.cat(
            [pooled_output, question_body_grp_count.float().reshape(-1, 1)], dim=1)
        pooled_output = self._classifier(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.model.resize_token_embeddings(token_num)

    def freeze_unfreeze_bert(self, freeze=True, logger=None):
        if freeze:
            sel_log('FREEZE bert model !', logger)
            # for name, child in self.model.module.named_children():
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            sel_log('UNFREEZE bert model !', logger)
            # for name, child in self.model.module.named_children():
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = True

    def _resize_embeddings(self, old_embeddings, new_num_tokens):
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy,
                                   :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        return new_embeddings


# --- metrics ---
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
    return rhos


def train_one_epoch(model, fobj, optimizer, loader, pair_fobj=None):
    model.train()

    running_loss = 0
    for (qa_id, input_ids, attention_mask,
         token_type_ids, cat_labels, position_ids, question_body_grp_count, labels) in tqdm(loader):
        # send them to DEVICE
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        # cat_labels = cat_labels.to(DEVICE)
        position_ids = position_ids.to(DEVICE)
        question_body_grp_count = question_body_grp_count.to(DEVICE)
        labels = labels.to(DEVICE)

        # forward
        outputs = model(
            input_ids=input_ids,
            # input_cats=cat_labels,
            labels=labels,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            #            position_ids=position_ids
            question_body_grp_count=question_body_grp_count,
        )
        loss = fobj(outputs[0], labels.float())
        if pair_fobj:
            shuffle_idx = torch.randperm(len(labels))
            shuffled_logits = outputs[0][shuffle_idx]
            shuffled_pair_labels = (
                outputs[0] > shuffled_logits) + (outputs[0] <= shuffled_logits) * -1
            loss += pair_fobj(outputs[0],
                              shuffled_logits,
                              shuffled_pair_labels)

        # backword and update
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # store loss to culc epoch mean
        running_loss += loss

    loss_mean = running_loss / len(loader)

    return loss_mean


def test(model, fobj, loader, tta=False):
    model.eval()

    with torch.no_grad():
        y_preds, y_trues, qa_ids = [], [], []

        running_loss = 0
        for (qa_id, input_ids, attention_mask,
             token_type_ids, cat_labels, position_ids, question_body_grp_count, labels) in tqdm(loader):
            # send them to DEVICE
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            # cat_labels = cat_labels.to(DEVICE)
            position_ids = position_ids.to(DEVICE)
            question_body_grp_count = question_body_grp_count.to(DEVICE)
            labels = labels.to(DEVICE)

            # forward
            outputs = model(
                input_ids=input_ids,
                # input_cats=cat_labels,
                labels=labels,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                #                position_ids=position_ids
                question_body_grp_count=question_body_grp_count,
            )
            logits = outputs[0]
            loss = fobj(logits, labels.float())

            running_loss += loss

            y_preds.append(torch.sigmoid(logits))
            y_trues.append(labels)
            qa_ids.append(qa_id)

        loss_mean = running_loss / len(loader)

        y_preds = torch.cat(y_preds).to('cpu').numpy()
        y_trues = torch.cat(y_trues).to('cpu').numpy()
        qa_ids = torch.cat(qa_ids).to('cpu').numpy()

        metric_raws = compute_spearmanr(y_trues, y_preds)
        metric = np.mean(metric_raws)

    return loss_mean, metric, metric_raws, y_preds, y_trues, qa_ids


def main(args, logger):
    # trn_df = pd.read_csv(f'{MNT_DIR}/inputs/origin/train.csv')
    trn_df = pd.read_pickle(f'{MNT_DIR}/inputs/nes_info/trn_df.pkl')
    trn_df['question_body_grp_count'] = pd.read_pickle(
        './mnt/inputs/nes_info/question_body_grp_count.pkl').values
    trn_df['question_body_grp_count'] = (
        trn_df['question_body_grp_count'] - trn_df['question_body_grp_count'].mean()) / trn_df['question_body_grp_count'].std()
    trn_df['is_original'] = 1
    # aug_df = pd.read_pickle(f'{MNT_DIR}/inputs/nes_info/ContextualWordEmbsAug_sub_df.pkl')
    # aug_df['is_original'] = 0

    # trn_df = pd.concat([trn_df, aug_df], axis=0).reset_index(drop=True)

    gkf = GroupKFold(
        n_splits=5).split(
        X=trn_df.question_body,
        groups=trn_df.question_body_le,
    )

    histories = {
        'trn_loss': {},
        'val_loss': {},
        'val_metric': {},
        'val_metric_raws': {},
    }
    loaded_fold = -1
    loaded_epoch = -1
    if args.checkpoint:
        histories, loaded_fold, loaded_epoch = load_checkpoint(args.checkpoint)

    # calc max_seq_len using quest dataset
    # max_seq_len = QUESTDataset(
    #     df=trn_df,
    #     mode='train',
    #     tokens=[],
    #     augment=[],
    #     pretrained_model_name_or_path=TOKENIZER_PRETRAIN,
    # ).MAX_SEQUENCE_LENGTH
    # max_seq_len = 9458
    # max_seq_len = 1504
    max_seq_len = 512

    fold_best_metrics = []
    fold_best_metrics_raws = []
    for fold, (trn_idx, val_idx) in enumerate(gkf):
        if fold < loaded_fold:
            fold_best_metrics.append(np.max(histories["val_metric"][fold]))
            fold_best_metrics_raws.append(
                histories["val_metric_raws"][fold][np.argmax(histories["val_metric"][fold])])
            continue
        sel_log(
            f' --------------------------- start fold {fold} --------------------------- ', logger)
        fold_trn_df = trn_df.iloc[trn_idx]  # .query('is_original == 1')
        fold_trn_df = fold_trn_df.drop(
            ['is_original', 'question_body_le'], axis=1)
        # use only original row
        fold_val_df = trn_df.iloc[val_idx].query('is_original == 1')
        fold_val_df = fold_val_df.drop(
            ['is_original', 'question_body_le'], axis=1)
        if args.debug:
            fold_trn_df = fold_trn_df.sample(100, random_state=71)
            fold_val_df = fold_val_df.sample(100, random_state=71)
        temp = pd.Series(list(itertools.chain.from_iterable(
            fold_trn_df.question_title.apply(lambda x: x.split(' ')) +
            fold_trn_df.question_body.apply(lambda x: x.split(' ')) +
            fold_trn_df.answer.apply(lambda x: x.split(' '))
        ))).value_counts()
        tokens = temp[temp >= 10].index.tolist()
        # tokens = []
        tokens = [
            'CAT_TECHNOLOGY'.casefold(),
            'CAT_STACKOVERFLOW'.casefold(),
            'CAT_CULTURE'.casefold(),
            'CAT_SCIENCE'.casefold(),
            'CAT_LIFE_ARTS'.casefold(),
        ]

        trn_dataset = QUESTDataset(
            df=fold_trn_df,
            mode='train',
            tokens=tokens,
            augment=[],
            pretrained_model_name_or_path=TOKENIZER_PRETRAIN,
            MAX_SEQUENCE_LENGTH=max_seq_len,
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
            MAX_SEQUENCE_LENGTH=max_seq_len,
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
        # fobj = MSELoss()
        # pair_fobj = MarginRankingLoss()
        model = BertModelForBinaryMultiLabelClassifier(num_labels=30,
                                                       pretrained_model_name_or_path=MODEL_PRETRAIN,
                                                       # cat_num=5,
                                                       token_size=len(
                                                           trn_dataset.tokenizer),
                                                       MAX_SEQUENCE_LENGTH=max_seq_len,
                                                       )
        optimizer = optim.Adam(model.parameters(), lr=3e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCH, eta_min=1e-5)

        # load checkpoint model, optim, scheduler
        if args.checkpoint and fold == loaded_fold:
            load_checkpoint(args.checkpoint, model, optimizer, scheduler)

        for epoch in tqdm(list(range(MAX_EPOCH))):
            if fold <= loaded_fold and epoch <= loaded_epoch:
                continue
            if epoch < 1:
                model.freeze_unfreeze_bert(freeze=True, logger=logger)
            else:
                model.freeze_unfreeze_bert(freeze=False, logger=logger)
            model = DataParallel(model)
            model = model.to(DEVICE)
            trn_loss = train_one_epoch(
                model, fobj, optimizer, trn_loader)
            val_loss, val_metric, val_metric_raws, val_y_preds, val_y_trues, val_qa_ids = test(
                model, fobj, val_loader)

            scheduler.step()
            if fold in histories['trn_loss']:
                histories['trn_loss'][fold].append(trn_loss)
            else:
                histories['trn_loss'][fold] = [trn_loss, ]
            if fold in histories['val_loss']:
                histories['val_loss'][fold].append(val_loss)
            else:
                histories['val_loss'][fold] = [val_loss, ]
            if fold in histories['val_metric']:
                histories['val_metric'][fold].append(val_metric)
            else:
                histories['val_metric'][fold] = [val_metric, ]
            if fold in histories['val_metric_raws']:
                histories['val_metric_raws'][fold].append(val_metric_raws)
            else:
                histories['val_metric_raws'][fold] = [val_metric_raws, ]

            logging_val_metric_raws = ''
            for val_metric_raw in val_metric_raws:
                logging_val_metric_raws += f'{float(val_metric_raw):.4f}, '

            sel_log(
                f'fold : {fold} -- epoch : {epoch} -- '
                f'trn_loss : {float(trn_loss.detach().to("cpu").numpy()):.4f} -- '
                f'val_loss : {float(val_loss.detach().to("cpu").numpy()):.4f} -- '
                f'val_metric : {float(val_metric):.4f} -- '
                f'val_metric_raws : {logging_val_metric_raws}',
                logger)
            model = model.to('cpu')
            model = model.module
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
        fold_best_metrics.append(np.max(histories["val_metric"][fold]))
        fold_best_metrics_raws.append(
            histories["val_metric_raws"][fold][np.argmax(histories["val_metric"][fold])])
        save_and_clean_for_prediction(
            f'{MNT_DIR}/checkpoints/{EXP_ID}/{fold}',
            trn_dataset.tokenizer)
        del model

    # calc training stats
    fold_best_metric_mean = np.mean(fold_best_metrics)
    fold_best_metric_std = np.std(fold_best_metrics)
    fold_stats = f'{EXP_ID} : {fold_best_metric_mean:.4f} +- {fold_best_metric_std:.4f}'
    sel_log(fold_stats, logger)
    send_line_notification(fold_stats)

    fold_best_metrics_raws_mean = np.mean(fold_best_metrics_raws, axis=0)
    fold_raw_stats = ''
    for metric_stats_raw in fold_best_metrics_raws_mean:
        fold_raw_stats += f'{float(metric_stats_raw):.4f},'
    sel_log(fold_raw_stats, logger)
    send_line_notification(fold_raw_stats)

    sel_log('now saving best checkpoints...', logger)


if __name__ == '__main__':
    args = parse_args(None)
    log_file = f'{EXP_ID}.log'
    logger = getLogger(__name__)
    logger = logInit(logger, f'{MNT_DIR}/logs/', log_file)
    sel_log(f'args: {sorted(vars(args).items())}', logger)

    # send_line_notification(f' ------------- start {EXP_ID} ------------- ')
    main(args, logger)
