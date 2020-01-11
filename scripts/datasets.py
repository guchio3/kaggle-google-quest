import os
from functools import partial
from multiprocessing import Pool
from math import floor, ceil

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import BertTokenizer

# def preprocess_text_row(row_pair, tokenizer, MAX_SEQUENCE_LENGTH, TBSEP):
#    i, row = row_pair
#
#    qa_id = row.qa_id
#    title = row.question_title.casefold().strip().strip(
#        '()[]{}.,;:-?!\"\'\n').encode('ascii', 'replace').decode()
#    body = row.question_body.casefold().strip().strip(
#        '()[]{}.,;:-?!\"\'\n').encode('ascii', 'replace').decode()
#    answer = row.answer.casefold().strip().strip(
#        '()[]{}.,;:-?!\"\'\n').encode('ascii', 'replace').decode()
#
#    encoded_texts_dict = tokenizer.encode_plus(
#        text=title + f' {TBSEP} ' + body,
#        # text=title + f' [SEP] ' + body,
#        text_pair=answer,
#        add_special_tokens=True,
#        max_length=MAX_SEQUENCE_LENGTH,
#        pad_to_max_length=True,
#        return_tensors='pt',
#        return_token_type_ids=True,
#        return_attention_mask=True,
#        return_overflowing_tokens=True,
#    )
#    encoded_texts_dict['qa_id'] = qa_id
#    return encoded_texts_dict


class QUESTDataset(Dataset):
    def __init__(self, df, mode, tokens, augment, pretrained_model_name_or_path,
                 trim=False, TBSEP='[TBSEP]',
                 MAX_SEQUENCE_LENGTH=512, logger=None):
        '''
        '''
        self.mode = mode
        self.augment = augment
        self.len = None
        self.trim = trim
        self.TBSEP = TBSEP
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.logger = logger

        if mode == "test":
            self.labels = pd.DataFrame([[-1] * 30] * len(df.qa_id))
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

        labels = self.labels.iloc[idx].values
        return qa_id, input_ids, attention_mask, token_type_ids, labels

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
        title = row.question_title.casefold().strip().strip(
            '()[]{}.,;:-?!\"\'\n').encode('ascii', 'replace').decode()
        body = row.question_body.casefold().strip().strip(
            '()[]{}.,;:-?!\"\'\n').encode('ascii', 'replace').decode()
        answer = row.answer.casefold().strip().strip(
            '()[]{}.,;:-?!\"\'\n').encode('ascii', 'replace').decode()

        if self.trim:
            title, body, answer = self._trim_input(title, body, answer)

        encoded_texts_dict = self.tokenizer.encode_plus(
            # text=title + f' {self.TBSEP} ' + body,
            text=title + f' [SEP] ' + body,
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
        return encoded_texts_dict

    def _preprocess_texts(self, df):
        '''
        could be multi-processed if you need speeding up

        '''
        res = []
        for i, row in tqdm(list(df.iterrows())):
            res.append(self.__preprocess_text_row(row))
#        with Pool(os.cpu_count()) as p:
#            iter_func = partial(
#                preprocess_text_row,
#                tokenizer=self.tokenizer,
#                MAX_SEQUENCE_LENGTH=self.MAX_SEQUENCE_LENGTH,
#                TBSEP=self.TBSEP)
#            imap = p.imap_unordered(iter_func, df.iterrows())
#            res = list(tqdm(imap, total=len(df)))
#            p.close()
#            p.join()

        return res
