from math import ceil, floor

import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer


class QUESTDataset(Dataset):
    def __init__(self, df, mode, tokens, augment,
                 tokenizer_type, pretrained_model_name_or_path, do_lower_case,
                 LABEL_COL, t_max_len, q_max_len, a_max_len, tqa_mode,
                 TBSEP, pos_id_type, MAX_SEQUENCE_LENGTH=None,
                 use_category=True, logger=None):
        self.mode = mode
        self.augment = augment
        self.len = len(df)
        self.t_max_len = t_max_len
        self.q_max_len = q_max_len
        self.a_max_len = a_max_len
        self.tqa_mode = tqa_mode
        self.TBSEP = TBSEP
        self.pos_id_type = pos_id_type
        if MAX_SEQUENCE_LENGTH:
            self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        else:
            raise NotImplementedError
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
            self.labels = pd.DataFrame([[-1] * len(LABEL_COL)] * len(df))
        else:  # train or valid
            self.labels = df[LABEL_COL]

        if tokenizer_type == 'bert':
            tokenizer = BertTokenizer
        elif tokenizer_type == 'roberta':
            tokenizer = RobertaTokenizer
        elif tokenizer_type == 'xlnet':
            tokenizer = XLNetTokenizer
        else:
            raise NotImplementedError
        self.tokenizer = tokenizer.from_pretrained(
            pretrained_model_name_or_path, do_lower_case=do_lower_case)
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

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # change online preprocess or off line preprocess
        idx_row = self.original_df.iloc[idx].copy()
        idx_row = self.__preprocess_text_row(idx_row,
                                             t_max_len=self.t_max_len,
                                             q_max_len=self.q_max_len,
                                             a_max_len=self.a_max_len)
        input_ids = idx_row['input_ids'].squeeze()
        token_type_ids = idx_row['token_type_ids'].squeeze()
        attention_mask = idx_row['attention_mask'].squeeze()
        qa_id = idx_row['qa_id'].squeeze()

        if self.pos_id_type == 'arange':
            position_ids = torch.arange(self.MAX_SEQUENCE_LENGTH)
        # elif self.pos_id_type == 'tq_a_sep':
        #     position_ids = torch.cat([
        #         torch.arange(self.t_max_len+self.q_max_len+3),
        #         torch.arange(self.q_max_len+1)])
        # elif self.pos_id_type == 't_q_sep':
        #     position_ids = torch.cat([
        #         torch.arange(self.t_max_len+self.q_max_len+3),
        #         torch.arange(self.q_max_len+1)])
        # elif self.pos_id_type == 't_a_sep':
        #     position_ids = torch.cat(torch.arange(self.t_max_len+self.q_max_len+3)
        #             self.MAX_SEQUENCE_LENGTH)
        else:
            raise NotImplementedError

        labels = self.labels.iloc[idx].values
        return qa_id, input_ids, attention_mask, \
            token_type_ids, position_ids, labels

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

            if t_new_len + a_new_len + q_new_len + 4 != self.MAX_SEQUENCE_LENGTH:
                raise ValueError("New sequence length should be %d, but is %d"
                                 % (self.MAX_SEQUENCE_LENGTH,
                                     (t_new_len + a_new_len + q_new_len + 4)))
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
        title = self.tokenizer.tokenize(row.question_title)
        body = self.tokenizer.tokenize(row.question_body)
        answer = self.tokenizer.tokenize(row.answer)
        category = ('CAT_' + row.category).casefold()

        # category を text として入れてしまう !!!
        if self.use_category:
            title = [category] + title

        title, body, answer = self._trim_input(title, body, answer,
                                               t_max_len=t_max_len,
                                               q_max_len=q_max_len,
                                               a_max_len=a_max_len)

        if len(title) == 0:
            print(f'NO TITLE, qa_id: {qa_id}')
            title = ['_']
        if len(body) == 0:
            print(f'NO BODY, qa_id: {qa_id}')
            body = ['_']
        if len(answer) == 0:
            print(f'NO ANSWER, qa_id: {qa_id}')
            answer = ['_']

        if self.tqa_mode == 'tq_a':
            text = title + [self.TBSEP] + body
            text_pair = answer
        elif self.tqa_mode == 't_q':
            text = title
            text_pair = body
        elif self.tqa_mode == 't_a':
            text = title
            text_pair = answer

        encoded_texts_dict = self.tokenizer.encode_plus(
            text=text,
            text_pair=text_pair,
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
