import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import BertTokenizer


class QUESTDataset(Dataset):
    def __init__(self, mode, qa_ids, augment, pretrained_model_name_or_path,
                 data_path='../../mnt/inputs/origin/', TBSEP='[TBSEP]',
                 MAX_SEQUENCE_LENGTH=512, logger=None):
        '''
        '''
        self.mode = mode
        self.augment = augment
        self.len = None
        self.TBSEP = TBSEP
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.logger = logger

        if mode == "test":
            df = pd.read_csv(f'{data_path}/test.csv')
            # dummy label
            self.labels = [[-1] * 30] * len(qa_ids)
        else:  # train or valid
            df = pd.read_csv(f'{data_path}/train.csv')\
                    .set_index('qa_id')\
                    .loc[qa_ids]\
                    .reset_index()
            self.labels = df.iloc[:, 11:]

        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path)
        res, res_tokenizer = self._preprocess_texts(df, tokenizer)
        self.tokenizer = res_tokenizer
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

    def __preprocess_text_row(self, row, tokenizer):
        qa_id = row.qa_id
        title = row.question_title
        body = row.question_body
        answer = row.answer

        encoded_texts_dict = tokenizer.encode_plus(
            text=title + f' {self.TBSEP} ' + body,
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

    def _preprocess_texts(self, df, tokenizer):
        '''
        could be multi-processed if you need speeding up

        '''
        tokenizer.add_special_tokens(
            {'additional_special_tokens': [self.TBSEP]})

        res = []
        for i, row in tqdm(list(df.iterrows())):
            res.append(self.__preprocess_text_row(row, tokenizer))

        return res, tokenizer
