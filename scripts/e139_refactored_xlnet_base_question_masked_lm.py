import pickle
import itertools
import os
import random
from logging import getLogger

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import optim
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
from transformers import XLNetModel, XLNetForMaskedLM

from refactor.datasets import QUESTDataset
from refactor.models import XLNetModelForBinaryMultiLabelClassifier
from refactor.utils import compute_spearmanr, test, train_one_epoch_ML, clean_data
from utils import (load_checkpoint, logInit, parse_args,
                   save_and_clean_for_prediction, save_checkpoint, sel_log,
                   send_line_notification)

EXP_ID = os.path.basename(__file__).split('_')[0]
MNT_DIR = './mnt'
DEVICE = 'cuda'
# DEVICE = 'cpu'
MODEL_PRETRAIN = 'xlnet-base-cased'
# MODEL_PRETRAIN = 'bert-base-cased-finetuned-mrpc'
# MODEL_CONFIG_PATH = './mnt/datasets/model_configs/bert-model-uncased-config.pkl'
MODEL_CONFIG_PATH = './mnt/datasets/model_configs/xlnet-model-base-cased-config.pkl'
TOKENIZER_TYPE = 'xlnet'
TOKENIZER_PRETRAIN = 'xlnet-base-cased'
BATCH_SIZE = 8
MAX_EPOCH = 6
MAX_SEQ_LEN = 512
T_MAX_LEN = 30
Q_MAX_LEN = 239 * 2
A_MAX_LEN = 239 * 0
DO_LOWER_CASE = True if MODEL_PRETRAIN == 'bert-base-uncased' else False
TQA_MODE = 'tq_a'


LABEL_COL = [
    'question_asker_intent_understanding',
    'question_body_critical',
    'question_conversational',
    'question_expect_short_answer',
    'question_fact_seeking',
    'question_has_commonly_accepted_answer',
    'question_interestingness_others',
    'question_interestingness_self',
    'question_multi_intent',
    'question_not_really_a_question',
    'question_opinion_seeking',
    'question_type_choice',
    'question_type_compare',
    'question_type_consequence',
    'question_type_definition',
    'question_type_entity',
    'question_type_instructions',
    'question_type_procedure',
    'question_type_reason_explanation',
    'question_type_spelling',
    'question_well_written',
    #    'answer_helpful',
    #    'answer_level_of_information',
    #    'answer_plausible',
    #    'answer_relevance',
    #    'answer_satisfaction',
    #    'answer_type_instructions',
    #    'answer_type_procedure',
    #    'answer_type_reason_explanation',
    #    'answer_well_written'
]


def seed_everything(seed=71):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()


def main(args, logger):
    # trn_df = pd.read_csv(f'{MNT_DIR}/inputs/origin/train.csv')
    trn_df = pd.read_pickle(f'{MNT_DIR}/inputs/nes_info/trn_df.pkl')
    tst_df = pd.read_csv(f'{MNT_DIR}/inputs/origin/test.csv')
    trn_df = pd.concat([trn_df, tst_df], axis=0).fillna(-1)
    trn_df['is_original'] = 1
    # raw_pseudo_df = pd.read_csv('./mnt/inputs/pseudos/top2_e078_e079_e080_e081_e082_e083/raw_pseudo_tst_df.csv')
    # half_opt_pseudo_df = pd.read_csv('./mnt/inputs/pseudos/top2_e078_e079_e080_e081_e082_e083/half_opt_pseudo_tst_df.csv')
    # opt_pseudo_df = pd.read_csv('./mnt/inputs/pseudos/top2_e078_e079_e080_e081_e082_e083/opt_pseudo_tst_df.csv')

    # clean texts
    # trn_df = clean_data(trn_df, ['question_title', 'question_body', 'answer'])

    # load additional tokens
    # with open('./mnt/inputs/nes_info/trn_over_10_vocab.pkl', 'rb') as fin:
    #     additional_tokens = pickle.load(fin)

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

    fold_best_metrics = []
    fold_best_metrics_raws = []
    for fold, (trn_idx, val_idx) in enumerate(gkf):
        if fold > 0:
            break
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
            trn_df = trn_df.sample(100, random_state=71)
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
        ]#  + additional_tokens
        fold_trn_df = trn_df.drop(['is_original', 'question_body_le'], axis=1)

        # fold_trn_df = pd.concat([fold_trn_df, raw_pseudo_df, opt_pseudo_df, half_opt_pseudo_df], axis=0)

        trn_dataset = QUESTDataset(
            df=fold_trn_df,
            mode='train',
            tokens=tokens,
            augment=[],
            tokenizer_type=TOKENIZER_TYPE,
            pretrained_model_name_or_path=TOKENIZER_PRETRAIN,
            do_lower_case=DO_LOWER_CASE,
            LABEL_COL=LABEL_COL,
            t_max_len=T_MAX_LEN,
            q_max_len=Q_MAX_LEN,
            a_max_len=A_MAX_LEN,
            tqa_mode=TQA_MODE,
            TBSEP='</s>',
            pos_id_type='arange',
            MAX_SEQUENCE_LENGTH=MAX_SEQ_LEN,
            use_category=False,
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
        model = XLNetForMaskedLM.from_pretrained(MODEL_PRETRAIN)

        optimizer = optim.Adam(model.parameters(), lr=3e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCH, eta_min=1e-5)

        # load checkpoint model, optim, scheduler
        if args.checkpoint and fold == loaded_fold:
            load_checkpoint(args.checkpoint, model, optimizer, scheduler)

        for epoch in tqdm(list(range(MAX_EPOCH))):
            if fold <= loaded_fold and epoch <= loaded_epoch:
                continue

            # model = DataParallel(model)
            model = model.to(DEVICE)
            trn_loss = train_one_epoch_ML(model, optimizer, trn_loader, DEVICE)

            scheduler.step()
            if fold in histories['trn_loss']:
                histories['trn_loss'][fold].append(trn_loss)
            else:
                histories['trn_loss'][fold] = [trn_loss, ]
            if fold in histories['val_loss']:
                histories['val_loss'][fold].append(trn_loss)
            else:
                histories['val_loss'][fold] = [trn_loss, ]
            if fold in histories['val_metric']:
                histories['val_metric'][fold].append(trn_loss)
            else:
                histories['val_metric'][fold] = [trn_loss, ]
            if fold in histories['val_metric_raws']:
                histories['val_metric_raws'][fold].append(trn_loss)
            else:
                histories['val_metric_raws'][fold] = [trn_loss, ]

            sel_log(
                f'fold : {fold} -- epoch : {epoch} -- '
                f'trn_loss : {float(trn_loss.detach().to("cpu").numpy()):.4f} -- ',
                logger)
            model = model.to('cpu')
            # model = model.module
            save_checkpoint(
                f'{MNT_DIR}/checkpoints/{EXP_ID}/{fold}',
                model,
                optimizer,
                scheduler,
                histories,
                [],
                [],
                [],
                fold,
                epoch,
                trn_loss,
                trn_loss,
                )
        save_and_clean_for_prediction(
            f'{MNT_DIR}/checkpoints/{EXP_ID}/{fold}',
            trn_dataset.tokenizer,
            clean=False)
        del model

    send_line_notification('fini!')

    sel_log('now saving best checkpoints...', logger)


if __name__ == '__main__':
    args = parse_args(None)
    log_file = f'{EXP_ID}.log'
    logger = getLogger(__name__)
    logger = logInit(logger, f'{MNT_DIR}/logs/', log_file)
    sel_log(f'args: {sorted(vars(args).items())}', logger)

    # send_line_notification(f' ------------- start {EXP_ID} ------------- ')
    main(args, logger)
