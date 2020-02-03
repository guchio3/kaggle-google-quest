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
from transformers import BertModel

from refactor.datasets import QUESTDataset
from refactor.models import BertModelForBinaryMultiLabelClassifier
from refactor.utils import compute_spearmanr, test, train_one_epoch
from utils import (load_checkpoint, logInit, parse_args,
                   save_and_clean_for_prediction, save_checkpoint, sel_log,
                   send_line_notification)

EXP_ID = os.path.basename(__file__).split('_')[0]
MNT_DIR = './mnt'
DEVICE = 'cuda'
MODEL_PRETRAIN = 'bert-base-uncased'
MODEL_CONFIG_PATH = './mnt/datasets/model_configs/bert-model-uncased-config.pkl'
TOKENIZER_TYPE = 'bert'
TOKENIZER_PRETRAIN = 'bert-base-uncased'
BATCH_SIZE = 8
MAX_EPOCH = 6
MAX_SEQ_LEN = 512


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
    trn_df['is_original'] = 1

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
            tokenizer_type=TOKENIZER_TYPE,
            pretrained_model_name_or_path=TOKENIZER_PRETRAIN,
            do_lower_case=True,
            LABEL_COL=LABEL_COL,
            t_max_len=30,
            q_max_len=239 * 2,
            a_max_len=239 * 0,
            tqa_mode='tq_a',
            TBSEP='[TBSEP]',
            pos_id_type='arange',
            MAX_SEQUENCE_LENGTH=MAX_SEQ_LEN,
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
            tokenizer_type=TOKENIZER_TYPE,
            pretrained_model_name_or_path=TOKENIZER_PRETRAIN,
            do_lower_case=True,
            LABEL_COL=LABEL_COL,
            t_max_len=30,
            q_max_len=239 * 2,
            a_max_len=239 * 0,
            tqa_mode='tq_a',
            TBSEP='[TBSEP]',
            pos_id_type='arange',
            MAX_SEQUENCE_LENGTH=MAX_SEQ_LEN,
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
        state_dict = BertModel.from_pretrained('MODEL_PRETRAIN').state_dict()
        model = BertModelForBinaryMultiLabelClassifier(num_labels=len(LABEL_COL),
                                                       config_path=MODEL_CONFIG_PATH,
                                                       state_dict=state_dict,
                                                       token_size=len(
                                                           trn_dataset.tokenizer),
                                                       MAX_SEQUENCE_LENGTH=MAX_SEQ_LEN,
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
            trn_loss = train_one_epoch(model, fobj, optimizer, trn_loader)
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
            trn_dataset.tokenizer,
            clean=False)
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
