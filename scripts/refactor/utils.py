import re

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm


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


def train_one_epoch(model, fobj, optimizer, loader, DEVICE, swa=False):
    model.train()

    running_loss = 0
    for (qa_id, input_ids, attention_mask,
         token_type_ids, position_ids, labels) in tqdm(loader):
        # send them to DEVICE
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        position_ids = position_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        # forward
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        loss = fobj(outputs[0], labels.float())

        # backword and update
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # store loss to culc epoch mean
        running_loss += loss
    if swa:
        print('now swa ing ...')
        optimizer.swap_swa_sgd()
        optimizer.bn_update(loader, model)

    loss_mean = running_loss / len(loader)

    return loss_mean


def test(model, fobj, loader, DEVICE, mode):
    model.eval()

    with torch.no_grad():
        y_preds, y_trues, qa_ids = [], [], []

        running_loss = 0
        for (qa_id, input_ids, attention_mask,
             token_type_ids, position_ids, labels) in tqdm(loader):
            # send them to DEVICE
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            position_ids = position_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            # forward
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids
            )
            logits = outputs[0]
            if mode != 'test':
                loss = fobj(logits, labels.float())

                running_loss += loss

            y_preds.append(torch.sigmoid(logits))
            y_trues.append(labels)
            qa_ids.append(qa_id)

        loss_mean = running_loss / len(loader)

        y_preds = torch.cat(y_preds).to('cpu').numpy()
        y_trues = torch.cat(y_trues).to('cpu').numpy()
        qa_ids = torch.cat(qa_ids).to('cpu').numpy()

        if mode == 'valid':
            metric_raws = compute_spearmanr(y_trues, y_preds)
            metric = np.mean(metric_raws)
        elif mode != 'test':
            raise NotImplementedError
        else:
            metric_raws = None
            metric = None

    return loss_mean, metric, metric_raws, y_preds, y_trues, qa_ids


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\n', '\xa0', '\t',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
mispell_dict = {"aren't": "are not",
                "can't": "cannot",
                "couldn't": "could not",
                "couldnt": "could not",
                "didn't": "did not",
                "doesn't": "does not",
                "doesnt": "does not",
                "don't": "do not",
                "hadn't": "had not",
                "hasn't": "has not",
                "haven't": "have not",
                "havent": "have not",
                "he'd": "he would",
                "he'll": "he will",
                "he's": "he is",
                "i'd": "I would",
                #                "i'd": "I had",
                "i'll": "I will",
                "i'm": "I am",
                "isn't": "is not",
                "it's": "it is",
                "it'll": "it will",
                "i've": "I have",
                "let's": "let us",
                "mightn't": "might not",
                "mustn't": "must not",
                "shan't": "shall not",
                "she'd": "she would",
                "she'll": "she will",
                "she's": "she is",
                "shouldn't": "should not",
                "shouldnt": "should not",
                "that's": "that is",
                "thats": "that is",
                "there's": "there is",
                "theres": "there is",
                "they'd": "they would",
                "they'll": "they will",
                "they're": "they are",
                "theyre": "they are",
                "they've": "they have",
                "we'd": "we would",
                "we're": "we are",
                "weren't": "were not",
                "we've": "we have",
                "what'll": "what will",
                "what're": "what are",
                "what's": "what is",
                "what've": "what have",
                "where's": "where is",
                "who'd": "who would",
                "who'll": "who will",
                "who're": "who are",
                "who's": "who is",
                "who've": "who have",
                "won't": "will not",
                "wouldn't": "would not",
                "you'd": "you would",
                "you'll": "you will",
                "you're": "you are",
                "you've": "you have",
                "'re": " are",
                "wasn't": "was not",
                "we'll": " will",
                "didn't": "did not",
                "tryin'": "trying"}


def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


def replace_typical_misspell(text):
    mispellings, mispellings_re = _get_mispell(mispell_dict)

    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


def clean_data(df, columns: list):
    for col in columns:
        df[col] = df[col].apply(lambda x: clean_numbers(x))
        df[col] = df[col].apply(lambda x: clean_text(x.lower()))
        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))

    return df


def train_one_epoch2(model, fobj, optimizer, loader, DEVICE, swa=False):
    model.train()

    running_loss = 0
    for (q_qa_id, q_input_ids, q_attention_mask,
         q_token_type_ids, q_position_ids, q_labels, a_input_ids, a_attention_mask,
         a_token_type_ids, a_position_ids, a_labels) in tqdm(loader):
        # send them to DEVICE
        q_input_ids = q_input_ids.to(DEVICE)
        q_attention_mask = q_attention_mask.to(DEVICE)
        q_token_type_ids = q_token_type_ids.to(DEVICE)
        q_position_ids = q_position_ids.to(DEVICE)
        a_input_ids = a_input_ids.to(DEVICE)
        a_attention_mask = a_attention_mask.to(DEVICE)
        a_token_type_ids = a_token_type_ids.to(DEVICE)
        a_position_ids = a_position_ids.to(DEVICE)
        labels = torch.cat([q_labels, a_labels], dim=1).to(DEVICE)

        # forward
        outputs = model(
            q_input_ids=q_input_ids,
            q_labels=q_labels,
            q_attention_mask=q_attention_mask,
            q_token_type_ids=q_token_type_ids,
            q_position_ids=q_position_ids,
            a_input_ids=a_input_ids,
            a_labels=a_labels,
            a_attention_mask=a_attention_mask,
            a_token_type_ids=a_token_type_ids,
            a_position_ids=a_position_ids
        )
        loss = fobj(outputs[0], labels.float())

        # backword and update
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # store loss to culc epoch mean
        running_loss += loss
    if swa:
        print('now swa ing ...')
        optimizer.swap_swa_sgd()
        optimizer.bn_update(loader, model)

    loss_mean = running_loss / len(loader)

    return loss_mean


def test(model, fobj, loader, DEVICE, mode):
    model.eval()

    with torch.no_grad():
        y_preds, y_trues, qa_ids = [], [], []

        running_loss = 0
        for (q_qa_id, q_input_ids, q_attention_mask,
             q_token_type_ids, q_position_ids, q_labels, a_input_ids, a_attention_mask,
             a_token_type_ids, a_position_ids, a_labels) in tqdm(loader):
            # send them to DEVICE
            q_input_ids = q_input_ids.to(DEVICE)
            q_attention_mask = q_attention_mask.to(DEVICE)
            q_token_type_ids = q_token_type_ids.to(DEVICE)
            q_position_ids = q_position_ids.to(DEVICE)
            a_input_ids = a_input_ids.to(DEVICE)
            a_attention_mask = a_attention_mask.to(DEVICE)
            a_token_type_ids = a_token_type_ids.to(DEVICE)
            a_position_ids = a_position_ids.to(DEVICE)
            labels = torch.cat([q_labels, a_labels], dim=1).to(DEVICE)
            qa_id = q_qa_id

            # forward
            outputs = model(
                q_input_ids=q_input_ids,
                q_labels=q_labels,
                q_attention_mask=q_attention_mask,
                q_token_type_ids=q_token_type_ids,
                q_position_ids=q_position_ids,
                a_input_ids=a_input_ids,
                a_labels=a_labels,
                a_attention_mask=a_attention_mask,
                a_token_type_ids=a_token_type_ids,
                a_position_ids=a_position_ids
            )
            logits = outputs[0]
            if mode != 'test':
                loss = fobj(logits, labels.float())

                running_loss += loss

            y_preds.append(torch.sigmoid(logits))
            y_trues.append(labels)
            qa_ids.append(qa_id)

        loss_mean = running_loss / len(loader)

        y_preds = torch.cat(y_preds).to('cpu').numpy()
        y_trues = torch.cat(y_trues).to('cpu').numpy()
        qa_ids = torch.cat(qa_ids).to('cpu').numpy()

        if mode == 'valid':
            metric_raws = compute_spearmanr(y_trues, y_preds)
            metric = np.mean(metric_raws)
        elif mode != 'test':
            raise NotImplementedError
        else:
            metric_raws = None
            metric = None

    return loss_mean, metric, metric_raws, y_preds, y_trues, qa_ids
