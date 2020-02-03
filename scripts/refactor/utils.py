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


def train_one_epoch(model, fobj, optimizer, loader, DEVICE):
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
