import pickle

import torch
from torch import nn
from transformers import (BertConfig, BertModel, RobertaConfig, RobertaModel,
                          XLNetConfig, XLNetModel)


class BertModelForBinaryMultiLabelClassifier(nn.Module):
    def __init__(self, num_labels, config_path, state_dict,
                 token_size=None, MAX_SEQUENCE_LENGTH=512):
        super(BertModelForBinaryMultiLabelClassifier, self).__init__()
        with open(config_path, 'rb') as fin:
            config = pickle.load(fin)
        self.model = BertModel(config)
        self.model.load_state_dict(state_dict)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

        # resize
        if token_size:
            self.model.resize_token_embeddings(token_size)

        # add modules
        self.add_module('my_fc_output', self.classifier)

    def forward(self, input_ids=None, input_cats=None, labels=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None):

        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)

        pooled_output = torch.mean(outputs[0], dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.model.resize_token_embeddings(token_num)

    def freeze_unfreeze_bert(self, freeze=True, logger=None):
        if freeze:
            print('FREEZE bert model !', logger)
            # for name, child in self.model.module.named_children():
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            print('UNFREEZE bert model !', logger)
            # for name, child in self.model.module.named_children():
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = True

    # def _resize_embeddings(self, old_embeddings, new_num_tokens):
    #     old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    #     if old_num_tokens == new_num_tokens:
    #         return old_embeddings

    #     # Build new embeddings
    #     new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
    #     new_embeddings.to(old_embeddings.weight.device)

    #     # Copy word embeddings from the previous weights
    #     num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
    #     new_embeddings.weight.data[:num_tokens_to_copy,
    #                                :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

    #     return new_embeddings


class RobertaModelForBinaryMultiLabelClassifier(nn.Module):
    def __init__(self, num_labels, config_path, state_dict,
                 token_size=None, MAX_SEQUENCE_LENGTH=512):
        super(RobertaModelForBinaryMultiLabelClassifier, self).__init__()
        with open(config_path, 'rb') as fin:
            config = pickle.load(fin)
        self.model = RobertaModel(config)
        self.model.load_state_dict(state_dict)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

        # resize
        if token_size:
            self.model.resize_token_embeddings(token_size)

        self.model.embeddings.token_type_embeddings = self._resize_embeddings(
            self.model.embeddings.token_type_embeddings, 2)

        # add modules
        self.add_module('my_fc_output', self.classifier)

    def forward(self, input_ids=None, input_cats=None, labels=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None):

        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)

        pooled_output = torch.mean(outputs[0], dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.model.resize_token_embeddings(token_num)

    def freeze_unfreeze_bert(self, freeze=True, logger=None):
        if freeze:
            print('FREEZE bert model !', logger)
            # for name, child in self.model.module.named_children():
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            print('UNFREEZE bert model !', logger)
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


class XLNetModelForBinaryMultiLabelClassifier(nn.Module):
    def __init__(self, num_labels, config_path, state_dict,
                 token_size=None, MAX_SEQUENCE_LENGTH=512):
        super(XLNetModelForBinaryMultiLabelClassifier, self).__init__()
        with open(config_path, 'rb') as fin:
            config = pickle.load(fin)
        self.model = XLNetModel(config)
        self.model.load_state_dict(state_dict)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.model.config.d_model, num_labels)

        # resize
        if token_size:
            self.model.resize_token_embeddings(token_size)

        # add modules
        self.add_module('my_fc_output', self.classifier)

    def forward(self, input_ids=None, input_cats=None, labels=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None):

        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)

        pooled_output = torch.mean(outputs[0], dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.model.resize_token_embeddings(token_num)

    def freeze_unfreeze_bert(self, freeze=True, logger=None):
        if freeze:
            print('FREEZE bert model !', logger)
            # for name, child in self.model.module.named_children():
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            print('UNFREEZE bert model !', logger)
            # for name, child in self.model.module.named_children():
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = True

    # def _resize_embeddings(self, old_embeddings, new_num_tokens):
    #     old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    #     if old_num_tokens == new_num_tokens:
    #         return old_embeddings

    #     # Build new embeddings
    #     new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
    #     new_embeddings.to(old_embeddings.weight.device)

    #     # Copy word embeddings from the previous weights
    #     num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
    #     new_embeddings.weight.data[:num_tokens_to_copy,
    #                                :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

    #     return new_embeddings
