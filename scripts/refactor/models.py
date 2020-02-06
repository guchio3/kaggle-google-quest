import pickle

import torch
from torch import nn
from transformers import (BertConfig, BertModel, RobertaConfig, RobertaModel,
                          XLNetConfig, XLNetModel)


class BertModelForBinaryMultiLabelClassifier(nn.Module):
    def __init__(self, num_labels, config_path, state_dict,
                 cat_last_layer_num=1, last_bn=False,
                 token_size=None, MAX_SEQUENCE_LENGTH=512):
        super(BertModelForBinaryMultiLabelClassifier, self).__init__()
        with open(config_path, 'rb') as fin:
            config = pickle.load(fin)
        self.model = BertModel(config)
        self.model.load_state_dict(state_dict)
        self.classifier = nn.Linear(self.model.config.hidden_size*cat_last_layer_num, num_labels)

        self.cat_last_layer_num = cat_last_layer_num
        self.last_bn = last_bn
        if last_bn:
            self.bn = nn.BatchNorm1d(self.model.config.hidden_size*cat_last_layer_num)
        else:
            self.dropout = nn.Dropout(0.2)

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

        if self.cat_last_layer_num > 1:
            pooled_output = torch.cat(
                    [torch.mean(outputs[2][-i-1], dim=1) for i in range(self.cat_last_layer_num)],
                    dim=1)
        else:
            pooled_output = torch.mean(outputs[0], dim=1)

        if self.last_bn:
            pooled_output = self.bn(pooled_output)
        else:
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
        # # only for roberta
        # if self.model.state_dict()['embeddings.token_type_embeddings.weight'].shape[0] == 1:
        #     self.model.load_state_dict(state_dict)
        #     self.model.embeddings.token_type_embeddings = self._resize_embeddings(
        #         self.model.embeddings.token_type_embeddings, 2)
        # elif self.model.state_dict()['embeddings.token_type_embeddings.weight'].shape[0] == 2:
        #     self.model.embeddings.token_type_embeddings = self._resize_embeddings(
        #         self.model.embeddings.token_type_embeddings, 2)
        #     self.model.load_state_dict(state_dict)
        # else:
        #     raise NotImplementedError
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

        if self.cat_last_layer_num > 1:
            pooled_output = torch.cat(
                    [torch.mean(outputs[2][-i-1], dim=1) for i in range(self.cat_last_layer_num)],
                    dim=1)
        else:
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
                             # position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             # encoder_hidden_states=encoder_hidden_states,
                             # encoder_attention_mask=encoder_attention_mask
                             )

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


class BertModelForBinaryMultiLabelClassifier2(nn.Module):
    def __init__(self, num_labels, config_path, q_state_dict, a_state_dict,
                 token_size=None, MAX_SEQUENCE_LENGTH=512):
        super(BertModelForBinaryMultiLabelClassifier2, self).__init__()
        with open(config_path, 'rb') as fin:
            config = pickle.load(fin)
        self.q_model = BertModel(config)
        self.a_model = BertModel(config)
        self.q_model.load_state_dict(q_state_dict)
        self.a_model.load_state_dict(a_state_dict)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.q_model.config.hidden_size*2, num_labels)

        # resize
        if token_size:
            self.q_model.resize_token_embeddings(token_size)
            self.a_model.resize_token_embeddings(token_size)

        # add modules
        # self.add_module('q_model', self.q_model)
        # self.add_module('a_model', self.a_model)
        # self.add_module('my_fc_output', self.classifier)

    def forward(self, q_input_ids=None, q_input_cats=None, q_labels=None, q_attention_mask=None,
                q_token_type_ids=None, q_position_ids=None, q_head_mask=None,
                q_inputs_embeds=None, q_encoder_hidden_states=None,
                q_encoder_attention_mask=None, a_input_ids=None, a_input_cats=None, a_labels=None, a_attention_mask=None,
                a_token_type_ids=None, a_position_ids=None, a_head_mask=None,
                a_inputs_embeds=None, a_encoder_hidden_states=None,
                a_encoder_attention_mask=None):

        q_outputs = self.q_model(input_ids=q_input_ids,
                                 attention_mask=q_attention_mask,
                                 token_type_ids=q_token_type_ids,
                                 position_ids=q_position_ids,
                                 head_mask=q_head_mask,
                                 inputs_embeds=q_inputs_embeds,
                                 encoder_hidden_states=q_encoder_hidden_states,
                                 encoder_attention_mask=q_encoder_attention_mask)

        a_outputs = self.a_model(input_ids=a_input_ids,
                                 attention_mask=a_attention_mask,
                                 token_type_ids=a_token_type_ids,
                                 position_ids=a_position_ids,
                                 head_mask=a_head_mask,
                                 inputs_embeds=a_inputs_embeds,
                                 encoder_hidden_states=a_encoder_hidden_states,
                                 encoder_attention_mask=a_encoder_attention_mask)

        q_pooled_output = torch.mean(q_outputs[0], dim=1)
        a_pooled_output = torch.mean(a_outputs[0], dim=1)
        pooled_output = torch.cat([q_pooled_output, a_pooled_output], dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,)  # + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.q_model.resize_token_embeddings(token_num)
        self.a_model.resize_token_embeddings(token_num)

    def freeze_unfreeze_bert(self, freeze=True, logger=None):
        if freeze:
            print('FREEZE bert model !', logger)
            # for name, child in self.model.module.named_children():
            for name, child in self.q_model.named_children():
                for param in child.parameters():
                    param.requires_grad = False
            for name, child in self.a_model.named_children():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            print('UNFREEZE bert model !', logger)
            # for name, child in self.model.module.named_children():
            for name, child in self.q_model.named_children():
                for param in child.parameters():
                    param.requires_grad = True
            for name, child in self.a_model.named_children():
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
