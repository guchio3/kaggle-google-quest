import torch
from torch import nn

from transformers import BertModel


class BertModelBinaryMultiLabelClassifier(nn.Module):
    def __init__(self, fobj, num_labels, pretrained_model_name_or_path=None):
        super(BertModelBinaryMultiLabelClassifier, self).__init__()
        if pretrained_model_name_or_path:
            self.model = BertModel.from_pretrained(
                pretrained_model_name_or_path)
        else:
            raise NotImplementedError
        self.fobj = fobj
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(
            self.model.pooler.dense.out_features, num_labels)
        self.add_module('fc_output', self.classifier)

    def forward(self, input_ids=None, labels=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)
        # pooled_output = outputs[1]
        pooled_output = torch.mean(outputs[0], dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss = self.fobj(logits, labels)
            outputs = (loss,) + outputs
        else:
            outputs = (None,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
