# all token classification model with crf head
from transformers import AutoModel, AutoModelForTokenClassification, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from .crf import ConditionalRandomField
from torch import nn


class PreTrainedModelForKpExtraction(PreTrainedModel):
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class AutoModelForKpExtraction(AutoModelForTokenClassification):
    pass


class AutoCrfModelforKpExtraction(PreTrainedModelForKpExtraction):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.base_layer = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = ConditionalRandomField(
            self.num_labels,
            label_encoding="BIO",
            id2label=self.config.id2label,  # TODO
            label2id=config.label2id,
            include_start_end_transitions=False,
        )
        self.post_init()
        # self.init_weights()

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.base_layer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss = -self.crf(logits, labels, attention_mask)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
