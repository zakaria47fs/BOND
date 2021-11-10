import logging
import numpy as np
import itertools
from nltk.tokenize import word_tokenize 

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import RobertaConfig, RobertaTokenizer
from modeling_roberta import RobertaForTokenClassification_v2

from tqdm import tqdm

from data_utils import convert_examples_to_features,\
                        InputExample, convert_examples_to_features, get_labels


logger = logging.getLogger(__name__)


mode = "test"
max_seq_length = 128
pad_token_label_id = -100
model_type = 'roberta'
model_type = model_type.lower()
model_name_or_path = 'roberta-base'
data_dir = './dataset/CompanyFormsData/'
cache_dir = './pretrained_model/'
do_lower_case = True
output_dir = './outputs/companyforms/self_training/roberta_reinit0_begin900_period450_soft_hp5.9_10_1e-5/'
local_rank = -1
labels = get_labels(data_dir)


# NLTK tokenizer
def default_tokenizer(text):
    text_tokenized = [[word_tokenize(w), ' '] for w in text.split()]
    tokens_spaces = list(itertools.chain(*list(itertools.chain(*text_tokenized))))
    
    tokens = []
    spaces = []
    
    if not len(tokens_spaces):
        return tokens, spaces

    if text[-1]!=' ': tokens_spaces.pop(-1)

    for i in range (len(tokens_spaces)-1):
        token = tokens_spaces[i]
        next_token = tokens_spaces[i+1]

        if token!=' ':
            tokens.append(token)

            if next_token==' ': spaces.append(True)
            else: spaces.append(False)

    if tokens_spaces[-1]!=' ':
        tokens.append(tokens_spaces[-1])
        spaces.append(False)
        
    return tokens, spaces


def convert_sentence2modelinput(sentence, tokenizer):
    
    sentence_words, spaces = default_tokenizer(sentence)
    init_labels = [0]*len(sentence_words)
    hp_labels = [None]*len(sentence_words)

    sentence_example = InputExample(guid='XX', words=sentence_words, labels=init_labels,\
                                    hp_labels=hp_labels)
    examples = [sentence_example]
    
    features = convert_examples_to_features(
            examples,
            labels,
            max_seq_length,
            tokenizer,
            cls_token_at_end=bool(model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id,
        )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([[f.input_ids for f in features]], dtype=torch.long)
    all_input_mask = torch.tensor([[f.input_mask for f in features]], dtype=torch.long)
    all_segment_ids = torch.tensor([[f.segment_ids for f in features]], dtype=torch.long)
    all_label_ids = torch.tensor([[f.label_ids for f in features]], dtype=torch.long)
    all_full_label_ids = torch.tensor([[f.full_label_ids for f in features]], dtype=torch.long)
    all_hp_label_ids = torch.tensor([[f.hp_label_ids for f in features]], dtype=torch.long)

    all_ids = torch.tensor([[f for f in range(len(features))]], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_full_label_ids, all_hp_label_ids, all_ids)
    
    return dataset, sentence_words, spaces


def predict(sentence, model, tokenizer):
    
    eval_dataset, sentence_words, spaces = convert_sentence2modelinput(sentence, tokenizer)

    preds = None
    out_label_ids = None
    model.eval()
    
    test_sample = tuple(t.to(device) for t in eval_dataset[0])

    # deactivate torch autograde engine (when evaluating and predicting)
    with torch.no_grad():
        inputs = {"input_ids": test_sample[0], "attention_mask": test_sample[1]}
        inputs["token_type_ids"] = None # XLM and RoBERTa don"t use segment_ids
        outputs = model(**inputs)
        logits = outputs[0].detach().cpu().numpy()
            
    preds = np.argmax(logits, axis=2)
    out_label_ids = test_sample[3].detach().cpu().numpy()

    label_map = {i: label for i, label in enumerate(labels)}
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                preds_list[i].append(label_map[preds[i][j]])

    return sentence_words, spaces, preds_list[0]


if __name__=='__main__':

    import pandas as pd

    df_companies = pd.read_csv('data/companies shuffled.csv')
    companies = list(df_companies.name)

    config_class, model_class, tokenizer_class = (RobertaConfig, RobertaForTokenClassification_v2, RobertaTokenizer)

    tokenizer = tokenizer_class.from_pretrained(output_dir, do_lower_case=do_lower_case)
    model = model_class.from_pretrained(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sentence = 'Absolute Rehabilitation Corporation LL C'

    companies_name = []
    companies_unofficial_name = []

    for name in tqdm(companies[20000:20100]):
        sentence_words, spaces, preds_list = predict(sentence, model, tokenizer)
        
        unofficial_name = ''
        for i, word in enumerate(sentence_words):
            if preds_list[i]!='O': continue
            # or ... : added case when entity removed has a space after while previous word hadn't
            # => affect that space to the previous word
            if spaces[i] or (i<len(sentence_words)-1 and spaces[i+1]==True and preds_list[i+1]!='O'):
                unofficial_name += word + ' '
            else:
                unofficial_name += word

        companies_name.append(name)
        companies_unofficial_name.append(unofficial_name)

        break

    #df_result = pd.DataFrame({'name': companies_name, 'unofficial name': companies_unofficial_name})
    #df_result.to_csv('data/result.csv', index=False)
    print(sentence_words, spaces, preds_list)
