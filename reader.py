import pandas as pd
import numpy as np
from transformers import BertTokenizer
from . import config
from .std import *


class Sample:
    def __init__(self, ids, segment_ids, mask_ids, label=None):
        self.input_ids = ids
        self.segment_ids = segment_ids
        self.mask_ids = mask_ids
        self.label = label


def datapreprocessing(data, return_df=False):
    # Save all the questions, potential supporting evidence and indices in three lists
    textQ_to_be_tokenized = []
    textA_to_be_tokenized = []
    sp_index = []

    for dictionary in data['QUESTIONS']:
        for element in dictionary:
            textQ_to_be_tokenized.append(element['QTEXT_CN'])
            sp_index.append(element['SHINT_'])
    for dictionary in data['SENTS']:
        current_text_sentence = []
        for element in dictionary:
            current_text_sentence.append(element['text'])
        textA_to_be_tokenized.append(current_text_sentence)

    QandA_label = pd.DataFrame({'Question': textQ_to_be_tokenized,
                                'Sentence_List': textA_to_be_tokenized,
                                'SE_Index': sp_index,
                                'Label': sp_index})

    QandA_label['Length'] = QandA_label['Sentence_List'].apply(lambda x: len(x))
    QandA_label['SE_Index'] = QandA_label['SE_Index'].apply(lambda x: [0])
    QandA_label['SE_Index'] = QandA_label['SE_Index'] * QandA_label['Length']
    QandA_label['SE_Index'] = list(zip(QandA_label['SE_Index'], QandA_label['Label']))

    # Extract label index
    for row in QandA_label['SE_Index']:
        for index in row[1]:
            row[0][index] = 1

    indexed = [i[0] for i in list(QandA_label['SE_Index'])]
    QandA_label['Label'] = indexed

    if return_df:
        return QandA_label

    Q_and_Sentence_all_Comb = pd.DataFrame(
        {'Question': np.repeat(QandA_label['Question'].values, QandA_label['Sentence_List'].str.len()),
         'Sentence': np.concatenate(QandA_label['Sentence_List'].values)})
    Q_and_Sentence_all_Comb['Label'] = QandA_label['Label'].sum()

    tokenizer = BertTokenizer.from_pretrained(config.BERT_EMBEDDING)

    # Put all question and sentence combination into a list
    All_instances = []
    for i in range(len(QandA_label)):
        for sentence in QandA_label['Sentence_List'][i]:
            question_token = tokenizer.tokenize(QandA_label['Question'][i])
            sentence_token = tokenizer.tokenize(sentence)
            instance = ['[CLS]'] + question_token + ['[SEP]'] + sentence_token + ['[SEP]']
            if len(instance) > 512:
                instance = instance[:512]
            All_instances.append(instance)

    # Convert ids to segment_ids
    segment_ids = []
    for token in All_instances:
        length_of_zeros = token.index('[SEP]') - token.index('[CLS]') + 1
        length_of_ones = len(token) - length_of_zeros
        zeros_and_ones = [0] * length_of_zeros + [1] * length_of_ones
        segment_ids.append(zeros_and_ones)

    ids = []
    for token in All_instances:
        ids.append(tokenizer.convert_tokens_to_ids(token))

    mask_ids = []
    for token in All_instances:
        mask_ids.append([1] * len(token))

    labels = list(Q_and_Sentence_all_Comb['Label'])
    labels = [[i] for i in labels]

    return All_instances, ids, segment_ids, mask_ids, labels


if __name__=='__main__':
    validation_data = json_load(config.FGC_DEV)
    training_data = json_load(config.FGC_TRAIN)
    dev_instances, dev_ids, dev_seg_ids, dev_mask_ids, dev_labels = datapreprocessing(validation_data)
    train_instances, train_ids, train_seg_ids, train_mask_ids, train_labels = datapreprocessing(training_data)