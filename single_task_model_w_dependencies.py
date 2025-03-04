from transformers import PreTrainedModel
from tqdm.auto import tqdm
from utils_generic import text_to_num, task_to_num
import matplotlib.pyplot as plt

from transformers import get_scheduler
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import numpy as np


from transformers import AutoModel


import torch
from torch import nn

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


#######################################################################################

# CREATE DATALOADER

#######################################################################################


class DatasetSingleTaskDep(Dataset):
    def __init__(self, data, task, eval):
        self.data = data
        self.task = task
        self.eval = eval

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        x = torch.tensor(self.data[index]['input_ids'])

        attention = torch.tensor(self.data[index]['attention_mask'])

        # Etiquetas de las dimensiones correspondientes

        raw_label = self.data[index]['labels'][self.task]

        if len(raw_label) > 1:
            label = np.random.choice(raw_label)
            if label == 2:
                label = np.random.choice([0, 1])
        elif len(raw_label) == 1:
            if raw_label[0] == 2:
                label = np.random.choice([0, 1])
            else:
                label = raw_label[0]
        elif len(raw_label) == 0 and self.eval == True:
            label = 2

        dep_tags = []
        for item in self.data[index]['dep_tags']:
            dep_tags.append(self.data[index]['dep_tags'][item])

        sample = {'input_ids': x,
                  'attention_mask': attention,
                  'dep_tags': torch.tensor(dep_tags),
                  'label': label,
                  'tasks': self.task,
                  # Esto es porque necesito el dato para collate_fn
                  'num_vocabs': len(dep_tags)
                  }

        return sample

    def __len__(self):
        return len(self.data)


def collate_fn(batch):

    input_ids = [b['input_ids'] for b in batch]
    input_ids = pad_sequence(input_ids, batch_first=True)

    label = torch.tensor([b['label'] for b in batch])

    attention = [b['attention_mask'] for b in batch]
    attention_mask = pad_sequence(attention, batch_first=True)

    num_vocabs = batch[0]['num_vocabs']
    batch_size = len(batch)

    dep_tags = [b['dep_tags'] for b in batch]

    aux_dep = {}
    for j in range(num_vocabs):
        aux_dep[j] = [dep_tags[i][j] for i in range(batch_size)]
        aux_dep[j] = pad_sequence(aux_dep[j], batch_first=True)

    deps = [aux_dep[i].tolist() for i in range(num_vocabs)]

    batched_input = {'input_ids': input_ids, 'attention_mask': attention_mask,
                     'dep_tags': torch.tensor(deps), 'label': label}

    return batched_input


#######################################################################################

# MODELS

#######################################################################################


class LSTM_enc(nn.Module):
    """Modelo encargado de crear los embeddings para los tags de dependencia y contextualizarlos
    mediante el uso de una red LSTM.
    Funciona independientemente del número de items en los que separemos los dependency tags"""

    def __init__(self, embedding_dim, hidden_dim, vocab):
        """vocab: objeto de la clase Vocabulary"""
        super(LSTM_enc, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_sizes = vocab.get_vocab_sizes()
        self.num_vocabs = len(self.vocab_sizes)

        self.emb_layers = nn.ModuleList([])
        for i in range(self.num_vocabs):
            self.emb_layers.append(nn.Embedding(
                self.vocab_sizes[i], embedding_dim))

        self.lstm = nn.LSTM(embedding_dim * self.num_vocabs,
                            hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, dep_tags):

        embeds = []

        for i in range(len(dep_tags)):

            e = self.emb_layers[i](dep_tags[i])

            embeds.append(e)

        concat_embeds = torch.cat(embeds, 2)

        return self.lstm(concat_embeds)


class SingleTaskDep(nn.Module):
    """Modelo multitask que considera también los tags de dependencia"""

    def __init__(self, conf, vocab, num_labels=2, dropout=0.1):
        super().__init__()

        self.name = conf['model_name']
        self.embedding_dim = conf['embedding_dim']
        self.lstm_hidden_dim = conf['lstm_hidden_dim']
        self.encoder_dim = conf['encoder_dim']
        self.num_labels = num_labels

        self.tasks = ['to', 'as', 'about']

        # Capas modelo
        self.encoder = AutoModel.from_pretrained(
            self.name, num_labels=self.num_labels, output_attentions=True, output_hidden_states=True)
        self.taskLayer = nn.ModuleList([])
        self.taskLayer.append(nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder_dim+self.lstm_hidden_dim*2, self.num_labels)
        ))
        self.LSTM_model = LSTM_enc(
            self.embedding_dim, self.lstm_hidden_dim, vocab)

    def forward(self, input_ids=None, attention_mask=None, dep_tags=None, **kwargs):

        dBertoutputs = self.encoder(
            input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True)

        outputs_last_hidden_state = dBertoutputs[0]

        cls_out = outputs_last_hidden_state[:, 0]

        lstm_out, (lstm_hidden_state, cell_state) = self.LSTM_model(dep_tags)

        lstm_cat = torch.cat(
            [lstm_hidden_state[0], lstm_hidden_state[1]], dim=1)
        # Concateno ambos outputs
        output = torch.cat((cls_out, lstm_cat.squeeze()), 1)

        task_output = output

        for layer in self.taskLayer:
            task_output = layer(task_output)

        return task_output
