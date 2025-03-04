import utils_generic as generic
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn

import numpy as np

from transformers import AutoModel

from tqdm.auto import tqdm
from transformers import get_scheduler

from datasets import load_metric


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


#######################################################################################

# CREATE DATALOADER

#######################################################################################


class DatasetSingleTaskSimple(Dataset):
    def __init__(self, data, task, eval):
        self.data = data
        self.task = task
        self.eval = eval

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        x = torch.tensor(self.data[index]['input_ids'])

        attention = torch.tensor(self.data[index]['attention_mask'])

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

        sample = {'input_ids': x,
                  'attention_mask': attention,
                  'tasks': self.task,
                  'label': label}

        return sample

    def __len__(self):
        return len(self.data)


class DatasetSingleTaskSimple_triple(Dataset):
    def __init__(self, data, task, eval):
        self.data = data
        self.task = task
        self.eval = eval

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        x = torch.tensor(self.data[index]['input_ids'])

        attention = torch.tensor(self.data[index]['attention_mask'])

        raw_label = self.data[index]['labels'][self.task]

        if len(raw_label) > 1:
            label = np.random.choice(raw_label)

        elif len(raw_label) == 1:
            label = raw_label[0]
        elif len(raw_label) == 0 and self.eval == True:
            label = 2

        sample = {'input_ids': x,
                  'attention_mask': attention,
                  'tasks': self.task,
                  'label': label}

        return sample

    def __len__(self):
        return len(self.data)


def collate_fn(batch):

    input_ids = [b['input_ids'] for b in batch]
    input_ids = pad_sequence(input_ids, batch_first=True)

    label = torch.tensor([b['label'] for b in batch])

    attention = [b['attention_mask'] for b in batch]
    attention_mask = pad_sequence(attention, batch_first=True)

    batched_input = {'input_ids': input_ids,
                     'attention_mask': attention_mask, 'label': label}

    return batched_input


#######################################################################################

#  MODELO SINGLETASK SIN DEPENDENCIAS

#######################################################################################


class SingleTaskSimple(nn.Module):
    """Single Task Model without dependency tags on input"""

    def __init__(self, conf, num_labels=2, dropout=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.name = conf['model_name']
        self.encoder_dim = conf['encoder_dim']
        self.dropout = dropout
        self.num_labels = num_labels
        # Capas modelo
        self.encoder = AutoModel.from_pretrained(
            self.name, num_labels=self.num_labels, output_attentions=True, output_hidden_states=True)
        self.taskLayer = nn.ModuleList([])
        self.taskLayer.append(nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder_dim, self.num_labels)
        ))

    def forward(self, input_ids=None, attention_mask=None, **kwargs):

        dBertoutputs = self.encoder(
            input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True)

        outputs_last_hidden_state = dBertoutputs[0]

        cls_out = outputs_last_hidden_state[:, 0]

        task_output = cls_out

        for layer in self.taskLayer:
            task_output = layer(task_output)

        return task_output
