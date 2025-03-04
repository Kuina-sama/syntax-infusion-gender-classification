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


class DatasetMultiTaskSimple(Dataset):
    def __init__(self, data, tasks, eval):
        self.data = data
        self.tasks = tasks
        self.eval = eval

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        x = torch.tensor(self.data[index]['input_ids'])

        attention = torch.tensor(self.data[index]['attention_mask'])

        raw_labels = self.data[index]['labels']

        labels = {'to': [], 'as': [], 'about': []}
        for task in self.tasks:
            aux = raw_labels[task]
            if len(aux) > 1:
                label = np.random.choice(aux)
                if label == 2:
                    label = np.random.choice([0, 1])
                labels[task].append(label)
            elif len(aux) == 1:
                if aux[0] == 2:
                    label = np.random.choice([0, 1])
                    labels[task].append(label)
                else:
                    labels[task].append(aux[0])
            elif len(aux) == 0 and self.eval == True:
                labels[task].append(2)

        sample = {'input_ids': x,
                  'attention_mask': attention,
                  'tasks': self.tasks}

        sample.update(labels)
        return sample

    def __len__(self):
        return len(self.data)


def collate_fn(batch):

    input_ids = [b['input_ids'] for b in batch]
    input_ids = pad_sequence(input_ids, batch_first=True)

    labels = {}
    for task in batch[0]['tasks']:
        labels[task] = torch.tensor([b[task][0] for b in batch])

    attention = [b['attention_mask'] for b in batch]
    attention_mask = pad_sequence(attention, batch_first=True)

    batched_input = {'input_ids': input_ids, 'attention_mask': attention_mask}
    batched_input.update(labels)

    return batched_input


#######################################################################################

#  MODELO MULTITASK

#######################################################################################


class MultiTaskSimple(nn.Module):

    """Multitask model without dependency tags on input"""

    def __init__(self, conf, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.name = conf['model_name']
        self.encoder_dim = conf['encoder_dim']
        self.dropout = conf.get('dropout', 0.1)
        self.tasks = ['to', 'as', 'about']

        # Capas modelo
        self.encoder = AutoModel.from_pretrained(
            self.name, num_labels=self.num_labels, output_attentions=True, output_hidden_states=True)
        self.taskLayer = nn.ModuleList([])
        self.taskLayer.append(nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder_dim, self.num_labels)

        ))

    def forward(self, input_ids=None, attention_mask=None, dep_tags=None, **kwargs):

        dBertoutputs = self.encoder(
            input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True)

        outputs_last_hidden_state = dBertoutputs[0]

        cls_out = outputs_last_hidden_state[:, 0]

        tasks_output = {v: cls_out for v in self.tasks}

        for layer in self.taskLayer:
            tasks_output = {v: layer(k) for v, k in tasks_output.items()}

        return tasks_output
