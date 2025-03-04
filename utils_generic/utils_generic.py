import matplotlib.pyplot as plt


from transformers import AutoTokenizer

import itertools
#######################################################################################

# CONSTANTS

#######################################################################################
text_to_num = {'to': {'PARTNER:female': 0, 'PARTNER:male': 1, "PARTNER:unknown": 2},
               'as': {'SELF:female': 0, 'SELF:male': 1, 'SELF:unknown': 2},
               'about': {'ABOUT:female': 0, 'ABOUT:male': 1, 'ABOUT:unknown': 2}}

num_to_text = {'to': {0: 'PARTNER:female', 1: 'PARTNER:male', 2: "PARTNER:unknown"},
               'as': {0: 'SELF:female', 1: 'SELF:male', 2: 'SELF:unknown'},
               'about': {0: 'ABOUT:female', 1: 'ABOUT:male', 2: 'ABOUT:unknown'}}

all_tasks_names = ['about', 'as', 'to']


task_to_num = {'about': 0, 'as': 1, 'to': 2}

num_to_task = {0: 'about', 1: 'as', 2: 'to'}

gender = {'female': 0, 'male': 1}

vocab_size_convai = 15066
#######################################################################################

# VOCABULARY

#######################################################################################


class Vocabulary():
    def __init__(self, data, encoding):
        self.encoding = encoding
        self._create_vocabulary(data, encoding)

        self.word_to_indx = []

        for i in self.vocabs:
            self.word_to_indx.append(
                {word: i for i, word in enumerate(self.vocabs[i])})

    def get_vocab_sizes(self):
        vocabs_len = []
        for i in self.vocabs:
            vocabs_len.append(len(self.vocabs[i]))

        return vocabs_len

    def total_vocabs(self):
        return self.total_vocabs

    def _create_vocabulary(self, dataset, encoding):

        if encoding == 'pos':
            self.total_vocabs = 3
            split_separator1 = '--'
            split_separator2 = '_'
        else:  # Cojo dos vocabularios salvo en el caso del pos based
            self.total_vocabs = 2
            split_separator = '_'

        self.vocabs = {}
        for i in range(self.total_vocabs):
            self.vocabs[i] = set()

        if encoding == 'pos':
            for item in list(dataset.values()):
                for dep_label in item[encoding]:
                    split1, split2 = dep_label.split(split_separator1)

                    split2, split3 = split2.split(split_separator2)
                    dep_label_split = (split1, split2, split3)
                    for indx in range(self.total_vocabs):

                        self.vocabs[indx].add(dep_label_split[indx])
                        self.vocabs[indx].add('unk')

        else:
            for item in list(dataset.values()):
                for dep_label in item[encoding]:
                    dep_label_split = dep_label.split(split_separator)

                    for indx in range(self.total_vocabs):

                        self.vocabs[indx].add(dep_label_split[indx])
                        self.vocabs[indx].add('unk')


class Vocabulary_basic():
    def __init__(self, data, encoding):
        self.encoding = encoding
        self._create_vocabulary(data, encoding)

        self.word_to_indx = []

        for i in self.vocabs:
            self.word_to_indx.append(
                {word: i for i, word in enumerate(self.vocabs[i])})

    def get_vocab_sizes(self):
        vocabs_len = []
        for i in self.vocabs:
            vocabs_len.append(len(self.vocabs[i]))

        return vocabs_len

    def total_vocabs(self):
        return self.total_vocabs

    def _create_vocabulary(self, dataset, encoding):

        self.total_vocabs = 1
        self.vocabs = {}
        for i in range(self.total_vocabs):
            self.vocabs[i] = set()

        for item in list(dataset.values()):
            for dep_label in item[encoding]:
                for indx in range(self.total_vocabs):

                    self.vocabs[indx].add(dep_label)
                    self.vocabs[indx].add('unk')



def split_sentence_dep_tags(dep_tags,pos_based_encoding=False):

    if pos_based_encoding:
        split_temp = [dep.replace('--', '_').split('_') for dep in dep_tags]
    else:
        split_temp = [dep.split('_') for dep in dep_tags]
        
    deps = [item for dep in split_temp for item in dep]

    return deps
    
#######################################################################################

# Funciones para formatear el dataset y tokenizar los textos

#######################################################################################


def tokenize_dataset(dataset, tasks_names, model_conf):
    """Tokeniza y formatea el dataset indicado para las tareas indicadas en tasks_names.
    Usa el tokenizer propio del modelo indicado.

    NO considera la información de parsing de dependencias"""

    tokenizer = model_conf['tokenizer']
    token_data = {}
    for index, text in enumerate(dataset):
        tokenized = tokenizer(text, truncation=True)

        labels = {}
        for task in tasks_names:
            aux_label = [text_to_num[task][x]
                         for x in dataset[text][f'label_{task}']]

            labels[task] = aux_label

        token_data[index] = {'text': text,
                             'input_ids': tokenized.input_ids,
                             'attention_mask': tokenized.attention_mask,
                             'labels': labels}

    return token_data


def tokenize_dataset_with_dependencies(dataset, tasks_names, vocab, model_conf):
    """ Tokeniza y formatea mi dataset. SI considera la información de parsing de dependencias.
    Por el momento funciona para encoding relative"""

    tokenizer = model_conf['tokenizer']
    token_data = {}
    for index, text in enumerate(dataset):
        tokenized = tokenizer(dataset[text]['tokenized'], truncation=True)

        labels = {}
        for task in tasks_names:
            aux_label = [text_to_num[task][x]
                         for x in dataset[text][f'label_{task}']]

            labels[task] = aux_label

        # Esta parte procesa los tags de dependencia
        aux = [x.split('_') for x in dataset[text][vocab.encoding]]

        if vocab.encoding == 'pos':
            aux = [x.replace('--', '_').split('_')
                   for x in dataset[text][vocab.encoding]]

        dep_tags = {}
        for x in range(vocab.total_vocabs):
            dep_tags[f'tag{x}'] = [vocab.word_to_indx[x].get(
                aux[i][x], vocab.word_to_indx[x]['unk']) for i in range(len(aux))]

        # Junto todo en un nuevo dataset
        token_data[index] = {'text': text,
                             'input_ids': tokenized.input_ids,
                             'attention_mask': tokenized.attention_mask,
                             'labels': labels,
                             'dep_tags': dep_tags}

    return token_data


def create_word_to_index(data):
    train_words = set(list(itertools.chain.from_iterable(
        [sentence.split(' ') for sentence in data.keys()])))
    train_words.add('unk')

    word_to_index = {word: i for i, word in enumerate(train_words)}

    return word_to_index


def create_w2v_vocab(train_data, w2v_model):
    v = set(create_word_to_index(train_data).keys()
            ) & set(w2v_model.key_to_index.keys())
    vocabulary = list(v)
    vocabulary.insert(0, 'pad')

    return vocabulary


def tokenize_dataset_rrnn(dataset, tasks_names, word_to_index=None, vect=None):
    """Tokeniza y formatea el dataset indicado para las tareas indicadas en tasks_names.

    NO considera la información de parsing de dependencias
    """

    token_data = {}
    for index, text in enumerate(dataset):
        if vect is not None:
            tokenized = vect.transform([dataset[text]['tokenized']]).toarray()
        else:
            tokenized = [word_to_index.get(
                word, word_to_index['unk']) for word in text.split(' ')]

        labels = {}
        for task in tasks_names:
            aux_label = [text_to_num[task][x]
                         for x in dataset[text][f'label_{task}']]

            labels[task] = aux_label

        token_data[index] = {'text': text,
                             'input_ids': tokenized,
                             'labels': labels}

    return token_data


def tokenize_dataset_with_dependencies_rrnn(dataset, tasks_names, vocab=None, word_to_index=None, vect=None):
    """ Tokeniza y formatea mi dataset. SI considera la información de parsing de dependencias.
    Por el momento funciona para encoding relative"""

    token_data = {}
    for index, text in enumerate(dataset):
        if vect is not None:

            tokenized = vect.transform(
                [dataset[text]['sentence_with_deps']]).toarray()
        else:
            tokenized = [word_to_index.get(
                word, word_to_index['unk']) for word in dataset[text]['tokenized'].split(' ')]

        labels = {}
        for task in tasks_names:
            aux_label = [text_to_num[task][x]
                         for x in dataset[text][f'label_{task}']]

            labels[task] = aux_label

        if vocab:
            # Esta parte procesa los tags de dependencia
            aux = [x.split('_') for x in dataset[text][vocab.encoding]]

            if vocab.encoding == 'pos':
                aux = [x.replace('--', '_').split('_')
                    for x in dataset[text][vocab.encoding]]

            dep_tags = {}
            for x in range(vocab.total_vocabs):
                dep_tags[f'tag{x}'] = [vocab.word_to_indx[x].get(
                    aux[i][x], vocab.word_to_indx[x]['unk']) for i in range(len(aux))]

            # Junto todo en un nuevo dataset
            token_data[index] = {'text': text,
                                'input_ids': tokenized,
                                'labels': labels,
                                'dep_tags': dep_tags}
        else:
            token_data[index] = {'text': text,
                    'input_ids': tokenized,
                    'labels': labels}


    return token_data
#######################################################################################

# PLOT FUNCTIONS

#######################################################################################


def plot_losses_val(train_loss, val_loss):
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['train loss', 'validation loss'])
    plt.title('Train-Validation loss')
    plt.show()

    return


def plot_losses_train(train_loss):
    plt.plot(train_loss)
    plt.legend(['train loss'])
    plt.title('Train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    return


def update_global_metric(metric_result, global_result):
    for metric, value in metric_result.items():
        global_result[metric].append(value)


def update_global_metric_multi(metric_result, global_result, mtype='global'):
    for task in global_result:
        for metric, value in metric_result[task].items():
            global_result[task][mtype][metric].append(value)
