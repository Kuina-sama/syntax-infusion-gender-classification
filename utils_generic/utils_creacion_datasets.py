
from tqdm import tqdm
from CoDeLin.src.encs.enc_deps import *
from CoDeLin.src.models.conll_node import ConllNode
import stanza
import torch

import time
import sys
sys.path.append(r"C:\Users\kuina\OneDrive\TFG\Codigo\CoDeLin")

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse',
                      verbose=False, tokenize_no_ssplit=True)


def create_dependency_tags(dataset, encoding_type: list, separator: str):
    """supported encoders: absolute
        relative,
        pos"""

    encoders = {'absolute': naive_absolute.D_NaiveAbsoluteEncoding(separator),
                'relative': naive_relative.D_NaiveRelativeEncoding(separator),
                'pos': pos_based.D_PosBasedEncoding(separator),
                'brk': brk_based.D_BrkBasedEncoding(separator, True)
                }

    all_sentences = list(dataset.keys())

    print('Comenzando parsing de dependencias....\n')
    start = time.time()
    docs_in = [stanza.Document([], text=doc) for doc in all_sentences]

    docs_out = nlp(docs_in)
    end = time.time()
    print(
        f'Parsing de dependencias terminado. Duración del proceso: {(end-start)/60} minutos')
    print('Comenzando a generar el encoding para las dependencias')

    for doc in tqdm(docs_out):

        ############################################
        # Pruebas para añadir el tokenizado de stanza como elemento del dataset
        dataset[doc.text]['tokenized'] = ' '.join(
            [token.text for token in doc.sentences[0].tokens])

        # Fin de pruebas
        ############################################
        dicts = doc.to_dict()
        conllu_nodes = []
        conllu_nodes.append(ConllNode.dummy_root())
        for item in dicts[0]:
            id = item.get('id', '_')
            form = item.get('text', '_')
            lemma = item.get('lemma', '_')
            upos = item.get('upos', '_')
            xpos = item.get('xpos', '_')
            feats = item.get('feats', '_')
            head = item.get('head', '_')
            deprel = item.get('deprel', '_')

            conllu_nodes.append(ConllNode(wid=id, form=form, lemma=lemma, upos=upos, xpos=xpos,
                                          feats=feats, head=head, deprel=deprel, deps='_', misc='_'))

        # for type in encoding_type:
        for type_en in encoding_type:

            dataset[doc.text][type_en] = [
                str(label) for label in encoders[type_en].encode(conllu_nodes)]

    print('Proceso terminado con éxito')

    return


def process_md_dataset(data, tasks):
    """Formatea md_gender para que considere todas las tareas indicadas en tasks.
    Si un texto no tiene etiqueta para una tarea, tendrá asociada una lista vacía"""  # El valor
    # de relleno lo meto al crear el PytorchDataset

    new_data = {}
    map_md_tasks = {0: 'label_about', 1: 'label_about',
                    2: 'label_to', 3: 'label_to', 4: 'label_as', 5: 'label_as'}

    for item in data:
        text = item['text']

        if text in new_data:
            pass
        else:
            new_data[text] = {f'label_{x}': [] for x in tasks}

        for label in item['labels']:
            task = map_md_tasks[label]
            if task in new_data[text]:
                new_data[text][task].append(
                    data.features['labels'][0].int2str(label))

    return new_data
