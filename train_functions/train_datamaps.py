import seaborn as sns
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from transformers import get_scheduler
from datasets import load_metric

from tqdm.auto import tqdm

import utils_generic as generic


from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score, f1_score


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

########################################################################
########################################################################
#
# Funciones para entrenamiento con datamaps
#
########################################################################
########################################################################

# Estas funciones realizan el entrenamiento (igual que las originales) y al mismo tiempo van almacenando los
# valores de las true probs a lo largo de los epochs y el un tensor que nos dice, para cada epoch, si el modelo
# acierta al predecir la muestra de entrenamiento (con los dos tensores que sacamos podremos calcualr a
# posteriori los valores de confidence, variability y correctness)


###
# Estas dos funciones me sirven para sacar datos para el datamaps durante entrenamiento
###

def true_label_prob(output, labels):
    out_soft = torch.softmax(output, dim=1)

    return torch.tensor([out[l].item() for out, l in zip(out_soft, labels)]).unsqueeze(1)


def is_correct(output, labels):
    predictions = torch.softmax(output, dim=1)
    y_pred = torch.argmax(predictions, dim=1)

    return (labels == y_pred).cpu().unsqueeze(1)


########################################################################
########################################################################
#
# Datamaps en train singletask
#
########################################################################
########################################################################
################################################
# Funciones de entrenamiento adaptadas (singletask)
################################################


def validation_func(model, dl_val):
    model.eval()

    loss_fct = nn.CrossEntropyLoss()
    val_loss = 0
    for batch in dl_val:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():

            val_output = model(**batch)
            loss = loss_fct(val_output, batch['label'])

        val_loss += loss.item()

    return val_loss/len(dl_val)


def train_one_epoch(model, dl_train, optimizer, lr_scheduler, progress_bar):

    loss_fct = nn.CrossEntropyLoss()
    epoch_loss = 0
    epoch_true_label_probs = torch.tensor([])
    epoch_correctness = torch.tensor([])
    for batch in dl_train:
        batch = {k: v.to(device)
                 for k, v in batch.items()}  # Manda los datos a la gpu

        optimizer.zero_grad()
        output = model(**batch)

        epoch_true_label_probs = torch.cat(
            [epoch_true_label_probs, true_label_prob(output, batch['label'])], dim=0)  # aquí es uno detrás de otro

        epoch_correctness = torch.cat(
            [epoch_correctness, is_correct(output, batch['label'])], dim=0)

        loss = loss_fct(output, batch['label'])

        epoch_loss += loss.item()

        loss.backward()  # Lo usa para calcular los gradientes

        optimizer.step()  # training step en el optimizador
        lr_scheduler.step()  # update del learning rate

        progress_bar.update(1)

    return epoch_loss/len(dl_train), epoch_true_label_probs, epoch_correctness


def train_function(model, num_epochs, dl_train, optimizer, early_stop=10, dl_val=None, save_path='model', es_threshold=0, emb_weights=False):
    '''
    model: modelo a entrenar.
    num_epoch: ciclos de entrenamiento
    dl_train: conjunto de entrenamiento, debe ser un Pytorch DataLoader
    optimizer: optimizador usado durante el entrenamiento.

    Esta función usará un learning rate scheduler para ir cambiando el 
    learning rate.

    Usará gpu siempre que esté disponible.

    Esta función irá usando paralelamente un conjunto de validación para calcular el error'''

    num_training_steps = num_epochs * len(dl_train)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    # model.to(device)

    model.train()

    train_loss = []
    val_loss = []
    epochs_with_no_improvement = 0
    best_loss = 1

    prob_true_labels = torch.tensor([])
    correctness_by_epoch = torch.tensor([])
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()

        epoch_loss, epoch_true_label_probs, epoch_correctness = train_one_epoch(
            model, dl_train, optimizer, lr_scheduler, progress_bar)

        train_loss.append(epoch_loss)

        prob_true_labels = torch.cat(
            [prob_true_labels, epoch_true_label_probs], dim=1)

        correctness_by_epoch = torch.cat(
            [correctness_by_epoch, epoch_correctness], dim=1)

        if dl_val:
            epoch_val_loss = validation_func(model, dl_val)
            val_loss.append(epoch_val_loss)
            print(
                f'Epoch {epoch+1} \t Training loss: {epoch_loss} \t Validation loss: {epoch_val_loss} \t ')
            if epoch_val_loss < best_loss:  # El error ha mejorado
                best_loss = epoch_val_loss
                epochs_with_no_improvement = 0
                torch.save(model.state_dict(), save_path)

            elif epoch_val_loss - best_loss >= es_threshold:  # El error ha empeorado
                epochs_with_no_improvement += 1
                print(f"\n{epochs_with_no_improvement} epoch without improvement")
                if epochs_with_no_improvement >= early_stop:
                    print(
                        f"Validation_loss hasn't improve in {early_stop} epoch. Stopping training after {epoch+1} epochs...")

                    break

        else:
            print(f'Epoch {epoch+1} \t Training loss: {epoch_loss} ')

        print(progress_bar)

    generic.plot_losses_val(train_loss, val_loss)
    return prob_true_labels, correctness_by_epoch, epoch+1


########################################################################
########################################################################
#
# Datamaps en train multitask
#
########################################################################
########################################################################
################################################
# Funciones de entrenamiento adaptadas (multitask)
################################################


def validation_func_multi(model, dl_val):
    model.eval()

    loss_fct = nn.CrossEntropyLoss()
    val_loss = 0
    for batch in dl_val:

        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():

            val_output = model(**batch)

        loss = 0
        for task in val_output:
            targets = batch[task].to(device)
            predicted = val_output[task].to(device)

            loss += loss_fct(predicted, targets)

        val_loss += loss.item()

    return val_loss/len(dl_val)


def train_one_epoch_multi(model, dl_train, optimizer, lr_scheduler, progress_bar):

    loss_fct = nn.CrossEntropyLoss()
    epoch_loss = 0

    epoch_true_label_probs = {'about': torch.tensor([]),
                              'to': torch.tensor([]),
                              'as': torch.tensor([])}

    epoch_correctness = {'about': torch.tensor([]),
                         'to': torch.tensor([]),
                         'as': torch.tensor([])}
    for batch in dl_train:
        batch = {k: v.to(device)
                 for k, v in batch.items()}  # Manda los datos a la gpu

        optimizer.zero_grad()

        output = model(**batch)

        loss = 0
        for task in output:
            epoch_true_label_probs[task] = torch.cat(
                [epoch_true_label_probs[task], true_label_prob(output[task], batch[task])], dim=0)

            epoch_correctness[task] = torch.cat(
                [epoch_correctness[task], is_correct(output[task], batch[task])], dim=0)
            targets = batch[task].to(device)
            predicted = output[task].to(device)

            loss += loss_fct(predicted, targets)

        epoch_loss += loss.item()

        loss.backward()  # Lo usa para calcular los gradientes

        optimizer.step()  # training step en el optimizador
        lr_scheduler.step()  # update del learning rate

        progress_bar.update(1)

    return epoch_loss/len(dl_train), epoch_true_label_probs, epoch_correctness


def train_function_multi(model, num_epochs, dl_train, optimizer, early_stop=10, dl_val=None, save_path='model', es_threshold=0, emb_weights=False):

    num_training_steps = num_epochs * len(dl_train)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    model.to(device)

    model.train()

    train_loss = []
    val_loss = []
    epochs_with_no_improvement = 0
    best_loss = 10

    probs_true_labels = {'about': torch.tensor([]),
                         'to': torch.tensor([]),
                         'as': torch.tensor([])}

    correctness_by_epoch = {'about': torch.tensor([]),
                            'to': torch.tensor([]),
                            'as': torch.tensor([])}
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()

        epoch_loss, epoch_true_label_probs, epoch_correctness = train_one_epoch_multi(
            model, dl_train, optimizer, lr_scheduler, progress_bar)

        train_loss.append(epoch_loss)
        for task in ['about', 'to', 'as']:
            probs_true_labels[task] = torch.cat(
                [probs_true_labels[task], epoch_true_label_probs[task]], dim=1)

            correctness_by_epoch[task] = torch.cat(
                [correctness_by_epoch[task], epoch_correctness[task]], dim=1)

        if dl_val:
            epoch_val_loss = validation_func_multi(model, dl_val)
            val_loss.append(epoch_val_loss)
            print(
                f'Epoch {epoch+1} \t Training loss: {epoch_loss} \t Validation loss: {epoch_val_loss} \t ')
            if epoch_val_loss < best_loss:  # El error ha mejorado
                best_loss = epoch_val_loss
                epochs_with_no_improvement = 0
                torch.save(model.state_dict(), save_path)

            elif epoch_val_loss - best_loss >= es_threshold:  # El error no ha mejorado lo suficiente
                epochs_with_no_improvement += 1
                print(f"\n{epochs_with_no_improvement} epoch without improvement")
                if epochs_with_no_improvement >= early_stop:
                    print(
                        f"Validation_loss hasn't improve in {early_stop} epoch. Stopping training after {epoch+1} epochs...")

                    break

        else:
            print(f'Epoch {epoch+1} \t Training loss: {epoch_train_loss} ')

        print(progress_bar)

    generic.plot_losses_val(train_loss, val_loss)
    return probs_true_labels, correctness_by_epoch, epoch+1

########################################################################
########################################################################
#
# Funciones para cálculo de variables de datamaps y creación de gráficas
#
########################################################################
########################################################################


def get_confidence(true_labels_probs, num_epochs=10):
    confidence = true_labels_probs[:, 0:num_epochs].mean(axis=1)

    return confidence


def get_variability(true_labels_probs, num_epochs=10):

    aux = [(true_labels_probs[:, i-1] - get_confidence(true_labels_probs,
            num_epochs=i))**2 for i in range(1, num_epochs+1)]
    aux_stacked = torch.stack(aux, axis=1)

    variability = (aux_stacked/num_epochs)**(1/2)

    return variability[:, -1:].squeeze()


def get_correctness_vector_from_true_labels(true_labels_probs):
    """Función que saca el vector de aciertos por epoch a partir de las true labels.
    Solo válida si tenemos dos clases (asigna como acierto si la probabilidad de la true label
    es mayor que 0.5)
    Usar solo en casos donde en train se hayan sacado las true_labels_probs pero no el correctness"""
    correctness = true_labels_probs.clone()
    correctness[true_labels_probs > 0.5] = 1
    correctness[true_labels_probs <= 0.5] = 0

    # Devuelvo el vector para no tener que modificar la función de get_datamaps_info
    return correctness


random_index_15k = torch.load('map_index.pt')

random_index_10k = torch.load('map_index_10k.pt')


def get_datamaps_info(true_labels_probs, correctness_vector, num_epochs=10, show_samples=False):
    if show_samples:
        variability = get_variability(
            true_labels_probs[random_index_15k], num_epochs)
        confidence = get_confidence(
            true_labels_probs[random_index_15k], num_epochs)
        correctness = correctness_vector[random_index_15k].mean(axis=1)
    else:
        variability = get_variability(true_labels_probs, num_epochs)
        confidence = get_confidence(true_labels_probs, num_epochs)
        correctness = correctness_vector.mean(axis=1)

    boundaries = torch.tensor([0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0])
    correctness_discrete = torch.bucketize(correctness, boundaries)
    correctness_for_plot = boundaries[correctness_discrete]

    data = pd.DataFrame(dict(variability=variability,
                        confidence=confidence, correctness_plot=correctness_for_plot, correctness=correctness))
    return data


def get_datamap(true_labels_probs, correctness, num_epochs=10, plot_title='Data Map', show_samples=False):

    data = get_datamaps_info(
        true_labels_probs, correctness, num_epochs, show_samples)
    #

    #
    fig = sns.scatterplot(data=data, x='variability', y='confidence',
                          style='correctness_plot', hue='correctness_plot', palette='deep')
    fig.set(title=plot_title)
    return


def plot_datamaps_density(data, variable, color='blue', bins=7):
    """data must be a pandas dataframe"""
    fig = sns.histplot(data, x=variable, bins=bins, color=color)
    fig.set(xlabel=variable, ylabel='density',
            title=f'{variable} Distirbution')

    return


def get_datamap_complete_graph(true_labels_probs, correctness_vector, num_epochs=10, show_samples=False):
    map_data = get_datamaps_info(
        true_labels_probs, correctness_vector=correctness_vector, num_epochs=num_epochs, show_samples=show_samples)

    f = plt.figure()

    f.set_figheight(15)
    f.set_figwidth(20)
    f.add_subplot(3, 2, (1, 5))
    get_datamap(true_labels_probs, correctness=correctness_vector,
                num_epochs=num_epochs, show_samples=show_samples)
    f.add_subplot(3, 2, 2)
    plot_datamaps_density(map_data, 'confidence', color='lightgreen')
    f.add_subplot(3, 2, 4)
    plot_datamaps_density(map_data, 'variability', color='purple')
    f.add_subplot(3, 2, 6)
    plot_datamaps_density(map_data, 'correctness', color='lightblue')

    plt.show()

    return
