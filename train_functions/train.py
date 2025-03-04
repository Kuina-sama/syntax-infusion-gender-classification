import torch
from torch import nn

from transformers import get_scheduler
from datasets import load_metric

from tqdm.auto import tqdm

import utils_generic as generic
import numpy as np

from sklearn.metrics import accuracy_score, recall_score,  precision_score, f1_score

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


##################################################
##################################################

# TRAIN SINGLE TASK

##################################################
##################################################


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
    for batch in dl_train:
        batch = {k: v.to(device)
                 for k, v in batch.items()}  # Manda los datos a la gpu

        optimizer.zero_grad()
        output = model(**batch)

        loss = loss_fct(output, batch['label'])

        epoch_loss += loss.item()

        loss.backward()  # Lo usa para calcular los gradientes

        optimizer.step()  # training step en el optimizador
        lr_scheduler.step()  # update del learning rate

        progress_bar.update(1)

    return epoch_loss/len(dl_train)


def train_function(model, num_epochs, dl_train, optimizer, early_stop=10, dl_val=None, save_path='model', es_threshold=0):
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

    model.to(device)

    model.train()

    train_loss = []
    val_loss = []
    epochs_with_no_improvement = 0
    best_loss = 1

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()

        epoch_loss = train_one_epoch(
            model, dl_train, optimizer, lr_scheduler, progress_bar)

        train_loss.append(epoch_loss)

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
    return



def eval_func(model, dl_eval, metrics=['recall', 'precision', 'f1', 'accuracy']):
    """Metrics supported: recall, precision,f1-score,accuracy"""
    metrics_score = {'recall': recall_score, 'precision': precision_score,
                     'f1': f1_score, 'accuracy': accuracy_score}
    model.eval()

    total_predictions = torch.tensor([]).to(device)
    total_labels = torch.tensor([]).to(device)

    for batch in dl_eval:
        batch = {k: v.to(device) for k, v in batch.items()}

        mask = batch['label'] != 2

        if len(batch['label'][mask]) == 0:  # Caso de que no tengamos esa tarea en la lista
            continue

        with torch.no_grad():

            outputs = model(**batch)

        predictions = torch.softmax(outputs, dim=1)[mask]
        y_pred = torch.argmax(predictions, dim=1)
        labels = batch['label'][mask]

        total_predictions = torch.cat([total_predictions, y_pred])
        total_labels = torch.cat([total_labels, labels])

    metrics_result = {}
    _, weights = np.unique(total_labels.cpu(), return_counts=True)
    for metric in metrics:
        if metric == 'accuracy':
            metrics_result[metric] = metrics_score[metric](
                total_labels.cpu(), total_predictions.cpu())
        else:
            m = metrics_score[metric](
                total_labels.cpu(), total_predictions.cpu(), average=None, labels=[0, 1])
            avg = m.mean()
            weighted_avg = np.average(m, weights=weights)
            metrics_result[metric] = dict(
                zip(['female', 'male', 'average', 'weighted_avg'], m.tolist()+[avg, weighted_avg]))

    return metrics_result


def eval_func3(model, dl_eval, metrics=['recall', 'precision', 'f1', 'accuracy']):
    """Metrics supported: recall, precision,f1-score,accuracy"""
    metrics_score = {'recall': recall_score, 'precision': precision_score,
                     'f1': f1_score, 'accuracy': accuracy_score}
    model.eval()

    total_predictions = torch.tensor([]).to(device)
    total_labels = torch.tensor([]).to(device)

    for batch in dl_eval:
        batch = {k: v.to(device) for k, v in batch.items()}

        mask = batch['label'] != 2

        if len(batch['label'][mask]) == 0:  # Caso de que no tengamos esa tarea en la lista
            continue

        with torch.no_grad():

            outputs = model(**batch)

        predictions = torch.softmax(outputs, dim=1)[mask]
        y_pred = torch.argmax(predictions, dim=1)
        labels = batch['label'][mask]

        total_predictions = torch.cat([total_predictions, y_pred])
        total_labels = torch.cat([total_labels, labels])

    metrics_result = {}
    for metric in metrics:
        if metric == 'accuracy':
            metrics_result[metric] = metrics_score[metric](
                total_labels.cpu(), total_predictions.cpu())
        else:
            m = metrics_score[metric](
                total_labels.cpu(), total_predictions.cpu(), average=None, labels=[0, 1, 2])
            # avg = m.mean()  # ojo que aqui calcula la métrica de unknown también
            metrics_result[metric] = dict(
                zip(['female', 'male', 'unknown'], m.tolist()))

    return metrics_result
##################################################
##################################################

# TRAIN MULTITASK

##################################################
##################################################


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
    for batch in dl_train:
        batch = {k: v.to(device)
                 for k, v in batch.items()}  # Manda los datos a la gpu

        optimizer.zero_grad()

        output = model(**batch)

        loss = 0
        for task in output:

            targets = batch[task].to(device)
            predicted = output[task].to(device)

            loss += loss_fct(predicted, targets)

        epoch_loss += loss.item()

        loss.backward()  # Lo usa para calcular los gradientes

        optimizer.step()  # training step en el optimizador
        lr_scheduler.step()  # update del learning rate

        progress_bar.update(1)

    return epoch_loss


def train_function_multi(model, num_epochs, dl_train, optimizer, early_stop=10, dl_val=None, save_path='model', es_threshold=0):

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

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()

        epoch_loss = train_one_epoch_multi(
            model, dl_train, optimizer, lr_scheduler, progress_bar)

        epoch_train_loss = epoch_loss/len(dl_train)
        train_loss.append(epoch_train_loss)

        if dl_val:
            epoch_val_loss = validation_func_multi(model, dl_val)
            val_loss.append(epoch_val_loss)
            print(
                f'Epoch {epoch+1} \t Training loss: {epoch_train_loss} \t Validation loss: {epoch_val_loss} \t ')
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
    return




def eval_func_multi(model, dl_eval, tasks, metrics=['recall', 'precision', 'f1', 'accuracy']):
    """Metrics supported: recall, precision,f1-score,accuracy"""
    metrics_score = {'recall': recall_score, 'precision': precision_score,
                     'f1': f1_score, 'accuracy': accuracy_score}
    model.eval()

    total_predictions = {}
    total_labels = {}
    for task in tasks:
        total_predictions[task] = torch.tensor([]).to(device)
        total_labels[task] = torch.tensor([]).to(device)

    for batch in dl_eval:
        batch = {k: v.to(device) for k, v in batch.items()}

        for task in tasks:

            mask = batch[task] != 2

            # Caso de que no tengamos muestras con etiquetas conocidas
            if len(batch[task][mask]) == 0:
                continue

            with torch.no_grad():
                outputs = model(**batch)

                predictions = torch.softmax(outputs[task], dim=1)[mask]
                y_pred = torch.argmax(predictions, dim=1)
                labels = batch[task][mask]

                total_predictions[task] = torch.cat(
                    [total_predictions[task], y_pred])
                total_labels[task] = torch.cat([total_labels[task], labels])

    metrics_result = {}
    for task in total_predictions:
        _, weights = np.unique(total_labels[task].cpu(), return_counts=True)
        metrics_result[task] = {}
        for metric in metrics:
            if metric == 'accuracy':
                metrics_result[task][metric] = metrics_score[metric](
                    total_labels[task].cpu(), total_predictions[task].cpu())
            else:
                m = metrics_score[metric](total_labels[task].cpu(
                ), total_predictions[task].cpu(), average=None, labels=[0, 1])

                avg = m.mean()
                weighted_avg = np.average(m, weights=weights)
                metrics_result[task][metric] = dict(
                    zip(['female', 'male', 'average', 'weighted_avg'], m.tolist()+[avg, weighted_avg]))

    return metrics_result
