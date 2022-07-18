from torch.autograd import Variable
from torchmetrics import F1Score
from torch import nn
import torch
import numpy as np
import pdb


def run_epoch(epoch, model, dataloader, cuda, training=False, optimizer=None):
    if training:
        model.train()
    else:
        model.eval()
    f1 = F1Score(num_classes=4, average="weighted", mdmc_average="samplewise")
    losssss = []
    acccc = []
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
#         pdb.set_trace()
        print("Targets that are fed to the model are: {}".format(targets.data))
        outputs = model(inputs)
        print("Output of the model is: {}".format(outputs))
        loss = nn.BCEWithLogitsLoss()(outputs, targets)
        print("Loss is: {}".format(loss))
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
        predicted = (outputs > 0.6).float()
        print("Predicted output is: {}".format(predicted))
        acc = f1(predicted.int().to("cpu"), targets.data.int().to("cpu"))
        total += targets.size(0)
        losssss.append(loss.item())
        acccc.append(acc)
#         if cuda:
#             print("predicted.eq(target.data) = {}".format(predicted.eq(targets.data)))
#             print("predicted.eq(target.data).cpu().sum() = {}".format(predicted.eq(targets.data).cpu().sum()))
#             print("predicted.eq(target.data).cpu().sum().item() = {}".format(predicted.eq(targets.data).cpu().sum().item()))
#             print("Counting correct: {}".format(predicted.eq(targets.data).cpu().sum().item()))
#             correct += predicted.eq(targets.data).cpu().sum().item()
#         else:
#             correct += predicted.eq(targets.data).sum().item()
#     acc = 100 * correct / total
#     acc = f1(predicted.int().to("cpu"), targets.data.int().to("cpu"))
    avg_loss = total_loss / total
    return acccc[-1], avg_loss, acccc, losssss


def get_predictions(model, dataloader, cuda, get_probs=False):
    preds = []
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets.long())
        outputs = model(inputs)
        if get_probs:
            probs = torch.nn.functional.softmax(outputs, dim=1)
            if cuda: probs = probs.data.cpu().numpy()
            else: probs = probs.data.numpy()
            preds.append(probs)
        else:
            _, predicted = torch.max(outputs.data, 1)
            if cuda: predicted = predicted.cpu()
            preds += list(predicted.numpy().ravel())
    if get_probs:
        return np.vstack(preds)
    else:
        return np.array(preds)
