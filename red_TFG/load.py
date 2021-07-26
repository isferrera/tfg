
import os

import argparse

import numpy as np

import torch
from torchvision import transforms
from tqdm.auto import tqdm

from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.network.network import load

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import SubsetRandomSampler

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=6000)
parser.add_argument("--n_train", type=int, default=24000)
parser.add_argument("--n_workers", type=int, default=0)
parser.add_argument("--update_steps", type=int, default=256)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--colab", dest="colab", action="store_true")
parser.add_argument("--local", dest="colab", action="store_false")
parser.add_argument("--alum", dest="alum", action="store_true")
parser.add_argument("--n_folds", type=int, default=6)
parser.set_defaults(gpu=False, colab=False)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
update_steps = args.update_steps
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
gpu = args.gpu
colab = args.colab
n_folds = args.n_folds
alum = args.alum

update_interval = update_steps * batch_size

if gpu and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    if gpu:
        print("Could not use CUDA")
        gpu = False
torch.manual_seed(seed)
print("Running on Device = %s\n" % (device))

if colab:
    dataset_path = "/content/drive/MyDrive/Isabel/MNIST"
    dirName = "/content/drive/MyDrive/Isabel/networks/"
    dirName_o = "/content/drive/MyDrive/Isabel/networks/"
    dirName_n = "/content/drive/MyDrive/Isabel/networks/400N_1BS_5E_1US_3TF_1CF"
elif alum:
    dataset_path = "/content/drive/MyDrive/data/MNIST"
    dirName = "/content/drive/MyDrive/networks/"
    dirName_o = "/content/drive/MyDrive/networks/"
    dirName_n = "/content/drive/MyDrive/networks/400N_1BS_5E_1US_3TF_1CF"
else:
    dataset_path = "/home/iferrera/bindsnet/bindsnet/data/MNIST"
    dirName = "/home/iferrera/networks/"
    dirName_o = "/home/iferrera/networks/"
    dirName_n = "/home/iferrera/networks/400N_1BS_5E_1US_3TF_1CF"
# Create the directory to store the networks

if not os.path.exists(dirName):
    os.mkdir(dirName)

# Load MNIST train data
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=dataset_path,
    download=False,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)


network = DiehlAndCook2015(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    nu=(1e-4, 1e-2),
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

# Directs network to GPU
if gpu:
    network.to("cuda")

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Set up monitors for spikes
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

spike_record = torch.zeros(
    (update_interval, int(time / dt), n_neurons), device=device)
###################### LOAD THE NETWORK ########################


network = load(dirName_n + "/network.pt", map_location="cpu", learning=False)

assignments = torch.load(dirName_n + "/assignments.pt")

proportions = torch.load(dirName_n + "/proportions.pt")

# Set up monitors for spikes
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)



################# EVALUATION VALIDATION SET #########################
# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root= dataset_path,
    download=False,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Create a dataloader to iterate and batch data
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=n_workers,
    pin_memory=gpu,
)

# Validation set accuracies
accuracy = {"all": 0, "proportion": 0}

print("Begin evaluation test set.\n")
network.train(mode=False)

folds_test_samples = 6000

predictions =[]
actual_labels = []
for step, batch in enumerate(tqdm(test_dataloader, desc="Batches processed")):
    if step >= folds_test_samples:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"]}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1)

    # Add to spikes recording.
    spike_record = spikes["Ae"].get("s").permute((1, 0, 2))

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )
    predictions.append(all_activity_pred.item())
    actual_labels.append(label_tensor.item())
    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long()
                                == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.

all_mean_accuracy = accuracy["all"] / folds_test_samples
proportion_mean_accuracy = accuracy["proportion"] / folds_test_samples

print("\nAll accuracy test set: %.2f" % (all_mean_accuracy*100))
print("Proportion weighting test validation set: %.2f \n" %
        (proportion_mean_accuracy))

print("\nEvaluation test set complete.\n")

from sklearn.metrics import confusion_matrix
array = confusion_matrix(actual_labels, predictions)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


df_cm = pd.DataFrame(array, range(10), range(10))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()
######################################