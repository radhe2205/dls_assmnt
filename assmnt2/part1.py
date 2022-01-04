import copy
import os
import traceback

import librosa
import numpy as np
import soundfile
import torch.nn as nn
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dim_after_kernels_n_stride_no_pad(in_dim, kernel_stride_seq):
    for seq in kernel_stride_seq:
        kernel_size = seq[0]
        stride = seq[1]
        last_kernel_start = in_dim - kernel_size + 1
        in_dim = int((last_kernel_start - 1) / stride) + 1
    return in_dim

def get_act_fn(act_fn):
    if act_fn == "sigmoid":
        return nn.Sigmoid()
    elif act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "leaky_relu":
        return nn.LeakyReLU()
    elif act_fn == "ELU":
        return nn.ELU()
    elif act_fn == "softplus":
        return nn.Softplus(beta=1)

def get_audio_data(file_path):
    s, sr = librosa.load(file_path, sr=None)
    S = librosa.stft(s, n_fft=1024, hop_length=512)
    return torch.from_numpy(S.T), torch.from_numpy(np.abs(S).T), sr

def get_dataset_from_files(noise_file, true_file):
    S, S_abs, s_rate = get_audio_data(true_file)
    X, X_abs, s_rate = get_audio_data(noise_file)

    return S, X, S_abs, X_abs, s_rate

def plot_signals(signals, labels):
    f, axarr = plt.subplots(1, len(signals), figsize=(8,8))
    axarr = axarr.flatten()

    for i, signal in enumerate(signals):
        for vals in signal:
            axarr[i].plot(vals, "bo", markersize=3)
            axarr[i].title.set_text(labels[i])
    plt.suptitle("Clean vs Noise Signal")
    plt.show()

class FCNet(nn.Module):
    def __init__(self, mid_act, last_act, num_layers, in_dim, layer_dim):
        super(FCNet, self).__init__()
        self.layers = nn.Sequential(
            *self.get_fc_layer(mid_act, in_dim, layer_dim, drop=0.2),
            # *self.get_fc_layer(mid_act, layer_dim, layer_dim, drop=0.5),
            *self.get_fc_layer(last_act, layer_dim, in_dim, drop=0.5)
        )

    def forward(self, batch_samples):
        return self.layers(batch_samples)

    def get_fc_layer(self, act_fn, in_dim, out_dim, drop):
        layers = [nn.Linear(in_features=in_dim, out_features = out_dim), get_act_fn(act_fn), nn.Dropout(0)]
        fc = layers[0]
        if act_fn == "relu" or act_fn == "leaky_relu" or act_fn == "softplus":
            torch.nn.init.kaiming_normal_(fc.weight.data)
        elif act_fn == "sigmoid":
            torch.nn.init.xavier_normal_(fc.weight.data)

        return layers

class AudioDataset(nn.Module):
    def __init__(self, true_sig, noise_sig):
        super(AudioDataset, self).__init__()
        self.true_sig = true_sig
        self.noise_sig = noise_sig

    def __len__(self):
        return self.true_sig.shape[0]

    def __getitem__(self, idx):
        return self.noise_sig[idx], self.true_sig[idx]

fc_train_options = {
    "batch_size": 64,
    "load_model": False,
    "save_model": True,
    "layer_dim": 1024,
    "num_layers": 4,
    "mid_act_fn": "relu",
    "last_act_fn": "softplus",
    "lr_rate": 0.001,
    "epochs": 500,
    "base_path": "",
    "model_path": "saved_models/fc/fc.model",
    "noise_file": "data/train_dirty_male.wav",
    "clean_file": "data/train_clean_male.wav",
    "test_files": ["data/test_x_01.wav", "data/test_x_02.wav"],
    "train_model": True
}

def save_model(model, model_path):
    directory_path = "/".join(model_path.split("/")[:-1])
    if len(directory_path) > 0:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    torch.save(model.state_dict(), model_path)

def load_model(model, model_path):
    try:
        if not os.path.exists(model_path):
            return model

        model.load_state_dict(torch.load(model_path))
        return model
    except Exception as e:
        traceback.print_exc(e)
        print("Error occured while loading, ignoring...")

def reconstruct_n_save(s_abs, x_abs, x_complex, s_rate, file_name):
    s_abs = s_abs.T
    x_abs = x_abs.T
    x_complex = x_complex.T

    s_hat = (x_complex.numpy() / x_abs.numpy()) * s_abs.numpy()
    s_hat_clean = librosa.istft(s_hat, hop_length=512, )
    s_hat_clean = s_hat_clean
    soundfile.write(file_name, s_hat_clean, s_rate)
    print("saved cleaned output:" + file_name)

def get_model_output(noise_data, model):
    output = torch.zeros((0, noise_data.shape[1]))

    model.eval()
    with torch.no_grad():
        for i in range(0, noise_data.shape[0], 64):
            end = min(noise_data.shape[0], i+64)
            batch = noise_data[i:end]
            output = torch.cat((output, model(batch.to(device)).cpu()), dim=0)
    return output

def save_model_output(noise_data, noise_complex, model, s_rate, path, model_type):
    if model_type == "conv2d":
        output = get_model_output_conv2d(noise_data, model)
    else:
        output = get_model_output(noise_data, model)
    reconstruct_n_save(output, noise_data, noise_complex, s_rate, path)

def get_dataloader(noise_file, true_file, train_options):
    true_complex, noise_complex, true_abs, noise_abs, s_rate = get_dataset_from_files(noise_file, true_file)
    # plot_signals([true_abs, noise_abs], ["True Signal", "Noise Signal"])
    rand_idx = np.arange(noise_abs.shape[0])
    pivot = int(noise_abs.shape[0] * .15)
    np.random.shuffle(rand_idx)

    validation_idx, train_idx = rand_idx[:pivot], rand_idx[pivot:]

    train_dataset = AudioDataset(true_abs[train_idx], noise_abs[train_idx])
    val_dataset = AudioDataset(true_abs[validation_idx], noise_abs[validation_idx])
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=train_options["batch_size"], drop_last=False)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=train_options["batch_size"], drop_last=False)
    return train_loader, val_loader, true_abs.shape[1], s_rate

def calculate_model_loss_n_snr(model, val_loader):

    loss_fn = nn.MSELoss()
    total_loss = 0
    model.eval()
    with torch.no_grad():
        sum_t_2 = torch.tensor(0.)
        dif_s_n = torch.tensor(0.)

        for n_sig, t_sig in val_loader:
            sum_t_2 += (t_sig * t_sig).sum()
            n_sig = n_sig.to(device)
            pred_t_sig = model(n_sig).cpu()
            total_loss += loss_fn(pred_t_sig, t_sig).item()
            dif_s_n += torch.square(t_sig - pred_t_sig).sum()
    total_loss = total_loss / len(val_loader)
    return (10 * torch.log10(sum_t_2 / dif_s_n)).item(), total_loss

def change_path_to_absolute(train_options):
    if train_options["base_path"] in train_options["model_path"]:
        return train_options

    train_options["model_path"] = train_options["base_path"] + train_options["model_path"]
    train_options["noise_file"] = train_options["base_path"] + train_options["noise_file"]
    train_options["clean_file"] = train_options["base_path"] + train_options["clean_file"]
    for i in range(len(train_options["test_files"])):
        train_options["test_files"][i] = train_options["base_path"] + train_options["test_files"][i]
    return train_options

def train_fc_model(train_options):
    train_loader, val_loader, in_dim, s_rate  = get_dataloader(train_options["noise_file"], train_options["clean_file"], train_options)
    model = FCNet(train_options["mid_act_fn"], train_options["last_act_fn"], train_options["num_layers"],
                  in_dim, train_options["layer_dim"])

    model.to(device)

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    if not train_options["train_model"]:
        return model

    optimizer = Adam(model.parameters(), lr=train_options["lr_rate"])
    loss_fn = nn.MSELoss()

    epoch_loss = {"val":[], "train":[]}
    epoch_snr = {"val": [], "train": []}
    max_snr, loss = calculate_model_loss_n_snr(model, val_loader)

    for i in range(train_options["epochs"]):
        total_loss = 0
        model.train()

        for n_sig, t_sig in train_loader:
            n_sig = n_sig.to(device)
            t_pred_sig = model(n_sig).cpu()
            loss = loss_fn(t_pred_sig, t_sig)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_snr, val_loss = calculate_model_loss_n_snr(model, val_loader)
        train_snr, train_loss = calculate_model_loss_n_snr(model, train_loader)

        if i % 50 == 0:
            print("Current Epoch:" + str(i + 1))
            print("Validation SNR: " + str(val_snr))
            print("Train SNR: " + str(train_snr))
            print("Validation Loss:" + str(val_loss))
            print("Train Loss:" + str(train_loss))

            # print("Parameter sum: " + str(sum([a.data.sum().item() for a in model.parameters()])))
            print("----------------------------------")
        epoch_loss["val"].append(val_loss)
        epoch_loss["train"].append(train_loss)
        epoch_snr["val"].append(val_snr)
        epoch_snr["train"].append(train_snr)

        if val_snr >= max_snr:
            save_model(model, train_options["model_path"])
            max_snr = val_snr

    if train_options["save_model"] and train_options["model_path"] is not None:
        model = load_model(model, train_options["model_path"])

    return model, epoch_loss, epoch_snr

def plot_epoch_stats(epoch_stat, label):
    for k in epoch_stat:
        plt.plot(epoch_stat[k], "o", label = k, markersize = 2)
    plt.legend()
    plt.title(label)
    plt.show()

def train_test_model_on_noise_files(train_options, model_type ="fc", train_model = True):

    train_options = change_path_to_absolute(train_options)

    train_options["train_model"] = train_model
    if model_type == "fc":
        model, epoch_loss, epoch_snr = train_fc_model(train_options)
    elif model_type == "conv1d":
        model, epoch_loss, epoch_snr = train_conv1d_model(train_options)
    elif model_type == "conv2d":
        model, epoch_loss, epoch_snr = train_conv2d_model(train_options)

    plot_epoch_stats(epoch_loss, "Train and validation Loss.")
    plot_epoch_stats(epoch_snr, "Train and Validation SNR.")

    noise_files = [train_options["noise_file"]] + train_options["test_files"]
    for file_name in noise_files:
        noise_complex, noise_abs, s_rate = get_audio_data(file_name)
        save_model_output(noise_abs, noise_complex, model, s_rate, file_name.replace("data", "cleaned_data/" + model_type),  model_type)

# train_test_model_on_noise_files(fc_train_options)

class OneDCNN(nn.Module):
    def __init__(self, mid_act_fn, last_act_fn, in_dim, out_dim):
        super(OneDCNN, self).__init__()
        self.layers = nn.Sequential(
            *self.get_convolutional_block(conv_stride=2, conv_kernel=7, in_channels=1, out_channels=64, pool_kernel=2, pool_stride=2, act_fn=mid_act_fn),
            *self.get_convolutional_block(conv_stride=2, conv_kernel=5, in_channels=64, out_channels=128, pool_kernel=2, pool_stride=2, act_fn=mid_act_fn),
            nn.Flatten(),
            nn.Linear(in_features=3968, out_features=513),
            get_act_fn(last_act_fn)
        )

    def get_convolutional_block(self, conv_stride, conv_kernel, in_channels, out_channels, act_fn, pool_stride, pool_kernel):
        conv1d = nn.Conv1d(kernel_size=conv_kernel, stride=conv_stride, in_channels=in_channels, out_channels = out_channels)
        mxpool = nn.MaxPool1d(stride=pool_stride, kernel_size=pool_kernel)
        return [conv1d, mxpool, get_act_fn(act_fn)]

    def forward(self, samples):
        samples = samples.unsqueeze(-2)
        return self.layers(samples)

def train_conv1d_model(train_options):
    train_loader, val_loader, in_dim, s_rate  = get_dataloader(train_options["noise_file"], train_options["clean_file"], train_options)
    model = OneDCNN(train_options["mid_act_fn"], train_options["last_act_fn"],
                  in_dim, in_dim)

    model.to(device)

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    if not train_options["train_model"]:
        return model

    optimizer = Adam(model.parameters(), lr=train_options["lr_rate"])
    loss_fn = nn.MSELoss()

    epoch_loss = []
    epoch_snr = [calculate_model_loss_n_snr(model, val_loader)]
    for i in range(train_options["epochs"]):
        total_loss = 0
        model.train()

        epoch_loss = {"val":[], "train":[]}
        epoch_snr = {"val": [], "train": []}
        max_snr, loss = calculate_model_loss_n_snr(model, val_loader)

        for n_sig, t_sig in train_loader:
            n_sig = n_sig.to(device)
            t_pred_sig = model(n_sig).cpu()
            loss = loss_fn(t_pred_sig, t_sig)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_snr, val_loss = calculate_model_loss_n_snr(model, val_loader)
        train_snr, train_loss = calculate_model_loss_n_snr(model, train_loader)

        if i % 50 == 0:
            print("Current Epoch:" + str(i + 1))
            print("Validation SNR: " + str(val_snr))
            print("Train SNR: " + str(train_snr))
            print("Validation Loss:" + str(val_loss))
            print("Train Loss:" + str(train_loss))

            # print("Parameter sum: " + str(sum([a.data.sum().item() for a in model.parameters()])))
            print("----------------------------------")
        epoch_loss["val"].append(val_loss)
        epoch_loss["train"].append(train_loss)
        epoch_snr["val"].append(val_snr)
        epoch_snr["train"].append(train_snr)

        if val_snr >= max_snr:
            save_model(model, train_options["model_path"])
            max_snr = val_snr

    if train_options["save_model"] and train_options["model_path"] is not None:
        model = load_model(model, train_options["model_path"])

    return model, epoch_loss, epoch_snr

conv1d_train_options = copy.deepcopy(fc_train_options)
# train_test_model_on_noise_files(conv1d_train_options, model_type="conv1d", train_model=True)
conv1d_train_options["lr_rate"] = 0.0001
conv1d_train_options["model_path"] = "saved_models/conv1/conv1d.model"

def get_model_output_conv2d(noise_data, model):
    true_data = torch.rand(noise_data.shape)
    dataset = AudioImgDataset(true_data, noise_data, np.arange(true_data.shape[0]))
    dataloader = DataLoader(dataset = dataset, shuffle=False, batch_size=64)
    output = torch.zeros((0, noise_data.shape[1]))

    model.eval()
    with torch.no_grad():
        for n_sig, t_sig in dataloader:
            output = torch.cat((output, model(n_sig.to(device)).cpu()), dim = 0)
    return output

def get_dataloader_for_conv2d(noise_file, true_file, train_options, split = True, shuffle = True):
    true_complex, noise_complex, true_abs, noise_abs, s_rate = get_dataset_from_files(noise_file, true_file)
    rand_idx = np.arange(noise_abs.shape[0])
    if split:
        pivot = int(noise_abs.shape[0] * .15)
    else:
        pivot = 0

    np.random.shuffle(rand_idx)

    validation_idx, train_idx = rand_idx[:pivot], rand_idx[pivot:]
    train_dataset = AudioImgDataset(true_abs, noise_abs, train_idx)
    val_dataset = AudioImgDataset(true_abs, noise_abs, validation_idx)

    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=train_options["batch_size"], drop_last=False)
    train_loader = DataLoader(dataset=train_dataset, shuffle=shuffle, batch_size=train_options["batch_size"], drop_last=False)

    return train_loader, val_loader, true_abs.shape[1], s_rate

def train_conv2d_model(train_options):
    train_loader, val_loader, in_dim, s_rate  = get_dataloader_for_conv2d(train_options["noise_file"], train_options["clean_file"], train_options)
    model = TwoDCNN(conv2d_kernels_strides)

    model.to(device)

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    if not train_options["train_model"]:
        return model
    optimizer = Adam(model.parameters(), lr=train_options["lr_rate"])
    loss_fn = nn.MSELoss()

    epoch_loss = {"val": [], "train":[]}
    epoch_snr = {"val": [], "train": []}

    max_snr = calculate_model_loss_n_snr(model, val_loader)

    for i in range(train_options["epochs"]):
        total_loss = 0
        model.train()


        for n_sig, t_sig in train_loader:
            n_sig = n_sig.to(device)
            t_pred_sig = model(n_sig).cpu()
            loss = loss_fn(t_pred_sig, t_sig)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_snr, val_loss = calculate_model_loss_n_snr(model, val_loader)
        train_snr, train_loss = calculate_model_loss_n_snr(model, train_loader)

        if i % 50 == 0:
            print("Current Epoch:" + str(i + 1))
            print("Validation SNR: " + str(val_snr))
            print("Train SNR: " + str(train_snr))
            print("Validation Loss:" + str(val_loss))
            print("Train Loss:" + str(train_loss))

            # print("Parameter sum: " + str(sum([a.data.sum().item() for a in model.parameters()])))
            print("----------------------------------")
        epoch_loss["val"].append(val_loss)
        epoch_loss["train"].append(train_loss)
        epoch_snr["val"].append(val_snr)
        epoch_snr["train"].append(train_snr)

        if val_snr >= max_snr:
            save_model(model, train_options["model_path"])
            max_snr = val_snr

    if train_options["save_model"] and train_options["model_path"] is not None:
        model = load_model(model, train_options["model_path"])

    return model, epoch_loss, epoch_snr


class AudioImgDataset(nn.Module):
    def __init__(self, true_sig, noise_sig, idxes, window_size = 20):
        super(AudioImgDataset, self).__init__()
        self.true_sig = true_sig
        self.noise_sig = noise_sig
        self.idxes = idxes
        self.window_size = window_size

    def __len__(self):
        return self.idxes.shape[0]

    def get_random_noise(self, idx):
        if idx >= self.window_size:
            return torch.zeros((0, self.noise_sig.shape[1]))

        return torch.rand((self.window_size - idx - 1, self.noise_sig.shape[1])) * 1e-5

    def __getitem__(self, idx):
        return torch.flip(torch.cat((self.get_random_noise(self.idxes[idx]), self.noise_sig[max(0, self.idxes[idx] - self.window_size + 1): self.idxes[idx] + 1]), dim=0), dims=[0]), self.true_sig[self.idxes[idx]]

conv2d_kernels_strides = [
    {"type": "conv", "kernel": (5, 2), "stride": (5, 1), "channels": (1, 3), "act_fn": "relu"},
    {"type": "maxpool", "kernel": (1, 1), "stride": (1, 1)},
]

def get_cnn_layers_from_config(config, dim0 = 20, dim1 = 513):
    all_layers = []
    channels = 1
    for layer in config:
        if layer["type"] == "conv":
            conv = nn.Conv2d(kernel_size = layer["kernel"], stride=layer["stride"], in_channels=layer["channels"][0], out_channels=layer["channels"][1])
            all_layers.append(conv)
            nn.init.kaiming_normal_(conv.weight.data)
            all_layers.append(get_act_fn(layer["act_fn"]))
            channels = layer["channels"][1]
        if layer["type"] == "maxpool":
            all_layers.append(nn.MaxPool2d(kernel_size=layer["kernel"], stride=layer["stride"]))

        dim0 = get_dim_after_kernels_n_stride_no_pad(dim0, [(layer["kernel"][0], layer["stride"][0])])
        dim1 = get_dim_after_kernels_n_stride_no_pad(dim1, [(layer["kernel"][1], layer["stride"][1])])
    return all_layers, dim0, dim1, channels

class TwoDCNN(nn.Module):

    def __init__(self, conv2d_model_config):
        super(TwoDCNN, self).__init__()

        self.conv_layers, dim0, dim1, channels = get_cnn_layers_from_config(conv2d_model_config)
        print("FC dim:" + str(dim1 * channels * dim0))

        self.fc1 = nn.Linear(in_features=dim0 * dim1 * channels, out_features=1026)
        self.fc2 = nn.Linear(in_features=1026, out_features=513)

        self.layers = nn.Sequential(
            *self.conv_layers,
            nn.Flatten(),
            nn.Dropout(0.05),
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU()
        )

        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.kaiming_normal_(self.fc2.weight.data)

    def forward(self, samples):
        return self.layers(samples.unsqueeze(1)).squeeze(-1).squeeze(-1)

conv2d_train_ops = copy.deepcopy(fc_train_options)
conv2d_train_ops["train_model"] = True
conv2d_train_ops["model_path"] = "saved_models/conv2d.model"
conv2d_train_ops["load_model"] = False
conv2d_train_ops["batch_size"] = 32
conv2d_train_ops["lr_rate"] = 0.0001
conv2d_train_ops["epochs"] = 300
train_test_model_on_noise_files(conv2d_train_ops, model_type="fc", train_model=True)
