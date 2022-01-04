import os
import traceback

import librosa
import soundfile
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.optim import Adam
import numpy as np
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

rnn_train_options = {
    "batch_size": 512,
    "load_model": False,
    "save_model": True,
    "lr_rate": 0.0005,
    "epochs": 500,
    "base_path": "",
    "num_layers": 2,
    "dataset_info": {
        "train": {"num_samples": 1200, "path": "data/audio-data/tr/tr{file_prefix}{0:04d}.wav"},
        "val": {"num_samples": 1200, "path": "data/audio-data/v/v{file_prefix}{0:04d}.wav"},
        "test": {"num_samples": 400, "path": "data/audio-data/te/te{file_prefix}{0:04d}.wav"}
    },
    "sequence_length": 10,
    "model_path": "saved_models/rnn/rnn.model",
    "train_model": True,
    "clean_data_path": "cleaned_audio"
}

def change_paths_to_absolute(train_ops):
    if train_ops["base_path"] in train_ops["model_path"]:
        return train_ops
    train_ops["model_path"] = train_ops["base_path"] + train_ops["model_path"]
    train_ops["clean_data_path"] = train_ops["base_path"] + train_ops["clean_data_path"]
    for k in train_ops["dataset_info"]:
        train_ops["dataset_info"][k]["path"] = train_ops["base_path"] + train_ops["dataset_info"][k]["path"]

def calculate_model_loss_n_snr(model, loader, clip_mask = False, threshold = 0.5):
    loss_fn = nn.BCELoss()
    total_loss = 0
    model.eval()
    with torch.no_grad():
        total_loss_insts = torch.tensor(0)
        sum_t_2 = torch.tensor(0.)
        dif_s_n = torch.tensor(0.)

        for X_abs, mask, T_abs in loader:
            sum_t_2 += (T_abs * T_abs).sum()
            X_abs = X_abs.to(device)
            mask_pred = model(X_abs).cpu()
            total_loss += loss_fn(mask_pred, mask.view(-1, mask_pred.shape[-1])).item()
            if clip_mask:
                mask_pred = mask_pred > threshold
            dif_s_n += torch.square(T_abs.view(-1, 513) - (X_abs.cpu().view(-1, 513) * mask_pred)).sum()
            total_loss_insts += mask.shape[0] * mask.shape[1]

    return (10 * torch.log10(sum_t_2 / dif_s_n)).item(), total_loss / total_loss_insts

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

        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    except Exception as e:
        traceback.print_exc(e)
        print("Error occured while loading, ignoring...")

def reconstruct_n_save(s_abs, x_abs, x_complex, s_rate, index, save_path):
    s_abs = s_abs.T
    x_abs = x_abs.T
    x_complex = x_complex.T

    s_hat = (x_complex.numpy() / x_abs.numpy()) * s_abs.numpy()
    s_hat_clean = librosa.istft(s_hat, hop_length=512, )
    s_hat_clean = s_hat_clean
    soundfile.write(save_path +"/"+ str(index) + ".wav", s_hat_clean, s_rate)
    # print("saved cleaned output:" + str(index))

def save_model_output(model, loader, save_path):
    batch_size = 512
    model.eval()
    with torch.no_grad():
        dataset = loader.dataset
        mask_preds = torch.zeros((0, dataset.sequence_len, 513))
        batch_start = 0
        while batch_start < dataset.X_abs.shape[0]:
            X_abs = dataset.X_abs[batch_start:min(batch_start+batch_size, dataset.X_abs.shape[0])].to(device)
            mask_pred = model(X_abs).cpu().view(-1, dataset.sequence_len, 513)
            mask_preds = torch.cat((mask_preds, mask_pred), dim=0)
            batch_start += X_abs.shape[0]

        batch_start = 0
        total_aud_len = 0
        for i, (aud_len, pad_len, sr_rate) in enumerate(zip(dataset.aud_lens, dataset.pad_lengths, dataset.sr_rates)):
            X_abs = dataset.X_abs[batch_start: batch_start + aud_len]
            C_abs = X_abs * mask_preds[batch_start: batch_start + aud_len]
            C_abs = C_abs.view(-1, 513)[:-pad_len]
            X_abs = X_abs.view(-1, 513)[:-pad_len]
            X = dataset.X[total_aud_len: total_aud_len + X_abs.shape[0]] * (mask_preds[batch_start: batch_start + aud_len].view(-1, 513)[:-pad_len])
            reconstruct_n_save(C_abs, X_abs, X, sr_rate, i, save_path)

            batch_start += aud_len
            total_aud_len += X_abs.shape[0]

def get_dataloader(train_options, split = "train"): # split = train | val | test
    base_path = train_options["base_path"] + train_options["dataset_info"][split]["path"]
    num_samples = train_options["dataset_info"][split]["num_samples"]
    all_file_paths = [[base_path.format(i, file_prefix = j) for j in [("s", "x", "n"), ("x")][split == "test"]] for i in range(num_samples)] # "{0:04d}".format(10)
    dataset = NoiseDataset(all_file_paths, train_options["sequence_length"])
    return DataLoader(dataset=dataset, batch_size=256, shuffle=split == "train")

def train_rnn_model(train_options):
    model = RNNModel(train_options["num_layers"], in_dim=513, layer_dim=1026, out_dim=513)

    model.to(device)

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    if not train_options["train_model"]:
        return model, None, None

    optimizer = Adam(model.parameters(), lr=train_options["lr_rate"])
    loss_fn = nn.BCELoss()

    train_loader = get_dataloader(train_options, "train")
    val_loader = get_dataloader(train_options, "val")

    epoch_loss = {"val":[], "train":[]}
    epoch_snr = {"val": [], "train": []}
    max_snr, loss = 0, 0#calculate_model_loss_n_snr(model, val_loader) #TODO

    for i in range(train_options["epochs"]):
        total_loss = 0
        total_insts = 0
        sum_t_2 = torch.tensor(0.)
        dif_s_n = torch.tensor(0.)
        model.train()

        for X_abs, mask, T_abs in train_loader:
            X_abs = X_abs.to(device)
            mask_pred = model(X_abs).cpu()
            loss = loss_fn(mask_pred, mask.view(-1, mask_pred.shape[-1]))
            total_loss += loss.item()
            total_insts += X_abs.shape[0] * X_abs.shape[1]
            sum_t_2 += (T_abs * T_abs).sum()
            dif_s_n += torch.square(T_abs.view(-1, 513) - (X_abs.cpu().view(-1, 513) * mask_pred)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_snr, val_loss = calculate_model_loss_n_snr(model, val_loader)
        # train_snr, train_loss = calculate_model_loss_n_snr(model, train_loader)
        train_loss = total_loss / total_insts
        train_snr = (10 * torch.log10(sum_t_2 / dif_s_n)).item()

        if i % 1 == 0:
            print("Current Epoch:" + str(i + 1))
            print("Validation SNR: " + str(val_snr))
            print("Train SNR: " + str(train_snr))
            print("Validation Loss:" + str(val_loss))
            print("Train Loss:" + str(train_loss))
            print("Parameter sum: " + str(sum([a.data.sum().item() for a in model.parameters()])))
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

    save_model_output(model, val_loader, train_options["clean_data_path"])

    return model, epoch_loss, epoch_snr

def plot_epoch_stats(epoch_stat, label):
    for k in epoch_stat:
        plt.plot(epoch_stat[k], "o", label = k, markersize = 2)
    plt.legend()
    plt.title(label)
    plt.show()

def get_linear_layer(in_dim, out_dim, act_fn):
    fc = nn.Linear(in_features=in_dim, out_features=out_dim)
    if act_fn == "relu":
        nn.init.kaiming_normal_(fc.weight.data)
    if act_fn == "sigmoid":
        nn.init.xavier_normal_(fc.weight.data)
    if act_fn == "softmax":
        nn.init.xavier_normal_(fc.weight.data)

    return [fc, get_act_fn(act_fn)]

class NoiseDataset(nn.Module):
    def __init__(self, data_files, sequence_len):
        super(NoiseDataset, self).__init__()
        # self.data_files = data_files # tuples of (X, True, Noise)
        self.sequence_len = sequence_len
        self.X, self.X_abs, self.masks, self.T_abs, self.pad_lengths, self.aud_lens, self.sr_rates = self.read_files(data_files)
        print(self.masks.sum() / self.masks.view(-1).shape[0])
        print("DATASET LOADED.")
        print("size: " + str(self.X_abs.shape))

    def pad_data(self, X_abs, mask):
        pad_len = self.sequence_len - X_abs.shape[0] % self.sequence_len
        X_abs = torch.cat((X_abs, torch.empty(pad_len, 513).fill_(1e-6)), dim=0)
        mask = torch.cat((mask, torch.zeros(pad_len, 513)), dim=0)

        return X_abs, mask, pad_len

    def read_files(self, data_files):
        batch_lengths = []
        pad_lengths = []
        sequence_lengths = []
        all_X = torch.zeros((0, 513))
        all_X_abs = torch.zeros((0, self.sequence_len, 513))
        all_T_abs = torch.zeros((0, self.sequence_len, 513))
        all_masks = torch.zeros((0, self.sequence_len, 513))
        sr_rates = []
        for i in range(len(data_files)):
            if len(data_files[i]) == 1: # For test data
                X, X_abs, sr_rate = get_audio_data(data_files[i][0])
                mask = torch.zeros(X.shape)
                X_abs, mask, pad_len = self.pad_data(X_abs, mask)
                X_abs = X_abs.view(-1, self.sequence_len, X_abs.shape[-1])
                mask = mask.view(-1, self.sequence_len, mask.shape[-1])
                pad_lengths.append(pad_len)
                batch_lengths.append(X_abs.shape[0])
                all_X = torch.cat((all_X, X), dim=0)
                all_X_abs = torch.cat((all_X_abs, X_abs), dim=0)
                all_masks = torch.cat((all_masks, mask), dim=0)
                sr_rates.append(sr_rate)

            else:
                T, T_abs, sr_rate = get_audio_data(data_files[i][0])
                X, X_abs, sr_rate = get_audio_data(data_files[i][1])
                N, N_abs, sr_rate = get_audio_data(data_files[i][2])
                mask = T_abs > N_abs

                X_abs, mask, pad_len = self.pad_data(X_abs, mask)
                T_abs = torch.cat((T_abs, torch.zeros((pad_len, 513))))
                X_abs = X_abs.view(-1, self.sequence_len, X_abs.shape[-1])
                T_abs = T_abs.view(-1, self.sequence_len, T_abs.shape[-1])
                mask = mask.view(-1, self.sequence_len, mask.shape[-1])


                pad_lengths.append(pad_len)
                batch_lengths.append(X_abs.shape[0])
                sequence_lengths.append(len(T_abs))

                all_X = torch.cat((all_X, X), dim=0)
                all_X_abs = torch.cat((all_X_abs, X_abs), dim=0)
                all_T_abs = torch.cat((all_T_abs, T_abs), dim=0)
                all_masks = torch.cat((all_masks, mask), dim=0)
                sr_rates.append(sr_rate)

        return all_X, all_X_abs, all_masks, all_T_abs, pad_lengths, batch_lengths, sr_rates

    def __len__(self):
        return self.X_abs.shape[0]

    def __getitem__(self, item):
        return self.X_abs[item], self.masks[item], self.T_abs[item]

class RNNModel(nn.Module):
    def __init__(self, num_layers, in_dim = 513, layer_dim = 1026, out_dim = 513, dropout = 0):
        super(RNNModel, self).__init__()
        self.layer_dim = layer_dim
        self.seq_layers = nn.LSTM(input_size=in_dim, hidden_size=layer_dim, batch_first=True, dropout=dropout, num_layers = num_layers )
        [torch.nn.init.xavier_normal_(j.data)  for i in self.seq_layers.all_weights for j in i if len(j.data.shape) > 1]
        self.last_layer = nn.Sequential(*get_linear_layer(layer_dim, out_dim, "sigmoid"))

    def forward(self, samples): # B, N, 513
        output, temp = self.seq_layers(samples)
        output = output.reshape(-1, self.layer_dim)
        return self.last_layer(output)

def search_params(train_options):
    lr = 0.01
    train_options["epochs"] = 5
    for i in range(5):
        train_options["lr_rate"] = lr
        lr = lr / 5
        model, loss, snr = train_rnn_model(train_options)
        print("LR: " + str(train_options["lr_rate"]) + " SNRs: " + str(snr))

# rnn_train_options["epochs"] = 200
#
# model, x, y = train_rnn_model(rnn_train_options)
# loader = get_dataloader(rnn_train_options, "test")
#
# save_model_output(model, loader, rnn_train_options["clean_data_path"])
#

#________________

change_paths_to_absolute(rnn_train_options)
rnn_train_options["train_model"] = False
rnn_train_options["load_model"] = True
rnn_train_options["epochs"] = 0
model, loss, snr = train_rnn_model(rnn_train_options)

loader = get_dataloader(rnn_train_options, "val")
# save_model_output(model, loader, rnn_train_options["clean_data_path"])
snr, loss = calculate_model_loss_n_snr(model, loader)
print(snr)

print("Done.")
