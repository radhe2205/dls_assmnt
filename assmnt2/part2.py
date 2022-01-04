import os

import librosa
import numpy as np
import soundfile
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

def get_audio_data(file_path):
    s, sr = librosa.load(file_path, sr=None)
    S = librosa.stft(s, n_fft=1024, hop_length=512)
    return torch.from_numpy(S.T), torch.from_numpy(np.abs(S).T), sr

def get_dataset_from_files(noise_file, true_file):
    S, S_abs, s_rate = get_audio_data(true_file)
    X, X_abs, s_rate = get_audio_data(noise_file)

    return S, X, S_abs, X_abs, s_rate


class FCNet(nn.Module):
    def __init__(self, mid_act, last_act, num_layers, in_dim, layer_dim):
        super(FCNet, self).__init__()
        self.layers = nn.Sequential(
                        *self.get_fc_layer(mid_act, in_dim, layer_dim),
                        *self.get_fc_layer(mid_act, layer_dim, layer_dim),
                        *self.get_fc_layer(mid_act, layer_dim, layer_dim),
                        *self.get_fc_layer(last_act, layer_dim, in_dim)
                    )

    def forward(self, batch_samples):
        return self.layers(batch_samples)

    def get_act_fn(self, act_fn):
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

    def get_fc_layer(self, act_fn, in_dim, out_dim):
        layers = [nn.Linear(in_features=in_dim, out_features = out_dim), self.get_act_fn(act_fn), nn.Dropout(0.25)]
        fc = layers[0]
        if act_fn == "relu" or act_fn == "leaky_relu":
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
    "save_model": False,
    "layer_dim": 1024,
    "num_layers": 4,
    "mid_act_fn": "relu",
    "last_act_fn": "relu",
    "lr_rate": 0.001,
    "epochs": 500,
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
    if not os.path.exists(model_path):
        return model

    model.load_state_dict(torch.load(model_path))
    return model

def reconstruct_n_save(s_abs, x_abs, x_complex, s_rate, file_name):
    s_abs = s_abs.T
    x_abs = x_abs.T
    x_complex = x_complex.T

    s_hat = (x_complex.numpy() / x_abs.numpy()) * s_abs.numpy()
    s_hat_clean = librosa.istft(s_hat, hop_length=512, )
    s_hat_clean = s_hat_clean
    soundfile.write(file_name, s_hat_clean, s_rate)

def get_model_output(noise_data, model):
    model.eval()
    output = torch.zeros((0, noise_data.shape[1]))
    with torch.no_grad():
        for i in range(0, noise_data.shape[0], 64):
            end = min(noise_data.shape[0], i+64)
            batch = noise_data[i:end]
            output = torch.cat((output, model(batch)), dim=0)
    return output

def save_model_output(noise_data, noise_complex, model, path):
    output = get_model_output(noise_data, model)
    reconstruct_n_save(output, noise_data, noise_complex, s_rate, path)

def get_dataloader(noise_file, true_file):
    true_complex, noise_complex, true_abs, noise_abs, s_rate = get_dataset_from_files(noise_file, true_file)
    dataset = AudioDataset(true_abs, noise_abs)
    return DataLoader(dataset=dataset, shuffle=True, batch_size=fc_train_options["batch_size"], drop_last=False), true_abs.shape[1], s_rate

def train_fc_model(train_options):

    dataloader, in_dim, s_rate  = get_dataloader(train_options["noise_file"], train_options["clean_file"])
    model = FCNet(train_options["mid_act_fn"], train_options["last_act_fn"], train_options["num_layers"],
                  in_dim, train_options["layer_dim"])

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    if not train_options["train_model"]:
        return model

    optimizer = Adam(model.parameters(), lr=train_options["lr_rate"])
    loss_fn = nn.MSELoss()

    model.eval()

    epoch_loss = []

    # save_model_output(X_, X, model, "cleaned_data/file.wav")

    for i in range(train_options["epochs"]):
        print("Epoch:" + str(i+1))

        total_loss = 0
        for n_sig, t_sig in dataloader:
            optimizer.zero_grad()
            t_pred_sig = model(n_sig)
            loss = loss_fn(t_pred_sig, t_sig)
            total_loss += loss

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        print("Total Loss:" + str(total_loss))
        if epoch_loss[-1] <= min(epoch_loss):
            save_model(model, train_options["model_path"])

    return model

model = train_fc_model(fc_train_options)
noise_complex, noise_abs, s_rate = get_audio_data(fc_train_options["noise_file"])
save_model_output(noise_abs, noise_complex, model, "cleaned_data/train_cleaned_relu.wav")
