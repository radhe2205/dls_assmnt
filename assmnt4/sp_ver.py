import os
import pickle
import random
import traceback
from itertools import combinations

import librosa
import soundfile
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.optim import Adam
import numpy as np
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_audio_data(file_path):
    s = pickle.load(open(file_path, "rb"))
    S_complex = librosa.stft(s[0], n_fft=1024, hop_length=512)
    S_complex_final, S_abs_final = torch.from_numpy(S_complex.T).unsqueeze(0), torch.from_numpy(np.abs(S_complex).T).unsqueeze(0)

    for i in range(1, s.shape[0]):
        S_complex = librosa.stft(s[i], n_fft=1024, hop_length=512)
        S_complex_final = torch.cat((S_complex_final, torch.from_numpy(S_complex.T).unsqueeze(0)), dim=0)
        S_abs_final = torch.cat((S_abs_final, torch.from_numpy(np.abs(S_complex).T).unsqueeze(0)), dim=0)

    return S_complex_final, S_abs_final

def get_samples_from_same_speaker(num_samples = 10, example_per_speaker = 45):
    return [a for a in combinations(range(num_samples), 2)]

def get_positive_pair_samples(signal, pair_per_speaker = 45): # Signal 500 x T x 513
    sample_per_speakers = 10
    num_speakers = int(signal.shape[0] / sample_per_speakers)
    final_shape = [pair_per_speaker * num_speakers, 2, signal.shape[1], 513] #(pair_per_speaker x num_speakers) x 2T x 513
    all_samples = torch.zeros(final_shape)

    for i in range(num_speakers):
        combs = get_samples_from_same_speaker(num_samples=10, example_per_speaker=45)
        for j, comb in enumerate(combs):
            all_samples[i * pair_per_speaker + j] = torch.cat(((signal[i * 10 + comb[0]].unsqueeze(0)), (signal[i * 10 + comb[1]].unsqueeze(0))), dim=0)
            # all_samples[i * pair_per_speaker + j] = torch.cat((torch.zeros(signal[i * 10 + comb[0]].unsqueeze(0).shape).fill_(2.),
            #                                                    torch.zeros(signal[i * 10 + comb[1]].unsqueeze(0).shape).fill_(2.)), dim=0)

    return all_samples

def get_negative_pair_samples(signal, pair_per_speaker = 45): # Signal 500 x T x 513
    sample_per_speakers = 10
    num_speakers = int(signal.shape[0] / sample_per_speakers)
    final_shape = [pair_per_speaker * num_speakers, 2, signal.shape[1], 513] #(pair_per_speaker x num_speakers) x 2T x 513
    all_samples = torch.zeros(final_shape)

    for cur_speaker in range(num_speakers):
        first_speaker_samples = [cur_speaker*sample_per_speakers + i % sample_per_speakers for i in range(pair_per_speaker)]
        other_speakers_idxes = [i % num_speakers for i in range(pair_per_speaker * 2) if i % num_speakers != cur_speaker][:pair_per_speaker]
        other_speaker_samples = [speaker_idx*sample_per_speakers + random.choice(range(sample_per_speakers)) for speaker_idx in other_speakers_idxes]
        for i in range(pair_per_speaker):
            all_samples[cur_speaker * pair_per_speaker + i] = torch.cat((signal[first_speaker_samples[i]].unsqueeze(0), signal[other_speaker_samples[i]].unsqueeze(0)), dim=0)
            # all_samples[cur_speaker * pair_per_speaker + i] = torch.cat((torch.zeros(signal[first_speaker_samples[i]].unsqueeze(0).shape).fill_(0.),
            #                                                              torch.zeros(signal[other_speaker_samples[i]].unsqueeze(0).shape).fill_(1.)), dim=0)
    return all_samples

class SpeakerIdenModel(nn.Module):
    def __init__(self):
        super(SpeakerIdenModel, self).__init__()

        self.rnn_layers = nn.GRU(input_size=513, num_layers=2, hidden_size=513, batch_first=True)
        # [torch.nn.init.xavier_normal_(j.data) for i in self.rnn_layers.all_weights for j in i if len(j.data.shape) > 1]

        self.conv_layers = nn.Sequential(
            nn.Conv1d(kernel_size = 3, in_channels=1, out_channels=1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.BatchNorm1d(num_features=128),
            nn.Linear(in_features=128, out_features=64),
            nn.Tanh()
        )

        self.last_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=1),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

        for layer in self.fc_layers:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, samples): # batch, seq * 2, 513

        o1n, temp = self.rnn_layers(samples[:, 0, ...])
        o2n, temp = self.rnn_layers(samples[:, 1, ...])

        o1n = self.fc_layers(o1n[:, -1, :])
        o2n = self.fc_layers(o2n[:, -1, :])

        vec = self.sigmoid((o1n * o2n).sum(-1)).unsqueeze(-1)

        return self.last_layer(vec).squeeze(-1)

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
        # traceback.print_exc(e)
        print("Error occured while loading, ignoring...")

def get_val_accuracy(model, loader):
    loss_fn = nn.BCELoss()
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total = 0
        total_correct = 0
        for X, Y in loader:
            pred = model(X.to(device)).cpu()
            total_loss += loss_fn(pred, Y)
            acc = (pred > 0.5) == Y
            total += X.size(0)
            total_correct += acc.sum().item()

    test_acc = total_correct / total
    test_loss = total_loss / loader.dataset.__len__()
    return test_acc, test_loss

class SpeakerDataset(nn.Module):
    def __init__(self, data, labels):
        super(SpeakerDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

def generate_dataloader_from_audios(audio_abs, pair_per_speaker, split, batch_size):
    data = get_positive_pair_samples(audio_abs, pair_per_speaker)
    data = torch.cat((data, get_negative_pair_samples(audio_abs, pair_per_speaker)), dim=0)

    labels = torch.cat((torch.ones(int(data.shape[0] / 2)), torch.zeros(int(data.shape[0] / 2))), dim=0)

    dataset = SpeakerDataset(data, labels)
    return DataLoader(dataset = dataset, shuffle=split=="train", batch_size=batch_size)

def get_dataloader(train_options, split):
    data_complex, data_abs = get_audio_data((train_options["test_data_path"], train_options["train_data_path"])[split == "train"])
    for i in range(int(data_abs.shape[0] / 10)):
        data_abs[i*10:i*10 + 10] = i
    return generate_dataloader_from_audios(data_abs, train_options["pair_per_speaker"], split, train_options["batch_size"])

def train_speaker_model(train_options):
    model = SpeakerIdenModel()

    model.to(device)

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    if not train_options["train_model"]:
        return model

    optimizer = torch.optim.Adam(model.parameters(), eps=train_options["lr_rate"])

    train_loader = get_dataloader(train_options, "train")
    test_loader = get_dataloader(train_options, "test")

    loss_fn = torch.nn.BCELoss()

    max_acc, temp = get_val_accuracy(model, test_loader)
    print("Initial Acc: " + str(max_acc))
    print("Initial Loss: " + str(temp))

    for epoch in range(1, train_options["epochs"] + 1):
        print("Epoch: " + str(epoch))
        model.train()
        total_loss = 0
        for i, (X, Y) in enumerate(train_loader):
            pred = model(X.to(device)).cpu()
            loss = loss_fn(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        test_acc, test_loss = get_val_accuracy(model, test_loader)
        if test_acc >= max_acc:
            max_acc = test_acc
            save_model(model, train_options["model_path"])

        print("Train loss: " + str(total_loss / train_loader.dataset.__len__()))
        print("Test Accuracy: " + str(test_acc))
        print("Test Loss: " + str(test_loss))

        # if test_acc >= train_options["cap_accuracy"]:
        #     save_model(model, train_options["model_path"])
        #     return model

train_options = {
    "lr_rate": 0.0000001,
    "batch_size": 64,
    "epochs": 100,
    "base_path": "",
    "model_path": "saved_models/speak_ver.model",
    "save_model": True,
    "load_model": False,
    "train_model": True,
    "train_data_path": "data/speaker_ver/hw4_trs.pkl",
    "test_data_path": "data/speaker_ver/hw4_tes.pkl",
    "pair_per_speaker": 45
}

train_speaker_model(train_options)

# data = pickle.load(open("data/speaker_ver/hw4_trs.pkl", "rb"))
# data_complex = librosa.stft(data[0], n_fft=1024, hop_length=512)
#
# print("Done")

