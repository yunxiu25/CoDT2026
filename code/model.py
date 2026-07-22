import torch
import torch.nn as nn


class   DQ(nn.Module):
    def __init__(self, input_dim, dim, n_embedding, m_book, mask_ratio=0.2):
        super(DQ, self).__init__()
        self.m_book = m_book
        self.encoders = nn.ModuleList()
        self.codebooks = nn.ModuleList()
        for m in range(m_book):
            codebook = nn.Embedding(n_embedding, dim)
            codebook.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
            encoder = nn.Sequential(
                nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.Dropout(0.5), nn.ReLU(),
                nn.Linear(128, 256), nn.BatchNorm1d(256), nn.Dropout(0.2), nn.ReLU(),
                nn.Linear(256, dim),
            )
            self.codebooks.append(codebook)
            self.encoders.append(encoder)

        self.pos = nn.Embedding(1, input_dim)
        self.pos.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
        self.mask_ratio = mask_ratio

        self.decoder = nn.Sequential(
            nn.Linear(dim, 256), nn.BatchNorm1d(256), nn.Dropout(0.5), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.Dropout(0.2), nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        b, e = x.shape
        patch = int(e / self.m_book)
        res_list, ce_list = [], []

        curr_x = x
        for m in range(self.m_book):
            mask = (torch.rand(e, device=x.device) < self.mask_ratio)
            mask[:m * patch] = False
            mask[(m + 1) * patch - 1:] = False

            curr_x = torch.masked_fill(curr_x, mask, 0.0)
            curr_x = curr_x + self.pos.weight

            ze = self.encoders[m](curr_x)
            embedding = self.codebooks[m].weight
            distance = torch.sum(
                (embedding.reshape(1, embedding.shape[0], ze.shape[1]) - ze.reshape(ze.shape[0], 1, ze.shape[1])) ** 2,
                2)
            nearest_neighbor = torch.argmin(distance, 1)
            ce = self.codebooks[m](nearest_neighbor)
            ce_list.append(ze + (ce - ze).detach())
            res_list.append(ze)

        decoder_input = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        return self.decoder(decoder_input), res_list, ce_list

    def valid(self, x):
        curr_x = x + self.pos.weight.data
        res_list, ce_list = [], []
        for m in range(self.m_book):
            ze = self.encoders[m](curr_x)
            embedding = self.codebooks[m].weight.data
            distance = torch.sum(
                (embedding.reshape(1, embedding.shape[0], ze.shape[1]) - ze.reshape(ze.shape[0], 1, ze.shape[1])) ** 2,
                2)
            nearest_neighbor = torch.argmin(distance, 1)
            ce_list.append(self.codebooks[m](nearest_neighbor))
            res_list.append(ze)
        return self.decoder(torch.sum(torch.stack(ce_list, dim=0), dim=0)), res_list, ce_list

    def encode(self, x):
        curr_x = x + self.pos.weight.data
        nearest_neighbor_list = []
        for m in range(self.m_book):
            ze = self.encoders[m](curr_x)
            embedding = self.codebooks[m].weight.data
            distance = torch.sum(
                (embedding.reshape(1, embedding.shape[0], ze.shape[1]) - ze.reshape(ze.shape[0], 1, ze.shape[1])) ** 2,
                2)
            nearest_neighbor_list.append(torch.argmin(distance, 1))
        return torch.stack(nearest_neighbor_list, dim=0).transpose(0, 1)


class projection(nn.Module):
    def __init__(self, input_dim, output_dim, target_length, hidden_dim=256):
        super(projection, self).__init__()
        self.l1 = nn.Linear(int(target_length * input_dim), hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.l2(self.relu(self.dropout(self.l1(self.flatten(x)))))
