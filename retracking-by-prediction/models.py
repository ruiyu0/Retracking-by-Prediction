import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim),
            torch.zeros(self.num_layers, batch, self.h_dim)
        )

    def forward(self, obs_traj):
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h


class Decoder(nn.Module):
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=64, num_layers=1,
        dropout=0.0, activation='relu', batch_norm=True
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim

        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple):
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos
            embedding_input = rel_pos
            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos
        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)

        return pred_traj_fake_rel, state_tuple[0]


class VanillaLSTMNet(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=64, num_layers=1, dropout=0.0,
        activation='relu', batch_norm=True
    ):
        super(VanillaLSTMNet, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm
        )

    def forward(self, obs_traj, obs_traj_rel, user_noise=None):
        batch = obs_traj_rel.size(1)

        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        mlp_decoder_context_input = final_encoder_h.view(-1, self.encoder_h_dim)
        decoder_h = torch.unsqueeze(mlp_decoder_context_input, 0)
        decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim)
        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]

        # Predict Trajectory
        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel
