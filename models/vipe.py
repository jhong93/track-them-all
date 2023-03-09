import os
from scipy.spatial.distance import pdist
import numpy as np
import torch
import torch.nn as nn


from util.io import load_json


class FCNet(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3,
                 batch_norm=False):
        super().__init__()

        layers = [nn.Linear(
            input_dim,
            hidden_dims[0] if len(hidden_dims) > 0 else output_dim
        )]
        for i in range(len(hidden_dims)):
            layers.append(nn.ReLU(inplace=True))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.Linear(
                hidden_dims[i],
                hidden_dims[i + 1] if i + 1 < len(hidden_dims) else output_dim
            ))
            if i + 1 < len(hidden_dims):
                layers.append(nn.Dropout(dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FcResidualBlock(nn.Module):

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout))

    def forward(self, x):
        x2 = self.block(x)
        return x2 + x


class FCResNet(nn.Module):

    def __init__(self, in_dim, out_dim, num_blocks, hidden_dim, dropout=0.3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_blocks):
            layers.append(FcResidualBlock(hidden_dim, dropout))
        if out_dim is not None:
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _BaseModel:

    def __init__(self, encoder, decoders, device):
        self.encoder = encoder
        self.decoders = decoders
        self.device = device

        # Move to device
        self.encoder.to(device)
        for decoder in self.decoders.values():
            decoder.to(device)

    def _train(self):
        self.encoder.train()
        for decoder in self.decoders.values():
            decoder.train()

    def _eval(self):
        self.encoder.eval()
        for decoder in self.decoders.values():
            decoder.eval()


class KeypointEmbeddingModel(_BaseModel):

    def _predict(self, pose, get_emb, decoder_target=None):
        assert get_emb or decoder_target is not None, 'Nothing to predict'
        if not isinstance(pose, torch.Tensor):
            pose = torch.FloatTensor(pose)

        pose = pose.to(self.device)
        if len(pose.shape) == 2:
            pose = pose.unsqueeze(0)

        self.encoder.eval()
        if decoder_target is not None:
            decoder = self.decoders['3d']
            decoder.eval()
        else:
            decoder = None

        with torch.no_grad():
            n = pose.shape[0]
            emb = self.encoder(pose.view(n, -1))
            if decoder is None:
                return emb.cpu().numpy(), None

            pred3d = decoder(emb, decoder_target)
            if get_emb:
                return emb.cpu().numpy(), pred3d.cpu().numpy()
            else:
                return None, pred3d.cpu().numpy()

    def embed(self, pose):
        return self._predict(pose, get_emb=True)[0]


NUM_COCO_KEYPOINTS_ORIG = 17

# Ignore eyes and ears
NUM_COCO_KEYPOINTS = 13
COCO_POINTS_IDXS = [0] + list(range(5, 17))

COCO_FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                    [13, 14], [15, 16]]
COCO_FLIP_IDXS = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
COCO_TORSO_POINTS = [5, 6, 11, 12]


def preprocess_2d_keyp(kp, flip, to_tensor=True):
    kp = kp.copy()

    # Make the hips 0
    kp[:, :2] -= (kp[11, :2] + kp[12, :2]) / 2

    # Normalize distance from left-right sides
    max_torso_dist = np.max(pdist(kp[COCO_TORSO_POINTS, :2]))
    if max_torso_dist == 0:
        max_torso_dist = 1  # prevent 0div
    kp[:, :2] *= 0.5 / max_torso_dist

    if flip:
        kp = kp[COCO_FLIP_IDXS, :]
        kp[:, 0] *= -1

    # Shift confidences to -0.5, 0.5
    # NOTE: this is silly, but the model was trained this way...
    kp[:, 2] -= 0.5

    kp = kp[COCO_POINTS_IDXS, :]
    return torch.FloatTensor(kp) if to_tensor else kp


DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'vipe_data')


def load_embedding_model(model_dir=DEFAULT_MODEL_DIR, device='cuda'):
    print('Loading embedding model:', model_dir)
    model_param_file = os.path.join(model_dir, 'config.json')
    model_params = load_json(model_param_file)

    embedding_dim = model_params['embedding_dim']
    encoder_arch = model_params['encoder_arch']
    embed_bones = model_params['embed_bones']
    assert not embed_bones, \
        'James does not care about bone features right now...'

    print('Embedding dim:', embedding_dim)
    print('Encoder architecture:', encoder_arch)

    model_name = 'best_epoch'
    print('Model name:', model_name)

    encoder_path = os.path.join(
        model_dir, '{}.encoder.pt'.format(model_name))
    print('Encoder weights:', encoder_path)

    print('Device:', device)
    encoder = FCResNet(NUM_COCO_KEYPOINTS * 3, embedding_dim, *encoder_arch)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    return KeypointEmbeddingModel(encoder, {}, device)