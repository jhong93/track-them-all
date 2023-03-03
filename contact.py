# Adapted from: https://github.com/yul85/movingcam

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.body import Pose


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'pretrained_contact', '10000.pt')

# Number of frames of input and output
DEFAULT_INPUT_LEN = 9
DEFAULT_OUTPUT_LEN = 5

NUM_KEYPOINTS = 7


def normalize_2d_keyp(keyp_window):
    # Make the middle frame's mid-hip (0, 0)
    # Mash everything into -0.8 to 0.8 square
    N, M, D = keyp_window.shape
    assert N % 2 == 1
    assert M == NUM_KEYPOINTS
    assert D == 3

    center_xy = keyp_window[N // 2 + 1, 0, :2]
    keyp_window = keyp_window.copy().reshape(-1, 3)
    max_xy = np.amax(keyp_window[:, :2], axis=0)
    min_xy = np.amin(keyp_window[:, :2], axis=0)

    ratio = 1.25 * np.maximum(max_xy - center_xy, center_xy - min_xy)
    ratio = np.nan_to_num(ratio, nan=1)

    keyp_window[:, :2] -= center_xy[None, :]
    keyp_window[:, :2] /= ratio[None, :]
    return keyp_window.flatten()


class ContactNet(nn.Module):

    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.device = torch.device(device)
        self.noise = torch.distributions.Normal(
            torch.tensor(0.0), torch.tensor(0.005))

        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop_layer = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, output_size)

        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc5.weight, nonlinearity='relu')

        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()
        self.fc3.bias.data.zero_()
        self.fc4.bias.data.zero_()
        self.fc5.bias.data.zero_()

    def forward(self, x):
        if self.training:
            x = F.relu(self.bn1(self.fc1(
                x + self.noise.sample(x.size()).to(self.device))))
        else:
            x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop_layer(self.bn2(F.relu(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return torch.sigmoid(x)


def load_contact_model(device='cuda'):
    model = ContactNet(
        input_size=3 * NUM_KEYPOINTS * DEFAULT_INPUT_LEN,
        output_size=2 * DEFAULT_OUTPUT_LEN, device=device)
    model.load_state_dict(torch.load(DEFAULT_MODEL_PATH))
    model = model.to(device)
    model.eval()
    return model


def convert_coco_to_lower_openpose(keyp_2d):
    result = keyp_2d[:, [
        0,
        Pose.RHip,
        Pose.RKnee,
        Pose.RAnkle,
        Pose.LHip,
        Pose.LKnee,
        Pose.LAnkle
    ]]
    result[:, 0] = (keyp_2d[:, Pose.RHip] + keyp_2d[:, Pose.LHip]) / 2
    return result


HALF_IN_LEN = DEFAULT_INPUT_LEN // 2
HALF_OUT_LEN = DEFAULT_OUTPUT_LEN // 2


def infer_contact(model, coco_pose, hard_ensemble=False):
    T, N, D = coco_pose.shape
    pose = convert_coco_to_lower_openpose(coco_pose)

    # Results
    score = np.zeros((T, 2))
    support = np.zeros(T, dtype=np.int32)

    # Prepare inputs
    pose_in = []
    for i in range(T):
        # Pull out sequences of 9 frames (pad if start or end)
        inds = np.clip(np.arange(
            i - HALF_IN_LEN, i - HALF_IN_LEN + DEFAULT_INPUT_LEN
        ), 0, T - 1)
        pose_in.append(normalize_2d_keyp(pose[inds, :, :]))
    assert len(pose_in) == T

    # Inference
    pose_in = torch.from_numpy(np.array(pose_in)).float().to(model.device)
    with torch.no_grad():
        pred_contact = model(pose_in).cpu().view(T, -1, 2).numpy()

    # Post-processing
    for i in range(T):
        for j in range(
                i - HALF_OUT_LEN, i - HALF_OUT_LEN
                + DEFAULT_OUTPUT_LEN
        ):
            if j < 0 or j >= T:
                continue

            support[j] += 1
            if hard_ensemble:
                score[j, :] += (
                    pred_contact[i, j - i + HALF_OUT_LEN, :] > 0.5)
            else:
                score[j, :] += pred_contact[i, j - i + HALF_OUT_LEN, :]

    assert np.min(support) > 0
    score /= support[:, None]
    return score