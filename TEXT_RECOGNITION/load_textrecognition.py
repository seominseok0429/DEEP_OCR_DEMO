import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from PIL import Image

import torchvision.transforms as transforms

from TEXT_RECOGNITION.utils import CTCLabelConverter, AttnLabelConverter
from TEXT_RECOGNITION.dataset import RawDataset, AlignCollate
from TEXT_RECOGNITION.model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class text_recognition(object):
    def __init__(self, use_cpu=False):
        self.converter = AttnLabelConverter('0123456789abcdefghijklmnopqrstuvwxyz')
        self.model = Model()
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.load_state_dict(torch.load("./PRETRAINED_WEIGHT/TPS-ResNet-BiLSTM-Attn.pth"))
        self.model.eval()
        self.demo_data = RawDataset()  # use RawDataset

    def __call__(self, path, bbox):
        img,_ = self.demo_data[path,bbox]
        batch_size = img.shape[0]
        img = img.cuda()
        length_for_pred = torch.IntTensor([25] * batch_size).cuda()
        text_for_pred = torch.LongTensor(batch_size, 25 + 1).fill_(0).cuda()

        preds = self.model(img, text_for_pred, is_train=False)

        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        results = []
        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]
            results.append(pred)
        return results

#b = text_recognition()

#a = b('/workspace/HANULSOFT/TEXT_RECOGNITION/demo_image/demo_1.png', [[0,0,184,72],[0,0,184,72]])
#print(a)

