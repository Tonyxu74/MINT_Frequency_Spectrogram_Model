from utils.model import Freq_Model
import torch
from tqdm import tqdm
from myargs import args
import numpy as np
from utils.dataset_online import GenerateIterator
import matplotlib.pyplot as plt
import random


def visualize(datalist, title, dims=(2, 2)):
    random.shuffle(datalist)

    fig, axs = plt.subplots(*dims)
    plt.suptitle(title)

    for i in range(dims[0]):
        for j in range(dims[1]):
            a = axs[i, j].imshow(datalist[i], cmap='hot')
            fig.colorbar(a, ax=axs[i, j])

    for ax in axs.flat:
        ax.set(xlabel='frequency (Hz)', ylabel='channels')
    for ax in axs.flat:
        ax.label_outer()

    plt.show()


def evaluate(modelpath):

    # define model
    model = Freq_Model()

    # load weights
    pretrained_dict = torch.load(modelpath)['state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    iterator_eval = GenerateIterator('./data/preproc_online_val/', eval=True, spectro_in=True)

    # cuda?
    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        model.eval()

        preds, gts = [], []

        # get true/false left/right classifications
        tleft, tright, fleft, fright = [], [], [], []

        progress_bar = tqdm(iterator_eval)

        for images, labels in progress_bar:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            prediction = model(images)

            prediction = torch.softmax(prediction.detach(), dim=1)
            pred_class = torch.argmax(prediction, dim=1)

            for data, pred, label in zip(images, pred_class, labels):
                data = data.view(args.patch_classes, args.frequency_size).cpu().numpy()

                # true left prediction
                if pred == 0 and label == 0:
                    tleft.append(data)

                # true right prediction
                if pred == 1 and label == 1:
                    tright.append(data)

                # false left prediction
                if pred == 0 and label == 1:
                    fleft.append(data)

                # false right prediction
                if pred == 1 and label == 0:
                    fright.append(data)

            preds.extend(pred_class.cpu().numpy().tolist())
            gts.extend(labels.cpu().numpy().tolist())

        preds = np.asarray(preds)
        gts = np.asarray(gts)

        eval_classification_score = (np.mean(preds == gts)).astype(float)

    print(f'eval score: {eval_classification_score}')
    visualize(tleft, 'true left')
    visualize(tright, 'true right')


if __name__ == '__main__':
    evaluate('./data/model/spectro_mlp_20.pt')
