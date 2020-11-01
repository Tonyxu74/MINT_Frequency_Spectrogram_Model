from utils.model import Simp_Model, MultiLayerModel
import torch
from tqdm import tqdm
from myargs import args
import time
import numpy as np
from utils.dataset import GenerateIterator
from sklearn.metrics import confusion_matrix


def train():

    # define model
    size_dict = {1: 7850, 2: 4900, 3: 1300, 4: 100}
    model = MultiLayerModel(args.patch_classes, size_dict)

    # check if continue training from previous epochs
    if args.continue_train:
        pretrained_dict = torch.load('./data/model/{}_{}.pt'.format(args.model_name, args.start_epoch))['state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # define optimizer, loss function, and iterators
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )
    class_weights = torch.tensor([0.35, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    lossfn = torch.nn.CrossEntropyLoss(weight=class_weights)

    iterator_train = GenerateIterator(args, './data/train', eval=False)
    iterator_val = GenerateIterator(args, './data/val', eval=True)

    # cuda?
    if torch.cuda.is_available():
        model = model.cuda()
        lossfn = lossfn.cuda()

    start_epoch = 1

    for epoch in range(start_epoch, args.num_epochs):

        # use annealing standard dev, max out on epoch 30, don't start until epoch 50
        if epoch > 50:
            noise_std = 0.03 * (epoch - 50) / 30
            iterator_train.dataset.std = noise_std if noise_std <= 0.03 else 0.03

        # values to look at average loss per batch over epoch
        loss_sum, batch_num = 0, 0
        progress_bar = tqdm(iterator_train, disable=False)
        start = time.time()

        t_preds, t_gts = [], []

        '''======== TRAIN ========'''
        for images, labels in progress_bar:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            prediction = model(images)

            loss = lossfn(prediction, labels).mean()

            pred_class = torch.argmax(prediction.detach(), dim=1)

            t_preds.extend(pred_class.cpu().numpy().tolist())
            t_gts.extend(labels.cpu().numpy().tolist())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            batch_num += 1

            progress_bar.set_description('Loss: {:.5f} '.format(loss_sum / (batch_num + 1e-6)))

        t_preds = np.asarray(t_preds)
        t_gts = np.asarray(t_gts)

        train_classification_score = (np.mean(t_preds == t_gts)).astype(np.float)
        print(confusion_matrix(t_gts, t_preds))

        '''======== VALIDATION ========'''
        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()

                preds, gts = [], []

                progress_bar = tqdm(iterator_val)
                val_loss, val_batch_num = 0, 0

                for images, labels in progress_bar:
                    if torch.cuda.is_available():
                        images, labels = images.cuda(), labels.cuda()

                    prediction = model(images)

                    loss = lossfn(prediction, labels).mean()

                    prediction = torch.softmax(prediction.detach(), dim=1)
                    pred_class = torch.argmax(prediction, dim=1)

                    preds.extend(pred_class.cpu().numpy().tolist())
                    gts.extend(labels.cpu().numpy().tolist())

                    val_loss += loss.item()
                    val_batch_num += 1

                preds = np.asarray(preds)
                gts = np.asarray(gts)

                val_classification_score = (np.mean(preds == gts)).astype(np.float)
                print(confusion_matrix(gts, preds))

                print(
                    '|| Ep {} || Secs {:.1f} || Train Score {:.3f} || Train Loss {:.5f} || Val score {:.3f} || Val Loss {:.5f} ||\n'.format(
                        epoch,
                        time.time() - start,
                        train_classification_score,
                        loss_sum / batch_num,
                        val_classification_score,
                        val_loss / val_batch_num,
                    ))

            model.train()

        # save models every 1 epoch
        if epoch % 50 == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, './data/model/{}_{}.pt'.format(args.model_name, epoch))


if __name__ == '__main__':
    train()
