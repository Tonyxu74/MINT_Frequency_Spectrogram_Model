from utils.model import Simp_Model
import torch
from tqdm import tqdm
from myargs import args
import time
import numpy as np
from utils.dataset import GenerateIterator


def train():

    # define model
    model = Simp_Model()

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

    lossfn = torch.nn.CrossEntropyLoss()

    iterator_train = GenerateIterator(args, './data/train', eval=False)

    '''
    NOTE!!!!!: VALIDATION FOLDER CURRENTLY HOLDS SAME DATA AS TRAIN FOLDER FOR SANITY CHECK
    
    If you would like to train/validate on specific experimentees, then simply copy their data folder into the
    train or validation folders. DO NOT MIX PEOPLE'S DATA BETWEEN TRAIN AND VALIDATION, we want to make sure that
    each individual that we test our models on have never been seen before
    '''

    iterator_val = GenerateIterator(args, './data/val', eval=True)

    # cuda?
    if torch.cuda.is_available():
        model = model.cuda()
        lossfn = lossfn.cuda()

    start_epoch = 1

    for epoch in range(start_epoch, args.num_epochs):

        # values to look at average loss per batch over epoch
        loss_sum, batch_num = 0, 0
        progress_bar = tqdm(iterator_train, disable=False)
        start = time.time()

        '''======== TRAIN ========'''
        for images, labels in progress_bar:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            prediction = model(images)

            loss = lossfn(prediction, labels).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            batch_num += 1

            progress_bar.set_description('Loss: {:.5f} '.format(loss_sum / (batch_num + 1e-6)))

        '''======== VALIDATION ========'''
        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()

                preds, gts = [], []

                progress_bar = tqdm(iterator_val)
                val_loss = 0

                for images, labels in progress_bar:
                    if torch.cuda.is_available():
                        images, labels = images.cuda(), labels.cuda()

                    prediction = model(images)

                    loss = lossfn(prediction, labels).mean()

                    prediction = torch.softmax(prediction, dim=1)
                    pred_class = torch.argmax(prediction, dim=1)

                    preds.append(pred_class.cpu().numpy())
                    gts.append(labels.cpu().numpy())

                    val_loss += loss.item()

                preds = np.asarray(preds)
                gts = np.asarray(gts)

                val_classification_score = (np.mean(preds == gts)).astype(np.float)

                print(
                    '|| Ep {} || Secs {:.1f} || Loss {:.1f} || Val score {:.3f} || Val Loss {:.3f} ||\n'.format(
                        epoch,
                        time.time() - start,
                        loss_sum,
                        val_classification_score,
                        val_loss,
                    ))

            model.train()

        # save models every 1 epoch
        if epoch % 1 == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, './data/model/{}_{}.pt'.format(args.model_name, epoch))


if __name__ == '__main__':
    train()
