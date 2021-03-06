<<<<<<< HEAD
from utils.model import simple_dnn
import utils.visualization as visualization
from torch import nn
=======
from utils.model import MultiLayerModel
>>>>>>> Tony
import torch
from tqdm import tqdm
from myargs import args
import time
import numpy as np
<<<<<<< HEAD
import utils.dataset as dataset
=======
from utils.dataset import GenerateIterator
from sklearn.metrics import confusion_matrix
>>>>>>> Tony


def train():

    # define model
<<<<<<< HEAD
    model = simple_dnn()

    #for tensorboard
    writer = visualization.writer

    # check if continue training from previous epochs
    #if args.continueTrain:
    #    pretrained_dict = torch.load('PATH HERE'.format(args.start_epoch))['state_dict']
    #    model_dict = model.state_dict()
=======
    size_dict = {1: 7850, 2: 4900, 3: 1300, 4: 100}
    model = MultiLayerModel(args.patch_classes, size_dict)

    if args.pretrain:
        pretrained_dict = torch.load(args.pretrained_path)['state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # check if continue training from previous epochs
    if args.continue_train:
        pretrained_dict = torch.load('./data/model/{}_{}.pt'.format(args.model_name, args.start_epoch))['state_dict']
        model_dict = model.state_dict()
>>>>>>> Tony
        # 1. filter out unnecessary keys
    #    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
    #    model_dict.update(pretrained_dict)
    #    model.load_state_dict(model_dict)

    # define optimizer, loss function, and iterators
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )
    class_weights = torch.tensor([0.4, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    lossfn = torch.nn.CrossEntropyLoss(weight=class_weights)

<<<<<<< HEAD
    lossfn = torch.nn.CrossEntropyLoss()

    iterator_train = dataset.GenerateIterator(args, '/data/train/trainfiles')
    iterator_val = dataset.GenerateIterator(args, '/data/train/valfiles')

    # sending model structure to tensorboard
    images, labels = next(iter(iterator_train))
    images=images.float().flatten()
    writer.add_graph(model, images)
    writer.close()
=======
    iterator_train = GenerateIterator(args, './data/train', eval=False, input_mode='train')
    iterator_val = GenerateIterator(args, './data/val', eval=True, input_mode='train')
>>>>>>> Tony

    # cuda?
    if torch.cuda.is_available():
        model = model.cuda()
        lossfn = lossfn.cuda()

    start_epoch = 1

<<<<<<< HEAD
    for epoch in range(start_epoch, args.num_epoch):
=======
    for epoch in range(start_epoch, args.num_epochs):

        # use annealing standard dev, max out on epoch 30, don't start until epoch 50
        # noise_std = 1 * (50 - epoch) / 50
        # iterator_train.dataset.std = noise_std if noise_std > 0 else 0
        iterator_train.dataset.std = 0.2
>>>>>>> Tony

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

            images = images.float()
            labels = labels.long()

            images = images.flatten()

            prediction = model(images)

            #print(prediction)
            #print(labels)

            loss = lossfn(prediction, labels)#.mean()

            pred_class = torch.argmax(prediction.detach(), dim=1)

            t_preds.extend(pred_class.cpu().numpy().tolist())
            t_gts.extend(labels.cpu().numpy().tolist())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            batch_num += 1

            progress_bar.set_description('Loss: {:.5f} '.format(loss_sum / (batch_num + 1e-6)))

<<<<<<< HEAD
        writer.add_scalar('train/loss', loss_sum, epoch)

=======
        t_preds = np.asarray(t_preds)
        t_gts = np.asarray(t_gts)

        train_classification_score = (np.mean(t_preds == t_gts)).astype(np.float)
        print(confusion_matrix(t_gts, t_preds))
>>>>>>> Tony

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

                    images = images.float()
                    labels = labels.long()

                    images = images.flatten()

                    prediction = model(images)

                    loss = lossfn(prediction, labels)

                    prediction = torch.softmax(prediction.detach(), dim=1)
                    pred_class = torch.argmax(prediction, dim=1)

<<<<<<< HEAD

                    #print(prediction)
                    #print('pc before ' + str(int(pred_class)))
                    #print('label before '+ str(int(labels)))

                    # if we need to simplify classification by considering all movmenet as 1 signal
                    #if (int(pred_class) != 0):
                    #    pred_class = 1.0
                    #if int(labels) != 0:
                    #    labels = 1.0

                    #print(pred_class)
                    #print(labels)

                    #preds.append(pred_class.cpu().numpy())
                    preds.append(int(pred_class))
                    #gts.append(labels.cpu().numpy())
                    gts.append(int(labels))
=======
                    preds.extend(pred_class.cpu().numpy().tolist())
                    gts.extend(labels.cpu().numpy().tolist())
>>>>>>> Tony

                    val_loss += loss.item()
                    val_batch_num += 1

                preds = np.asarray(preds)
                gts = np.asarray(gts)

<<<<<<< HEAD
                #val_classification_score = (np.mean(preds == gts)).astype(np.float)
                val_classification_score = (preds == gts).sum()/len(preds) # raw accuracy
=======
                val_classification_score = (np.mean(preds == gts)).astype(np.float)
                print(confusion_matrix(gts, preds))
>>>>>>> Tony

                print(
                    '|| Ep {} || Secs {:.1f} || Train Score {:.3f} || Train Loss {:.5f} || Val score {:.3f} || Val Loss {:.5f} ||\n'.format(
                        epoch,
                        time.time() - start,
                        train_classification_score,
                        loss_sum / batch_num,
                        val_classification_score,
                        val_loss / val_batch_num,
                    ))

                writer.add_scalar('test/loss', val_loss, epoch)
                writer.add_scalar('test/accuracy', val_classification_score, epoch)

                #model.train()

        #save models every 10 epoch
        if epoch % 25 == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
<<<<<<< HEAD
            #torch.save(state, '/trained_models/dnn_epoch{1}'.format(str(args.model_name), epoch))
            torch.save(state, 'trained_models/dnn_epoch_' + str(epoch) )
            print('saved model!')
=======
            torch.save(state, './data/model/{}_{}.pt'.format(args.model_name, epoch))
>>>>>>> Tony


if __name__ == '__main__':
    # train()
    model = MultiLayerModel(8, {1: 7850, 2: 4900, 3: 1300, 4: 100})
    pretrained_dict = torch.load('./data/model/multilayer_conv_178.pt')['state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    example = torch.rand(1, 1, 8, 500)
    model.eval()
    traced_model = torch.jit.trace(model, example)
    traced_model.save('traced_multilayer_conv.pt')
