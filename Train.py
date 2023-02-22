from torchnet import meter
import torch
import os
from torch.utils.data import DataLoader
from models import DeepVANetBio, DeepVANetVision, DeepVANet
from dataset import DEAP, MAHNOB, DEAPAll, MAHNOBAll
from utils import out_put



def train(modal, dataset, task, epoch, lr, batch_size, face_feature_size=16, bio_feature_size=64, use_gpu=False, save_model=True, mse_weight=0.01):
    '''
    Train and test the model. Output the results.
    :param modal: data modality
    :param dataset: used dataset
    :param task: the type of training task
    :param epoch: the number of epoches
    :param lr: learn rate
    :param batch_size: training batach size
    :param face_feature_size: face feature size
    :param bio_feature_size: bio-sensing feature size
    :param use_gpu: use gpu or not
    :return: the best test accuracy
    '''
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    file_name = f'./drive/MyDrive/DeepVADNet/results/{dataset}/{modal}/{dataset}_{modal}_{task}_{face_feature_size}_{bio_feature_size}_{mse_weight}'
    # directory = file_name.split('/')[-2]
    if not os.path.exists(f'./drive/MyDrive/DeepVADNet/results/{dataset}/{modal}'):
        os.mkdir(f'./drive/MyDrive/DeepVADNet/results/{dataset}/{modal}')

    if dataset == 'DEAP':
        train_data = DEAP(modal=modal, kind='train')
        val_data = DEAP(modal=modal, kind='val')
        test_data = DEAP(modal=modal, kind='test')
        bio_input_size = 40
        peri_input_size = 8
    if dataset == 'MAHNOB':
        train_data = MAHNOB(modal=modal, kind='train')
        val_data = MAHNOB(modal=modal, kind='val')
        test_data = MAHNOB(modal=modal, kind='test')
        bio_input_size = 38
        peri_input_size = 6

    # model
    if modal == 'face':
        model = DeepVANetVision(feature_size=face_feature_size,task=task).to(device)
    if modal == 'bio':
        model = DeepVANetBio(input_size=bio_input_size, feature_size=bio_feature_size,task=task).to(device)
    if modal == 'eeg':
        model = DeepVANetBio(input_size=32, feature_size=bio_feature_size,task=task).to(device)
    if modal == 'peri':
        model = DeepVANetBio(input_size=peri_input_size, feature_size=bio_feature_size, task=task).to(device)
    if modal == 'face_eeg':
        model = DeepVANet(bio_input_size=32, face_feature_size=face_feature_size, bio_feature_size=bio_feature_size, task=task).to(device)
    if modal == 'face_peri':
        model = DeepVANet(bio_input_size=peri_input_size, face_feature_size=face_feature_size, bio_feature_size=bio_feature_size, task=task).to(device)
    if modal == 'face_bio':
        model = DeepVANet(bio_input_size=bio_input_size, face_feature_size=face_feature_size, bio_feature_size=bio_feature_size, task=task).to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # criterion and optimizer
    if task == 'VADClassification' or task == 'VADRegression':
        criterion = torch.nn.MSELoss()
    elif task == 'emotionClassification':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion_ce = torch.nn.CrossEntropyLoss()
        criterion_mse = torch.nn.MSELoss()
    # criterion = torch.nn.BCELoss()
    lr = lr
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    # meters
    loss_meter = meter.AverageValueMeter()

    best_accuracy = 0
    best_epoch = 0

    # train
    for epoch in range(epoch):
        pred_label_valence = []
        pred_label_arousal = []
        true_label_valence = []
        true_label_arousal = []
        pred_label_control = []
        true_label_control = []

        pred_valence = []
        pred_arousal = []
        true_valence = []
        true_arousal = []
        pred_control = []
        true_control = []
        pred_emotion = []
        true_emotion = []

        loss_meter.reset()

        for ii, (data,label, emotion) in enumerate(train_loader):
            # print(ii)
            # train model
            if modal == 'face_eeg' or modal == 'face_peri' or modal == 'face_bio':
                input = (data[0].float().to(device), data[1].float().to(device))
            else:
                input = data.float().to(device)
            label = label.float().to(device)
            emotion = emotion.float().to(device)

            optimizer.zero_grad()
            pred = model(input)
            if task == 'VADRegression' or task=='VADClassification' or task == 'emotionClassification':
                pred = pred.float()
            else:
                pred = (pred[0].float(),pred[1].float())
            # print(pred.shape,label.shape)
            if task == 'VADRegression' or task == 'VADClassification':
                loss = criterion(pred, label)
            elif task == 'emotionClassification':
                loss = criterion(pred, emotion.long())
                #loss = mse_weight*criterion_mse(pred[0],label) + criterion_ce(pred[1],emotion.long())
            else:
                loss = mse_weight*criterion_mse(pred[0],label) + criterion_ce(pred[1],emotion.long())
                # loss = criterion_ce(pred[1],emotion.long())

            loss.backward()
            optimizer.step()

            # meters update
            loss_meter.add(loss.item())

            if task == 'VADRegression' or task == 'VADClassification':
                pred_valence.append(pred.data[:,0])
                pred_arousal.append(pred.data[:,1])
                pred_control.append(pred.data[:,2])

                true_valence.append(label[:,0])
                true_arousal.append(label[:,1])
                true_control.append(label[:,2])
            elif task == 'emotionClassification':
                pred_emotion.append(torch.argmax(pred,-1))
                true_emotion.append(emotion)
            else:
                pred_valence.append(pred[0].data[:, 0])
                pred_arousal.append(pred[0].data[:, 1])
                pred_control.append(pred[0].data[:, 2])

                true_valence.append(label[:, 0])
                true_arousal.append(label[:, 1])
                true_control.append(label[:, 2])
                pred_emotion.append(torch.argmax(pred[1],-1))
                true_emotion.append(emotion)

        if task != 'emotionClassification':
            pred_arousal = torch.cat(pred_arousal, 0)
            pred_valence = torch.cat(pred_valence, 0)
            pred_control = torch.cat(pred_control, 0)
            true_valence = torch.cat(true_valence, 0)
            true_arousal = torch.cat(true_arousal, 0)
            true_control = torch.cat(true_control, 0)
            pred_label_valence = (pred_valence >= 5).float().to(device).data
            pred_label_arousal = (pred_arousal >= 5).float().to(device).data
            pred_label_control = (pred_control >= 5).float().to(device).data
            true_label_valence = (true_valence >= 5).float().to(device).data
            true_label_arousal = (true_arousal >= 5).float().to(device).data
            true_label_control = (true_control >= 5).float().to(device).data

        if task=='emotionClassification' or task =='emotionClassification_VAD':
            true_emotion = torch.cat(true_emotion,0)
            pred_emotion = torch.cat(pred_emotion,0)
            train_accuracy_emotion = torch.sum((pred_emotion==true_emotion).float()) / true_emotion.size(0)

        if task != 'emotionClassification':
            train_accuracy_arousal = torch.sum((pred_label_arousal == true_label_arousal).float()) / true_label_arousal.size(0)
            train_accuracy_valence = torch.sum((pred_label_valence == true_label_valence).float()) / true_label_valence.size(0)
            train_accuracy_control = torch.sum((pred_label_control == true_label_control).float()) / true_label_control.size(0)
            train_mse_valence = torch.sum((pred_valence - true_valence)*(pred_valence - true_valence)) / true_valence.size(0)
            train_mse_arousal = torch.sum((pred_arousal - true_arousal)*(pred_arousal - true_arousal)) / true_arousal.size(0)
            train_mse_control = torch.sum((pred_control - true_control)*(pred_control - true_control)) / true_control.size(0)


            out_put('Epoch: ' + 'train' + str(epoch) + '| train accuracy (valence): ' + str(train_accuracy_valence.item()) + '| train mse (valence): ' + str(train_mse_valence.item()),
                    file_name)
            out_put('Epoch: ' + 'train' + str(epoch) + '| train accuracy (arousal): ' + str(train_accuracy_arousal.item()) + '| train mse (arousal): ' + str(train_mse_arousal.item()),
                    file_name)
            out_put('Epoch: ' + 'train' + str(epoch) + '| train accuracy (control): ' + str(train_accuracy_control.item()) + '| train mse (valence): ' + str(train_mse_control.item()),
                    file_name)

        if task == 'emotionClassification' or task == 'emotionClassification_VAD':
            out_put('Epoch: ' + 'train' + str(epoch) + '| train accuracy (emotion): ' + str(
                train_accuracy_emotion.item()) ,file_name)

        if task != 'emotionClassification':
            val_accuracy_valence, val_accuracy_arousal, val_accuracy_control, val_mse_valence, val_mse_arousal, val_mse_control, val_accuracy_emotion = val(modal, model, val_loader, use_gpu, task)
            test_accuracy_valence, test_accuracy_arousal, test_accuracy_control, test_mse_valence, test_mse_arousal, test_mse_control, test_accuracy_emotion = val(modal, model, test_loader, use_gpu, task)

        else:
            val_accuracy_emotion = val(modal, model, val_loader, use_gpu, task)
            test_accuracy_emotion = val(modal, model, test_loader, use_gpu, task)

        if task != 'emotionClassification':
            out_put('Epoch: ' + 'train' + str(epoch) + '| train loss: ' + str(loss_meter.value()[0]) +
                    '| val accuracy (valence): ' + str(val_accuracy_valence.item()) + '| val accuracy (arousal): ' +
                    str(val_accuracy_arousal.item()) + '| val accuracy (control): ' +
                    str(val_accuracy_control.item()) + '| val mse (valence): ' + str(val_mse_valence.item())
                    + '| val mse (arousal): ' + str(val_mse_arousal.item())
                    + '| val mse (control): ' + str(val_mse_control.item()) + '| test accuracy (valence): ' + str(test_accuracy_valence.item()) + '| test accuracy (arousal): ' +
                    str(test_accuracy_arousal.item()) + '| test accuracy (control): ' +
                    str(test_accuracy_control.item()) + '| test mse (valence): ' + str(test_mse_valence.item())
                    + '| test mse (arousal): ' + str(test_mse_arousal.item())
                    + '| test mse (control): ' + str(test_mse_control.item()), file_name)
        if task == 'emotionClassification' or task == 'emotionClassification_VAD':
            out_put('Epoch: ' + 'train' + str(epoch) + '| train loss: ' + str(loss_meter.value()[0]), file_name)
            out_put('Epoch: ' + 'train' + str(epoch) + '| val accuracy (emotion): ' + str(
                val_accuracy_emotion.item()) + '| test accuracy (emotion): ' + str(
                test_accuracy_emotion.item()), file_name)

        val_accuracy = (val_accuracy_valence + val_accuracy_arousal + val_accuracy_control)/3 if task=='VADRegression' else val_accuracy_emotion
        test_accuracy = (test_accuracy_valence + test_accuracy_arousal + test_accuracy_control)/3 if task=='VADRegression' else test_accuracy_emotion

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            test_accuracy_final = test_accuracy
            if save_model:
                model.save(f'./drive/MyDrive/DeepVADNet/models/{dataset}/{modal}/{dataset}_{modal}_{task}_{mse_weight}_best.pth')

    if save_model:
        model.save(f'./drive/MyDrive/DeepVADNet/models/{dataset}/{modal}/{dataset}_{modal}_{task}_{mse_weight}.pth')

    perf = f"best val accuracy is {best_accuracy} in epoch {best_epoch}" + "\n" + f'test accuracy is {test_accuracy_final}'
    out_put(perf,file_name)

    return best_accuracy
    return perf
    #return out_put

@torch.no_grad() 
def val(modal, model, dataloader, use_gpu, task):
    model.eval()
    if use_gpu:
      device = torch.device('cuda')
    else:
      device = torch.device('cpu')

    pred_valence = []
    pred_arousal = []
    true_valence = []
    true_arousal = []
    pred_control = []
    true_control = []
    pred_emotion = []
    true_emotion = []

    for ii, (data, label, emotion) in enumerate(dataloader):
        if modal == 'face_eeg' or modal == 'face_peri' or modal == 'face_bio':
            input = (data[0].float().to(device), data[1].float().to(device))
        else:
            input = data.float().to(device)
        label = label.to(device)
        pred = model(input)
        if task == 'VADRegression' or task == 'VADClassification' or task == 'emotionClassification':
            pred = pred.float()
        else:
            pred = (pred[0].float(), pred[1].float())
        emotion = emotion.float().to(device)

        if task == 'VADRegression' or task == 'VADClassification':
            pred_valence.append(pred.data[:, 0])
            pred_arousal.append(pred.data[:, 1])
            pred_control.append(pred.data[:, 2])

            true_valence.append(label[:, 0])
            true_arousal.append(label[:, 1])
            true_control.append(label[:, 2])
        elif task == 'emotionClassification':
            pred_emotion.append(torch.argmax(pred, -1))
            true_emotion.append(emotion)
        else:
            pred_valence.append(pred[0].data[:, 0])
            pred_arousal.append(pred[0].data[:, 1])
            pred_control.append(pred[0].data[:, 2])

            true_valence.append(label[:, 0])
            true_arousal.append(label[:, 1])
            true_control.append(label[:, 2])
            pred_emotion.append(torch.argmax(pred[1], -1))
            true_emotion.append(emotion)
    if task != 'emotionClassification':
        pred_arousal = torch.cat(pred_arousal, 0)
        pred_valence = torch.cat(pred_valence, 0)
        pred_control = torch.cat(pred_control, 0)
        true_valence = torch.cat(true_valence, 0)
        true_arousal = torch.cat(true_arousal, 0)
        true_control = torch.cat(true_control, 0)
        pred_label_valence = (pred_valence >= 5).float().to(device).data
        pred_label_arousal = (pred_arousal >= 5).float().to(device).data
        pred_label_control = (pred_control >= 5).float().to(device).data
        true_label_valence = (true_valence >= 5).float().to(device).data
        true_label_arousal = (true_arousal >= 5).float().to(device).data
        true_label_control = (true_control >= 5).float().to(device).data

    if task == 'emotionClassification' or task == 'emotionClassification_VAD':
        true_emotion = torch.cat(true_emotion, 0)
        pred_emotion = torch.cat(pred_emotion, 0)
        val_accuracy_emotion = torch.sum((pred_emotion == true_emotion).float()) / true_emotion.size(0)
    else:
        val_accuracy_emotion = None

    if task != 'emotionClassification':
        val_accuracy_arousal = torch.sum((pred_label_arousal == true_label_arousal).float()) / true_label_arousal.size(0)
        val_accuracy_valence = torch.sum((pred_label_valence == true_label_valence).float()) / true_label_valence.size(0)
        val_accuracy_control = torch.sum((pred_label_control == true_label_control).float()) / true_label_control.size(0)
        val_mse_valence = torch.sum((pred_valence - true_valence) * (pred_valence - true_valence)) / true_valence.size(0)
        val_mse_arousal = torch.sum((pred_arousal - true_arousal) * (pred_arousal - true_arousal)) / true_arousal.size(0)
        val_mse_control = torch.sum((pred_control - true_control) * (pred_control - true_control)) / true_control.size(0)

    model.train()
    if task != 'emotionClassification':
        return val_accuracy_valence, val_accuracy_arousal, val_accuracy_control, val_mse_valence, val_mse_arousal, val_mse_control , val_accuracy_emotion
    else:
        return val_accuracy_emotion
