import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix
#import random
import numpy as np

from timing_decor import timing_decorator
from gradCAM import gradCAM

def gradCAMS_saver(val_loader, model, encoded_labels, get_all=False):
    # running gradCAM specific functions
    cams = {}
    samples = {}
    model.eval()
    for i, data in enumerate(val_loader):
        inputs, y_val_temp = data
        #print(inputs.shape, y_val_temp.shape)
        for j in range(inputs.shape[0]):
            target_layer = model.conv_block2[-1]
            grad_cam = gradCAM(model, target_layer)
            single_input = inputs[j].unsqueeze(0)
            cam_hm, pred_class = grad_cam.generate_cam(single_input, target_class=None)
            predicted_label = list(encoded_labels.keys())[list(encoded_labels.values()).index(pred_class)]
            #print(f"Predicted class: {predicted_label}")
            if predicted_label not in cams.keys():
                cams[predicted_label] = [cam_hm]
            else:
                cams[predicted_label].append(cam_hm)

            if predicted_label not in samples.keys():
                samples[predicted_label] = inputs[j]

    class_cams = {}
    for key in cams.keys():
        mean_class_cam = np.mean(cams[key], axis=0)
        #print(mean_class_cam.shape)
        class_cams[key] = mean_class_cam

    if get_all:
        return class_cams, cams, samples
    else:
        return class_cams, samples

@timing_decorator
def run_training(model, train_loader, val_loader, encoded_labels, rate_l, NUM_EPOCHS=800,  save=True, thresh=75):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=rate_l)
    losses_epoch_mean = []
    for epoch in range(NUM_EPOCHS):
        losses_epoch = []
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # Check for NaN loss
            if torch.isnan(inputs).any():
                print(f"NaN input at epoch {epoch}, batch {i}")
                i_err = inputs
                break
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch}, batch {i}")
                l_err = loss
                break
            
            loss.backward()
            optimizer.step()
            losses_epoch.append(loss.item())
        
        losses_epoch_mean.append(np.mean(losses_epoch))
        if epoch % (int(NUM_EPOCHS/10)) == 0:
            print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {np.mean(losses_epoch):.16f}')

    #sns.lineplot(x=list(range(len(losses_epoch_mean))), y=losses_epoch_mean)
    y_val = []
    y_val_hat = []
    model.eval()
    for i, data in enumerate(val_loader):
        inputs, y_val_temp = data
        with torch.no_grad():
            y_val_hat_temp = model(inputs).round()
    
        y_val.extend(y_val_temp.numpy())
        y_val_hat.extend(y_val_hat_temp.numpy())
    
    # Accuracy
    acc = accuracy_score(y_val, np.argmax(y_val_hat, axis=1))

    print(f'Accuracy lr={rate_l}: {acc*100:.2f} %')
    # confusion matrix
    cm = confusion_matrix(y_val, np.argmax(y_val_hat, axis=1))

    # running gradCAM specific functions
    gradCAM_out = gradCAMS_saver(val_loader, model, encoded_labels, get_all=True)

    data = {'mean_loss': losses_epoch_mean, 'acc':acc, 'cm': cm, 'lr': rate_l, 'model': model.state_dict(), 'gradCAM_out': gradCAM_out}

    if save and acc*100>thresh:
        print('saving model and data')
        torch.save(model.state_dict(), f'gradCAM_model_a{acc*100:.1f}_LR_{rate_l}_full.pth')
        #data = {'mean_loss': losses_epoch_mean, 'acc':acc, 'cm': cm, 'model': model.state_dict(), 'gradCAM_out': gradCAM_out}
        np.save(f'results_and_model_acc_{acc*100:.1f}_nclasses_{model.n_classes}', data)

    return data