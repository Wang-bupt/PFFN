import torch

from torch import nn
import numpy as np

from torchvision import transforms
import torchvision

from models import *
from bert import BertEmbedding

from extract_emotion_ch import *

bert = BertEmbedding()
resnet = torchvision.models.resnet50(pretrained = True)

classify_path = './classify/classify_best_model'
classify_model = torch.load(classify_path)

imgs_path = './data_samples/imgs_set.pickle'
texts_path = './data_samples/texts_set.pickle'
labels_path = './data_samples/labels_set.pickle'

texts = torch.load(texts_path)
labels = torch.load(labels_path)
imgs_dataset = torch.load(imgs_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

classify_model = classify_model.to(device)

def get_sentiment(original_text):
    content = original_text
    content_words = cut_words_from_text(content)
    sentiment = extract_publisher_emotion(content, content_words)
    sentiment = torch.tensor(sentiment)
    return sentiment

print(len(texts))

split = int(0.75*len(texts))

class textcnn(nn.Module):
    def __init__(self):
        super(textcnn,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=200, kernel_size=(4, 768))
        self.pool1 = nn.MaxPool2d(kernel_size=4)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.squeeze(3)
        x = self.pool1(x)
        x = self.flatten(x)
        return x

class imgsExpertsNet(nn.Module):
    def __init__(self):
        super(imgsExpertsNet, self).__init__()
        
        expert = []
        for i in range(9):
            expert.append(FullyConnectedOutput(1009,1000,1000))
            
        self.expert = nn.ModuleList(expert)
        
    def forward(self, img_f,expert_ratio):
        img_f = img_f.unsqueeze(0)
        img_f = torch.concat((img_f,expert_ratio),3)
        for i in range(9):
            
            tmp_feature = self.expert[i](img_f)
            shared_feature = torch.zeros(1,1000).to(device)
            t = tmp_feature * expert_ratio[0][0][0][i]
            t = t.squeeze(0).squeeze(0)
            shared_feature += (t)
            
        return shared_feature

class PFFN(nn.Module):
    def __init__(self):
        super(PFFN, self).__init__()
        
        self.attn = myAttention(47,4000,4000,2000)
        self.experts = imgsExpertsNet()
        self.tc = textcnn()

        self.l1 = nn.Linear(2450,4000)
        self.relu = nn.ReLU()
        self.attnf = attentionFusion(1000,2000,1000,1000)      

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.04)
                nn.init.constant_(m.bias, 0)

    def forward(self,img_f,class_f,text_f,sentiment_f):
        experts_result =  self.experts(img_f.squeeze(0),class_f)

        text_tc = self.tc(text_f) 
        text_tc = self.relu(self.l1(text_tc))
        text_tc = text_tc.unsqueeze(0).unsqueeze(0)

        text_sent,atn = self.attn(sentiment_f.to(torch.float32),text_tc.to(torch.float32),text_tc.to(torch.float32))
        x = self.attnf(experts_result.unsqueeze(0).unsqueeze(0),text_sent)
     
        return x.squeeze(0).squeeze(0)
    

import time

def detect01(n,lab):
    if n>0.5:
      n=1
    else:
      n=0
    if n == lab:
      return 1
    else:
      return 0

def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    record = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        time1=time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()
        acc = 0
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        model.cuda()
        correct_counts = 0

        FP = 0
        FN = 0
        rumor = 0
        nonrumor = 0

        TP = 0
        TN = 0
        for i in range(split):
            train_img = imgs_dataset[i]

            text_f = bert(texts[i],200)
            classlabel = classify_model(train_img.unsqueeze(0).to(device))
            classlabel = torch.exp(classlabel)
            img_f = resnet(train_img.unsqueeze(0))
            text_sentiment = get_sentiment(texts[i])
            label = torch.tensor(labels[i]).to(torch.float32).to(device)
            optimizer.zero_grad()

            img_f = img_f.unsqueeze(0).unsqueeze(0).to(device)
            classlabel = classlabel.unsqueeze(0).unsqueeze(0).to(device)
            text_f = text_f.unsqueeze(0).to(device)
            text_sentiment = text_sentiment.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

            output = model(img_f,classlabel,text_f,text_sentiment)[0]
            loss = loss_function(output, label.unsqueeze(0))
            loss.backward()

            optimizer.step()

            train_loss += loss.item() * text_f.size(0)
            predictions = detect01(output,label)
            correct_counts+=predictions
            
            if label>0.5:
                rumor+=1
                if predictions==1:
                    TP+=1
            else:
                nonrumor+=1
                if predictions==1:
                    TN+=1
                    
            FN = rumor - TP
            FP = nonrumor - TN 
     
                    
            train_rumor_acc = TP/max(rumor,1) 
            train_nonrumor_acc = TN/max(nonrumor,1)

            train_precision = TP/max(1,FP+TP) 
            trainF1 = 2*train_rumor_acc*train_precision/max(1,(train_rumor_acc+train_precision))
            acc = correct_counts/(i+1)
            train_acc = acc

            
            if (i+1)%100 == 0:
              time2 = time.time()
              print(output,' ',label,' ',predictions,' ',acc)
              print(f'{i}/{split}   train_acc={train_acc}   time={time2-time1}')
              time1 = time.time()

        with torch.no_grad():
            model.eval()
            print('valid:')
            correct_counts = 0
            FP = 0
            FN = 0
            rumor = 0
            nonrumor = 0

            TP = 0
            TN = 0
            for j in range(len(texts)-split):
                test_img = imgs_dataset[j+split]
                test_texts = texts[j+split]

                text_f = bert(test_texts,200)
                classlabel = classify_model(test_img.unsqueeze(0).to(device))
                classlabel = torch.exp(classlabel)
                img_f = resnet(test_img.unsqueeze(0))
                text_sentiment = get_sentiment(test_texts)

                label = torch.tensor(labels[j+split]).to(torch.float32).to(device)
                optimizer.zero_grad()

                img_f = img_f.unsqueeze(0).unsqueeze(0).to(device)

                classlabel = classlabel.unsqueeze(0).unsqueeze(0).to(device)
                text_f = text_f.unsqueeze(0).to(device)
                text_sentiment = text_sentiment.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

                output = model(img_f,classlabel,text_f,text_sentiment)[0]
                loss = loss_function(output, label.unsqueeze(0))
                valid_loss += loss.item() * text_f.size(0)
                predictions = detect01(output,label)
                correct_counts+=predictions
                if label>0.5:
                    rumor+=1
                    if predictions==1:
                        TP+=1
                else:
                    nonrumor+=1
                    if predictions==1:
                        TN+=1

                FN = rumor - TP
                FP = nonrumor - TN 
                
                valid_rumor_acc = TP/max(rumor,1)
                valid_nonrumor_acc = TN/max(nonrumor,1)
                valid_precision = TP/max(1,FP+TP) 
                validF1 = 2*valid_rumor_acc*valid_precision/max(1,(valid_rumor_acc+valid_precision))   
                acc = correct_counts/(j+1)
                valid_acc = acc

                if (j+1)%100 == 0:
                    time2 = time.time()
                    print(output,' ',label,' ',predictions,' ',acc)
                    print(f'{j}/{len(texts)-split}   valid_acc={valid_acc}   time={time2-time1}')
                    
                    time1 = time.time()
                

        avg_train_loss = train_loss / split

        avg_valid_loss = valid_loss / (len(texts)-split)


        record.append([avg_train_loss, avg_valid_loss, \
                       train_acc, valid_acc,train_rumor_acc,\
                        train_nonrumor_acc,valid_rumor_acc,\
                        valid_nonrumor_acc,train_precision,\
                        valid_precision,trainF1,validF1,TP,FP,TN,FN])

        if valid_acc > best_acc  :
            best_acc = valid_acc
            best_epoch = epoch + 1
            best_model = model
            
            torch.save(best_model,'./best_model')
        torch.save(record, './model_saved_record')
        epoch_end = time.time()

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, train_acc * 100, avg_valid_loss, valid_acc * 100,
                epoch_end - epoch_start))
        print("Epoch: {:03d}, Training: rumor_acc: {:.4f}%, nonrumor_acc: {:.4f}%, \n\t\tValidation: rumor_acc: {:.4f}%, nonrumor_acc: {:.4f}%".format(
                epoch + 1, train_rumor_acc*100, train_nonrumor_acc * 100, valid_rumor_acc*100, valid_nonrumor_acc * 100))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        print('-'*60)

    return model, record, best_model

t = PFFN()
t = torch.load('best_model')

loss_func = torch.nn.BCELoss()
optimizer = torch.optim.SGD(t.parameters(),lr=0.001)
num_epochs = 80

trained_model, record, best_model = train_and_valid(t, loss_func, optimizer, num_epochs)

record = np.array(record)

