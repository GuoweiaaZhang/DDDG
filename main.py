import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
import time
from module import Classifier, Classifier_ad, Classifier_all, CNN, Projector, Masker, RandMix, loglikeli, club
from construct_loader import construct_loader
from load_data import load_data
import torch.nn.functional as F
# =================== 参数设置 ===================
epoch = 100
lr = 0.0005
weight_decay = 0.00005
n_classes = 4
noise = 1
tau = 0.01
T1 = 1
BatchSize = 40
# ===================================== Data loading =====================================
# JNU 数据路径  JNU data format is sample size * 1 * 2049, The 2049 dimension contains 2048 dimensions of sample points and 1 dimension of labels.
x_600_path = r'E:\Code\DDDG/DATA/x_600.mat'
x_train_x_600,  x_train_y_600,  x_test_x_600,  x_test_y_600 = load_data(x_600_path, 'x_600', 4, 200, [])  # Data segmentation and data fft transformation

x_800_path = r'E:\Code\DDDG/DATA/x_800.mat'
x_train_x_800,  x_train_y_800,  x_test_x_800,  x_test_y_800 = load_data(x_800_path, 'x_800', 4, 200, [])

x_1000_path = r'E:\Code\DDDG/DATA/x_1000.mat'
x_train_x_1000,  x_train_y_1000,  x_test_x_1000,  x_test_y_1000 = load_data(x_1000_path, 'x_1000', 4, 200, [])

Task_data = [x_train_x_600.cuda(), x_train_y_600.view(-1),
            x_train_x_800.cuda(), x_train_y_800.view(-1),
            x_train_x_1000.cuda(), x_train_y_1000.view(-1)]

# Generalized Task Setting. Setting up different generalization tasks by adjusting the index
Train_X1 = Task_data[2]  # Source Domain Data
Train_Y1 = Task_data[3]  # Source Domain labels
data_aug = RandMix(noise).cuda()
Train_X2 = data_aug(Train_X1)
Train_Y2 = Task_data[1]
Test_X = Task_data[0]  # Target Domain Data
Test_Y = Task_data[1]  # Target Domain labels

# Standardization of enhanced data
min_val = Train_X2.min()
max_val = Train_X2.max()
Train_X2 = (Train_X2 - min_val) / (max_val - min_val)

# Construct training and testing data loaders
TrainX1_loader = construct_loader([Train_X1, Train_Y1], batch_size = BatchSize)  # Original training set loader
TrainX2_loader = construct_loader([Train_X2, Train_Y2], batch_size = BatchSize)  # Enhanced training set loader
Test_loader = construct_loader([Test_X, Test_Y], batch_size = BatchSize)      # Test set loader

# Store all loaders in a list for model training
TR_dataloader = [TrainX1_loader, TrainX2_loader, Test_loader]

# ==================Initialize models =====================
# Initialize models and optimizers if CUDA is available
if torch.cuda.is_available():
    # Instantiate models
    feature_extractor = CNN(n_classes).cuda()
    projector = Projector().cuda()
    classifier = Classifier(n_classes).cuda()
    classifier_ad = Classifier_ad(n_classes).cuda()
    classifier_all = Classifier_all(n_classes).cuda()
    masker = Masker().cuda()

    # Define loss functions
    kl_loss = nn.KLDivLoss(reduction='batchmean')  # KL divergence loss
    criterion = nn.CrossEntropyLoss()  # Cross entropy loss
    amp_grad_scaler = GradScaler()  # Mixed precision gradient scaler

    Acc = []  # List to store accuracy per epoch
    # Initialize data placeholders
    t_x, s1_y, s1_x, s2_x, s2_y = [], [], [], [], []
    # Training loss tracking lists
    step1_loss_list = list()
    step2_loss_list = list()
    # Placeholder variables for step1_loss and step2_loss 
    step1_loss = None
    step2_loss = None
    # Define optimizers for each model component
    optimizer = optim.Adam(feature_extractor.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_prompt = optim.Adam(projector.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_classifier_ad = optim.Adam(classifier_ad.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_masker = optim.Adam(masker.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_classifier_all = optim.Adam(classifier_all.parameters(), lr=lr, weight_decay=weight_decay)
    # =================== Model Training ===================
    for i in range(epoch):
        # Setting the model to training mode
        feature_extractor.train()
        projector.train()
        classifier.train()
        classifier_ad.train()
        classifier_all.train()
        masker.train()

        start_time = time.time()
        # Get Data Loader
        iter_source1, iter_source2, iter_target = iter(TR_dataloader[0]), iter(TR_dataloader[1]), iter(TR_dataloader[2])

        # Small batch data loops
        for step in range(len(TR_dataloader[0])):

            s1_data, s1_label = iter_source1.__next__()
            s2_data, s2_label = iter_source2.__next__()

            if torch.cuda.is_available():
                s1_x = Variable(s1_data).type(dtype=torch.float32).cuda()
                s1_y = Variable(s1_label).type(dtype=torch.LongTensor).cuda()
                s2_x = Variable(s2_data).type(dtype=torch.float32).cuda()
                s2_y = Variable(s2_label).type(dtype=torch.LongTensor).cuda()
            # -----Optimizer gradient clearing-------
            optimizer.zero_grad()
            optimizer_classifier.zero_grad()
            optimizer_classifier_ad.zero_grad()
            optimizer_prompt.zero_grad()
            # Training classifier, classifier_ad, prompt
            with autocast():
                batch = torch.cat([s1_x, s2_x], dim=0)
                labels = torch.cat([s1_y, s2_y], dim=0)
                # Extracting domain invariant and domain specific features
                f_invariant, f_specific = feature_extractor.forward_first_layer(s1_x, tau=tau)
                f_invariant_fda, f_specific_fda = feature_extractor.forward_first_layer(s2_x, tau=tau)

                f_invariant_p = projector(f_invariant)[1]
                f_specific_p = projector(f_specific)[1]
                f_invariant_fda_p = projector(f_invariant_fda)[1]
                f_specific_fda_p = projector(f_specific_fda)[1]

                # Mapping of domain-invariant features to domain-specific features
                f_invariant_p1 = projector(f_invariant)[0]
                f_specific_p1 = projector(f_specific)[0]
                f_invariant_fda_p1 = projector(f_invariant_fda)[0]
                f_specific_fda_p1 = projector(f_specific_fda)[0]

                mu = f_specific_p['mu']
                logvar = f_specific_p['logvar']
                y_samples = f_specific_fda_p['Embedding']

                likeli = -loglikeli(mu, logvar, y_samples)  # Negative log likelihood

                di_loss = F.l1_loss(f_invariant_p1, f_invariant_fda_p1, reduction='mean') + F.l1_loss(f_invariant_fda_p1, f_invariant_p1, reduction='mean')

                div = club(mu, logvar, y_samples)  # Mutual Information Upper Bound Estimation

                L_DC = 10 * di_loss + 0.001 * div + 1 * likeli  # Dual Contrastive Disentanglement

                features = feature_extractor.forward(batch, tau=tau)[0]
                masks_cn = masker(features.detach())             # Class-irrelated feature mask
                masks_cr = torch.ones_like(masks_cn) - masks_cn  # Class-related feature mask

                if i <= 40:  # Iterations below i do not use class masks
                    masks_cn = torch.ones_like(features.detach())
                    masks_cr = torch.ones_like(features.detach())

                features_cn = features * masks_cn
                features_cr = features * masks_cr
                scores_cn = classifier(features_cn)
                scores_cr = classifier_ad(features_cr)

                loss_cls_cn = criterion(scores_cn, labels.long())
                loss_cls_cr = criterion(scores_cr, labels.long())

                loss_step1 = 0.01 * L_DC + loss_cls_cn + loss_cls_cr  # Loss of the first training step. It is recommended to adjust the hyperparameters so that loss_step1 and loss_step2 are on the same order of magnitude.

            amp_grad_scaler.scale(loss_step1).backward()
            amp_grad_scaler.unscale_(optimizer_prompt)
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.unscale_(optimizer_classifier)
            amp_grad_scaler.unscale_(optimizer_classifier_ad)
            amp_grad_scaler.step(optimizer_prompt)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.step(optimizer_classifier)
            amp_grad_scaler.step(optimizer_classifier_ad)
            amp_grad_scaler.update()

            # Training classifiers_all
            optimizer_classifier_all.zero_grad()
            with autocast():
                features = feature_extractor.forward(batch, tau=tau)[0]
                scores_all = classifier_all(features)
                loss_cls_all = criterion(scores_all, labels.long())
                loss = loss_cls_all
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer_classifier_all)
            amp_grad_scaler.step(optimizer_classifier_all)
            amp_grad_scaler.update()

            # Training class mask
            optimizer_masker.zero_grad()
            with autocast():
                features = feature_extractor.forward(batch, tau=tau)[0]
                masks_cn = masker(features.detach())
                masks_cr = torch.ones_like(masks_cn) - masks_cn

                features_cn = features * masks_cn  # Class-irrelated feature
                features_cr = features * masks_cr  # Class-related feature
                scores_cn = classifier(features_cn)
                scores_cr = classifier_ad(features_cr)
                scores_all = classifier_all(features)

                loss_cls_cn = criterion(scores_cn, labels.long())
                loss_cls_cr = criterion(scores_cr, labels.long())
                loss_cls_all = criterion(scores_all, labels.long())

                loss_kl = kl_loss(torch.log_softmax((scores_cr / T1), dim=-1), torch.softmax((scores_all / T1), dim=-1))

                loss_step2 = -0.1 * loss_cls_cn + loss_cls_cr + 0.1 * loss_kl  # Adversarial mask disentanglement loss

            amp_grad_scaler.scale(loss_step2).backward()
            amp_grad_scaler.unscale_(optimizer_masker)
            amp_grad_scaler.unscale_(optimizer_classifier)
            amp_grad_scaler.step(optimizer_masker)
            amp_grad_scaler.step(optimizer_classifier)
            amp_grad_scaler.update()

        if loss_step1 is not None:
            step1_loss_list.append(loss_step1.detach().cpu().numpy())
            mean_step1_loss = np.mean(step1_loss_list)
        else:
            mean_step1_loss = 0
        if loss_step2 is not None:
            step2_loss_list.append(loss_step2.detach().cpu().numpy())
            mean_step2_loss = np.mean(step2_loss_list)
        else:
            mean_step2loss = 0

        end_time = time.time()
        time_len = end_time - start_time
        # =================== Model Testing ===================
        with torch.no_grad():
            feature_extractor.eval()
            classifier.eval()
            masker.eval()
            classifier_ad.eval()
            classifier_all.eval()
            features = feature_extractor(Train_X1.cuda(), tau)[0]
            scores_sup = classifier(features)
            pre_label = torch.max(torch.softmax(scores_sup, 1), 1)[1].cpu().numpy()
            train_acc = accuracy_score(pre_label, Train_Y1.detach().numpy())
            features_test = feature_extractor.forward(Test_X.cuda(), tau)[0]
            scores_test = classifier_ad(features_test)
            test_label = torch.max(torch.softmax(scores_test, 1), 1)[1].cpu().numpy()
            test_acc = accuracy_score(test_label, Test_Y.detach().numpy())  # Generalization accuracy
            Acc.append(test_acc)

        if i == 0 or (i + 1) % 1 == 0:
            print(
                '==> Epoch: {}/{}, step1_loss: {:.5f}, step2_loss: {:.5f}, Test_Acc: {:.5f}, Running time is {:.3f} s'.format(
                    i + 1, epoch, mean_step1_loss, mean_step2_loss, test_acc, time_len))
