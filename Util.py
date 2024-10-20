from Dataset import *
from UT import *
from NTU import *
from widar import *
from widar10 import *
from widar6 import *
import torch


def load_data_n_model(dataset_name, model_name, root):
    classes = {'UT_HAR_data': 7, 'NTU-Fi_HAR': 6, 'Widar': 22,'Widar3': 22,'Widar6':6,'Widar10':10,'Widardata6_six':6}#Widar和Widar3采用的数据集一样，只是模型不一样，Widarraw为原始四个数据集，Widar为要志敏的数据集，Widar6-10为数据集手势分割，与原论文一样分为两个，分割数据集也为志敏的数据集
    if dataset_name == 'UT_HAR_data':
        print('using dataset: UT-HAR DATA')
        data = UT_HAR_dataset(root)
        train_set = torch.utils.data.TensorDataset(data['X_train'], data['y_train'])
        test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'], data['X_test']), 0),
                                                  torch.cat((data['y_val'], data['y_test']), 0))
        #拼接函数，torch.cat
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True,
                                                   drop_last=True, num_workers=4, pin_memory=True)  # drop_last=True
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        if model_name == 'MLP':
            print("using model: MLP")
            model = UT_HAR_MLP()
            train_epoch = 200
        elif model_name == 'LeNet':
            print("using model: LeNet")
            model = UT_HAR_LeNet()
            train_epoch = 200  # 40
        elif model_name == 'ResNet18':
            print("using model: ResNet18")
            model = UT_HAR_ResNet18()
            train_epoch = 200  # 70
        elif model_name == 'ResNet50':
            print("using model: ResNet50")
            model = UT_HAR_ResNet50()
            train_epoch = 200  # 100
        elif model_name == 'ResNet101':
            print("using model: ResNet101")
            model = UT_HAR_ResNet101()
            train_epoch = 200  # 100
        elif model_name == 'RNN':
            print("using model: RNN")
            model = UT_HAR_RNN()
            train_epoch = 3000
        elif model_name == 'GRU':
            print("using model: GRU")
            model = UT_HAR_GRU()
            train_epoch = 200
        elif model_name == 'LSTM':
            print("using model: LSTM")
            model = UT_HAR_LSTM()
            train_epoch = 200
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = UT_HAR_BiLSTM()
            train_epoch = 200
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU")
            model = UT_HAR_CNN_GRU()
            train_epoch = 150  # 20
        elif model_name == 'ViT':
            print("using model: ViT")
            model = UT_HAR_ViT()
            train_epoch = 200  # 100
        elif model_name == 'ViTLSTM':
            print("using model: ViTLSTM")
            model = UT_HAR_ViTLSTM()
            train_epoch = 200  # 100
        elif model_name == 'ViTBiLSTM':
            print("using model: ViTBiLSTM")
            model = UT_HAR_ViTBiLSTM()
            train_epoch = 200  # 100
        elif model_name == 'ViTBiLSTMMCF':
            print("using model: ViTBiLSTMMCF")
            model = ViTBiLSTMMCF()
            train_epoch = 100
        elif model_name == 'SAE':
            print("using model: SAE")
            model = UT_HAR_SAE()
            train_epoch = 100
        elif model_name == 'ABLSTM':
            print("using model: ABLSTM")
            model = UT_HAR_ABLSTM()
            train_epoch = 100
        elif model_name == 'THAT':
            print("using model: THAT")
            model = UT_HAR_THAT()
            train_epoch = 100
        elif model_name == 'Sanet':
            print("using model: Sanet")
            model = UT_HAR_Sanet()
            train_epoch = 300
        elif model_name == 'ACT':
            print("using model: ACT")
            model = UT_HAR_ACT()
            train_epoch = 300
        elif model_name == 'CNN+BiLSTM':
            print("using model: CNN+BiLSTM")
            model = UT_HAR_CNN_BiLSTM()
            train_epoch = 100
        return train_loader, test_loader, model, train_epoch


    elif dataset_name == 'NTU-Fi_HAR':
        print('using dataset: NTU-Fi_HAR')
        num_classes = classes['NTU-Fi_HAR']
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR_raw/train_amp/'), batch_size=16,
                                                   shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR_raw/test_amp/'), batch_size=16,
                                                  shuffle=False, num_workers=4, pin_memory=True)
        if model_name == 'MLP':
            print("using model: MLP")
            model = NTU_Fi_MLP(num_classes)
            train_epoch = 30  # 10
        elif model_name == 'LeNet':
            print("using model: LeNet")
            model = NTU_Fi_LeNet(num_classes)
            train_epoch = 30  # 10
        elif model_name == 'ResNet18':
            print("using model: ResNet18")
            model = NTU_Fi_ResNet18(num_classes)
            train_epoch = 30
        elif model_name == 'ResNet50':
            print("using model: ResNet50")
            model = NTU_Fi_ResNet50(num_classes)
            train_epoch = 30  # 40
        elif model_name == 'ResNet101':
            print("using model: ResNet101")
            model = NTU_Fi_ResNet101(num_classes)
            train_epoch = 30
        elif model_name == 'RNN':
            print("using model: RNN")
            model = NTU_Fi_RNN(num_classes)
            train_epoch = 70
        elif model_name == 'GRU':
            print("using model: GRU")
            model = NTU_Fi_GRU(num_classes)
            train_epoch = 30  # 20
        elif model_name == 'LSTM':
            print("using model: LSTM")
            model = NTU_Fi_LSTM(num_classes)
            train_epoch = 40  # 20
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = NTU_Fi_BiLSTM(num_classes)
            train_epoch = 30  # 20
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU")
            model = NTU_Fi_CNN_GRU(num_classes)
            train_epoch = 50  # 20
        elif model_name == 'ViT':
            print("using model: ViT")
            model = NTU_Fi_ViT(num_classes=num_classes)
            train_epoch = 30
        elif model_name == 'ViTBiLSTM':
            print("using model: ViTBiLSTM")
            model = NTU_Fi_ViTBiLSTM(num_classes=num_classes)
            train_epoch = 50  # 100
        # elif model_name == 'ViTBiLSTM':
        #     print("using model: ViTBiLSTM")
        #     model = ViTBiLSTM(num_classes=num_classes)
        #     train_epoch = 100  # 100
        return train_loader, test_loader, model, train_epoch

    # ***********************************************************************************

    elif dataset_name == 'Widar':
    #else:
        print('using dataset: Widar')
        num_classes = classes['Widar']
        train_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/train/'), batch_size=64,
                                                   shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/test/'), batch_size=64,
                                                  shuffle=False, num_workers=4, pin_memory=True)
        if model_name == 'MLP':
            print("using model: MLP")
            model = Widar_MLP(num_classes)
            train_epoch = 30  # 20
        elif model_name == 'LeNet':
            print("using model: LeNet")
            model = Widar_LeNet(num_classes)
            train_epoch = 100  # 40
        elif model_name == 'ResNet18':
            print("using model: ResNet18")
            model = Widar_ResNet18(num_classes)
            train_epoch = 100
        elif model_name == 'ResNet50':
            print("using model: ResNet50")
            model = Widar_ResNet50(num_classes)
            train_epoch = 100  # 40
        elif model_name == 'ResNet101':
            print("using model: ResNet101")
            model = Widar_ResNet101(num_classes)
            train_epoch = 100
        elif model_name == 'RNN':
            print("using model: RNN")
            model = Widar_RNN(num_classes)
            train_epoch = 50
        elif model_name == 'CNN+BiLSTM':
            print("using model: CNN_BiLSTM")
            model = Widar_CNN_BiLSTM(num_classes)
            train_epoch = 100
        elif model_name == 'GRU':
            print("using model: GRU")
            model = Widar_GRU(num_classes)
            train_epoch = 200
        elif model_name == 'LSTM':
            print("using model: LSTM")
            model = Widar_LSTM(num_classes)
            train_epoch = 150  # 20
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = Widar_BiLSTM(num_classes)
            train_epoch = 100
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU")
            model = Widar_CNN_GRU(num_classes)
            train_epoch = 100  # 20
        elif model_name == 'ViT':
            print("using model: ViT")
            model = Widar_ViT(num_classes=num_classes)
            train_epoch = 50
        elif model_name == 'ViTLSTM':
            print("using model: ViTLSTM")
            model = Widar_ViTLSTM(num_classes=num_classes)
            train_epoch = 1  # 100
        elif model_name == 'ViTBiLSTM':
            print("using model: ViTBiLSTM")
            model = Widar_ViTBiLSTM(num_classes=num_classes)
            train_epoch = 100  # 100
        elif model_name == 'CNNViT':
            print("using model: CNNViT")
            model = CNNVIT
            train_epoch = 100
        elif model_name == 'ViTBiLSTMMCF':
            print("using model: ViTBiLSTMMCF")
            model = Widar_ViTBiLSTMMCF(num_classes=num_classes)
            train_epoch = 150
        elif model_name == 'ABLSTM':
            print("using model: ABLSTM")
            model = Widar_ABLSTM(num_classes=num_classes)
            train_epoch = 50
        elif model_name == 'SAE':
            print("using model: SAE")
            model = Widar_SAE(num_classes=num_classes)
            train_epoch = 100
        return train_loader, test_loader, model, train_epoch

    elif dataset_name == 'Widar3':
    #else:
        print('using dataset: Widar3')
        num_classes = classes['Widar3']
        train_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/train/'), batch_size=64,
                                                   shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/test/'), batch_size=128,
                                                  shuffle=False, num_workers=4, pin_memory=True)
        if model_name == 'ViTBiLSTMMCF':
            print("using model: ViTBiLSTMMCF")
            model = Widar10_ViTBiLSTMMCF(num_classes=num_classes)
            train_epoch = 100  # 100
        return train_loader, test_loader, model, train_epoch
    elif dataset_name == 'Widar6':
    #else:
        print('using dataset: Widar6')
        num_classes = classes['Widar6']
        train_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata6_10/six_9_1/train/'), batch_size=32,
                                                   shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata6_10/six_9_1/test/'), batch_size=64,
                                                  shuffle=False, num_workers=4, pin_memory=True)
        if model_name == 'ViTBiLSTMMCF':
            print("using model: ViTBiLSTMMCF")
            model = Widar6_ViTBiLSTMMCF(num_classes=num_classes)
            train_epoch = 50  # 100
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU")
            model = Widar6_CNN_GRU(num_classes=num_classes)
            train_epoch = 100   # 100
        elif model_name == 'ViT':
            print("using model: ViT")
            model = Widar6_ViT(num_classes=num_classes)
            train_epoch = 100   # 100
        return train_loader, test_loader, model, train_epoch
    elif dataset_name == 'Widar10':
    #else:
        print('using dataset: Widar10')
        num_classes = classes['Widar10']
        train_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata6_10/ten_9_1/train/'), batch_size=50,
                                                   shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata6_10/ten_9_1/test/'), batch_size=25,
                                                  shuffle=False, num_workers=4, pin_memory=True)
        if model_name == 'ViTBiLSTMMCF':
            print("using model: ViTBiLSTMMCF")
            model = Widar10_ViTBiLSTMMCF(num_classes=num_classes)
            train_epoch = 100  # 100
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU")
            model = Widar10_CNN_GRU(num_classes=num_classes)
            train_epoch = 100  # 100
        return train_loader, test_loader, model, train_epoch
    elif dataset_name == 'Widar6_six':
    #else:
        print('using dataset: Widar6')
        num_classes = classes['Widar6']
        train_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata6_six/six/train/'), batch_size=64,
                                                   shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata6_six/six/test/'), batch_size=128,
                                                  shuffle=False, num_workers=4, pin_memory=True)
        if model_name == 'ViTBiLSTM':
            print("using model: ViTBiLSTM")
            model = Widar3_ViTBiLSTM(num_classes=num_classes)
            train_epoch = 50  # 100

        return train_loader, test_loader, model, train_epoch

    # else:
    #     num_classes = classes['Widar']
    #     train_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/train/'), batch_size=64,
    #                                                shuffle=False)
    #     test_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/test/'), batch_size=128,
    #                                               shuffle=False)
    #
    #     print("using model: MLP")
    #     model = Widar_MLP(num_classes)
    #     train_epoch = 30  # 20
    #
    #     return train_loader, test_loader, model, train_epoch

