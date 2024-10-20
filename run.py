import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import time
from Util import load_data_n_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def generate_and_plot_confusion_matrix(output_folder, dataset_name, y_true, y_pred):
    # 根据数据集名称确定标签
    if dataset_name == 'Widar6':
        labels = ['Push&Pull', 'Sweep', 'Clap', 'Slide', 'Draw-O',  'Draw-Zigzag']
    elif dataset_name == 'Widar10':
        labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    elif dataset_name=="Widar":
        labels=['Push&Pull', 'Sweep', 'Clap', 'Slide', 'Drwa-N(H)','Draw-O(H)','Draw-Retangle(H)', 'Draw-Triangle(H)',
                'Draw-Zigzag(H)','Draw-Zigzag(V)','Draw-N(V)','Draw-O(V)',  '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    elif dataset_name=="NTU-Fi_HAR":
        labels=['Box', 'Circle', 'Clean', 'Fall','Run','Walk']
    elif dataset_name=="UT_HAR_data":
        labels=['Lie down', 'Fall', 'Walk','Pick up', 'Run','Sit down','Stand up']

    else:
        raise ValueError("Unsupported dataset name")

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    column_sums = cm.sum(axis=0)
    epsilon = 1e-10
    proportion_per_cell = cm / (column_sums[np.newaxis, :]+ epsilon)
    if dataset_name=='Widar':
        proportion_per_cell = np.round(proportion_per_cell, 4)
        plt.figure(figsize=(20, 14))
        sns.heatmap(proportion_per_cell, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    else:
        proportion_per_cell = np.round(proportion_per_cell, 4)
        plt.figure(figsize=(10, 7))
        sns.heatmap(proportion_per_cell, annot=True, fmt='.4f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    # 可视化混淆矩阵


    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(output_folder + "confusion_matrix" + ".png"), dpi=150)
    plt.close()


def plot_and_save_results(epoch_test_acc, epoch_test_loss, epoch_val_acc, epoch_val_loss, output_folder):
    epochs = list(range(1, len(epoch_test_acc) + 1))

    # 将字符串列表转换为浮点数列表
    epoch_test_acc = [float(acc) for acc in epoch_test_acc]
    epoch_test_loss = [float(loss) for loss in epoch_test_loss]
    epoch_val_acc = [float(acc) for acc in epoch_val_acc]
    epoch_val_loss = [float(loss) for loss in epoch_val_loss]

    # 绘制准确率图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, epoch_test_acc, label='Test Acc')
    plt.plot(epochs, epoch_val_acc, label='Val Acc')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制损失图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, epoch_test_loss, label='Test Loss')
    plt.plot(epochs, epoch_val_loss, label='Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_validation_results.png'))
    plt.close()  # 关闭图表，以避免重复显示


def train(model, tensor_loader, num_epochs, learning_rate, criterion, device, output_folder, test_tensor_loader):
    start_time = time.time()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    epoch_test_acc = []
    epoch_test_loss = []
    epoch_val_acc = []
    epoch_val_loss = []
    folder_path = output_folder
    file_name = "epoch_acc_loss_text.txt"
    full_path = os.path.join(folder_path, file_name)
    file_name2 = "best_model.pth"
    best_model_path = os.path.join(folder_path, file_name2)
    best_epoch_acc = 0.0
    best_epoch_Loss = 200.0

    #l1_lambda = 1e-4

    for epoch in range(num_epochs):
        model.train()  # 启用 Batch Normalization 和 Dropout
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tensor_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)

            optimizer.zero_grad()
            # 梯度置零，也就是把loss关于weight的导数变成0.
            # 为什么每一轮batch需要设置optimizer.zero_grad：
            # 根据pytorch中的backward()
            # 函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs, labels)
            #l1_loss = sum(p.abs().sum() for p in model.parameters())
            #loss=loss+l1_loss*l1_lambda
            loss.backward()  # 将损失loss向输入侧进行反向传播
            optimizer.step()  # 优化器对的值进行更新

            epoch_loss += loss.item() * inputs.size(0)  # train_loss很可能是求一个epoch的loss之和
            predict_y = torch.argmax(outputs, dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss / len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy / len(tensor_loader)
        if epoch_loss< best_epoch_Loss :
            best_epoch_Loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)
        epoch_test_acc.append(f"{epoch_accuracy:.4f}")
        epoch_test_loss.append(f"{epoch_loss:.4f}")
        output_string = 'Epoch:{}, Accuracy:{:.9f},Loss:{:.9f}'.format(epoch + 1, float(epoch_accuracy),
                                                                       float(epoch_loss))
        with open(full_path, 'a') as f:
            f.write(output_string)
        f.close()
        print('Epoch:{}, Accuracy:{:.9f},Loss:{:.9f}'.format(epoch + 1, float(epoch_accuracy), float(epoch_loss)))

        model.eval()
        test_acc = 0
        test_loss = 0
        with torch.no_grad():
            for data in test_tensor_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels.to(device)
                labels = labels.type(torch.LongTensor)

                outputs = model(inputs)
                outputs = outputs.type(torch.FloatTensor)
                outputs.to(device)

                loss = criterion(outputs, labels)
                predict_y = torch.argmax(outputs, dim=1).to(device)
                accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
                test_acc += accuracy
                test_loss += loss.item() * inputs.size(0)
            test_acc = test_acc / len(test_tensor_loader)
            test_loss = test_loss / len(test_tensor_loader.dataset)
            epoch_val_acc.append(f"{test_acc:.4f}")
            epoch_val_loss.append(f"{test_loss:.4f}")
            print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc), float(test_loss)))
            output_string = 'Val Accuracy:{:.4f},Loss:{:.5f}'.format(float(test_acc), float(test_loss))
            with open(full_path, 'a') as f:
                f.write(output_string + '\n')
            f.close()
    end_time = time.time()
    training_time = end_time - start_time
    print("**********************训练已完成**************************")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Epoch best model saved with accuracy: {best_epoch_Loss}")
    with open(full_path, 'a') as file:
        file.write("**********************训练已完成**************************\n")
        file.write(f"Epoch best model saved with opoch loss: {best_epoch_Loss}\n")
        file.write("Test Accuracies:\n")
        file.write(', '.join(epoch_test_acc))
        file.write("\n")
        file.write(f"Total training time: {training_time:.2f} seconds\n")
    f.close()

    plot_and_save_results(epoch_test_acc, epoch_test_loss, epoch_val_acc, epoch_val_loss, folder_path)
    return


def test(model, tensor_loader, criterion, device, output_folder, dataset_name):
    start_time = time.time()
    model_file_name = 'best_model.pth'
    model_path = os.path.join(output_folder, model_file_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    file_name = "epoch_acc_loss_text.txt"
    full_path = os.path.join(output_folder, file_name)
    full_path1 = output_folder
    test_acc = 0
    test_loss = 0
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)

        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)

        loss = criterion(outputs, labels)
        predict_y = torch.argmax(outputs, dim=1).to(device)
        all_preds.extend(predict_y.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc / len(tensor_loader)
    test_loss = test_loss / len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc), float(test_loss)))
    output_string = 'Val Accuracy:{:.4f},Loss:{:.5f}'.format(float(test_acc), float(test_loss))
    end_time = time.time()
    testing_time = end_time - start_time
    with open(full_path, 'a') as f:
        f.write(output_string + '\n')
        f.write(f"Total testing time: {testing_time:.2f} seconds\n")
    f.close()
    generate_and_plot_confusion_matrix(full_path1, dataset_name, all_labels, all_preds)
    return


def main():
    root = "D:/WIFIData/data1/"
    print(
        "************************************************************************************************************")
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')  # 创建解析器
    parser.add_argument('--dataset',
                        choices=['UT_HAR_data', 'NTU-Fi-HumanID', 'NTU-Fi_HAR', 'Widar', 'Widar3', 'Widar6',
                                 'Widar10'])  # 添加函数
    parser.add_argument('--model',
                        choices=['MLP', 'LeNet', 'ResNet18', 'ResNet50', 'ResNet101', 'RNN', 'GRU', 'LSTM', 'BiLSTM',
                                 'CNN+GRU', 'ViT', 'ViTLSTM', 'ViTBiLSTM', 'CNNLSTM','CNNViT','TCNBiLSTM','ViTBiLSTMMCF','SAE','ABLSTM',
                                 'THAT','Sanet','ACT','CNN+BiLSTM'])
    args = parser.parse_args()  # 解析参数
    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root)
    model_name = model.__class__.__name__

    OUTPUT_FOLDER_PATTERN01 = "{0}/"
    output_folder_dataset = OUTPUT_FOLDER_PATTERN01.format(args.dataset)
    if not os.path.exists(output_folder_dataset):
        os.makedirs(output_folder_dataset)

    OUTPUT_FOLDER_PATTERN = "{0}/"
    output_foldername = OUTPUT_FOLDER_PATTERN.format(model_name)
    output_folder = os.path.join(output_folder_dataset, output_foldername)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(
        model=model,
        tensor_loader=train_loader,
        num_epochs=train_epoch,
        learning_rate=0.001,
        criterion=criterion,
        device=device,
        output_folder=output_folder,
        test_tensor_loader=test_loader
    )
    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device,
        output_folder=output_folder,
        dataset_name=args.dataset
    )
    return


if __name__ == "__main__":
    main()
