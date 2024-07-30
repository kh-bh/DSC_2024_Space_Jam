import torch
import torch.optim as optim


# def train(model, train_loader, val_loader, num_epochs, learning_rate):
#     criterion = torch.nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for batch_ecg, batch_activation in train_loader:
#             batch_ecg, batch_activation = batch_ecg.to(device), batch_activation.to(device)
#             optimizer.zero_grad()
#             outputs = model(batch_ecg)
#             loss = criterion(outputs, batch_activation)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#         avg_train_loss = running_loss / len(train_loader)
#         writer.add_scalar('Loss/train', avg_train_loss, epoch)
#
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for batch_ecg, batch_activation in val_loader:
#                 batch_ecg, batch_activation = batch_ecg.to(device), batch_activation.to(device)
#                 outputs = model(batch_ecg)
#                 val_loss += criterion(outputs, batch_activation).item()
#
#             avg_val_loss = val_loss / len(val_loader)
#             writer.add_scalar('Loss/val', avg_val_loss, epoch)
#
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss / len(val_loader):.4f}')


# def train(model, train_loader, val_loader, num_epochs, learning_rate):
#     criterion = torch.nn.MSELoss().to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
#     #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
#
#     start_time = time.time()
#
#     for epoch in range(num_epochs):
#         model.train()
#         for batch_ecg, batch_activation in train_loader:
#             batch_ecg_gpu, batch_activation_gpu = batch_ecg.to(device), batch_activation.to(device)
#             optimizer.zero_grad()
#             with torch.enable_grad():
#                 outputs = model(batch_ecg_gpu)
#                 loss = criterion(outputs, batch_activation_gpu)
#                 loss.backward()
#                 optimizer.step()
#
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for batch_ecg, batch_activation in val_loader:
#                 batch_ecg_gpu_one, batch_activation_gpu_one = batch_ecg.to(device), batch_activation.to(device)
#                 outputs = model(batch_ecg_gpu_one)
#                 val_loss += criterion(outputs, batch_activation_gpu_one).item()
#
#         #scheduler.step(val_loss)
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss / len(val_loader):.4f}')
#     end_time = time.time()
#
#     print('Training took: {} s'.format(end_time - start_time))
#
#
# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     sqmodel = SqueezeNet1D().to(device)
#     sqmodel = torch.compile(sqmodel)
#     print(summary(sqmodel), (32, 12, 500))
#
#     data_dirs = []
#     regex = r'data_hearts_dd_0p2*'
#     DIR = '../intracardiac_dataset/'
#     for x in os.listdir(DIR):
#         if re.match(regex, x):
#             data_dirs.append(DIR + x)
#     file_pairs = read_data_dirs(data_dirs)
#     print('Number of file pairs: {}'.format(len(file_pairs)))
#     # example of file pair
#     print("Example of file pair:")
#     print("{}\n{}".format(file_pairs[0][0], file_pairs[0][1]))
#     print("{}".format(file_pairs[1]))
#
#     train_file_pairs = file_pairs[0:15312]
#     val_file_pairs = file_pairs[15312:16117]
#
#     train_dataset = ECGDataset(train_file_pairs)
#     val_dataset = ECGDataset(val_file_pairs)
#     train_loader = DataLoader(train_dataset, batch_size=32, num_workers=20, shuffle=True, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, num_workers=20, pin_memory=True)
#
#     train(sqmodel, train_loader, val_loader, num_epochs=100, learning_rate=0.001)


def train(writer, device, model, train_loader, val_loader, num_epochs, learning_rate):
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=1.0)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_ecg, batch_activation in train_loader:
            batch_ecg, batch_activation = batch_ecg.to(device), batch_activation.to(device)
            optimizer.zero_grad()
            outputs = model(batch_ecg)
            loss = criterion(outputs, batch_activation)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_ecg, batch_activation in val_loader:
                batch_ecg, batch_activation = batch_ecg.to(device), batch_activation.to(device)
                outputs = model(batch_ecg)
                val_loss += criterion(outputs, batch_activation).item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)

        scheduler.step(avg_val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_LRonPlat.pth')

    print(f'Best validation loss: {best_val_loss:.4f}')
    return model
