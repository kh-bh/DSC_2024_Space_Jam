import torch
import torch.optim as optim
from datetime import datetime


def train(writer, device, model, train_loader, val_loader, num_epochs, learning_rate):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

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
            model_path = 'model_{}_{}'.format(timestamp, num_epochs)
            torch.save(model.state_dict(), model_path)

    print(f'Best validation loss: {best_val_loss:.4f}')
    return model
