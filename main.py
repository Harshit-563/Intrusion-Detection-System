import sys
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nsl_kdd import NUM_CLIENTS, Net, prepare_datasets, split_train_dataset

# ==============================
# Config
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 0.01
EPOCHS = 3

# ==============================
# Client ID
# ==============================
def get_client_id():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python main.py <client_id>")
    return int(sys.argv[1])


client_id = get_client_id()

# ==============================
# Data & Model
# ==============================
def load_data():
    train_dataset, test_dataset, _, label_encoder, _ = prepare_datasets()
    client_train = split_train_dataset(train_dataset, client_id, NUM_CLIENTS)

    train_loader = DataLoader(client_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(
        f"[Client {client_id}] Train: {len(client_train)} | "
        f"Test: {len(test_dataset)} | Classes: {len(label_encoder.classes_)}"
    )

    return train_loader, test_loader


model = Net().to(DEVICE)
trainloader, testloader = load_data()

# ==============================
# Training
# ==============================
def train(model, loader, epochs=EPOCHS):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Client {client_id}] Epoch {epoch+1} Loss: {total_loss:.4f}")


# ==============================
# Evaluation
# ==============================
def evaluate(model, loader):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            outputs = model(data)
            loss = criterion(outputs, target)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)

            correct += (preds == target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(loader)

    return avg_loss, accuracy


# ==============================
# Flower Helpers
# ==============================
def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


# ==============================
# Flower Client
# ==============================
class IDSClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_parameters(model)

    def fit(self, parameters, config):
        print(f"[Client {client_id}] Training started")

        set_parameters(model, parameters)
        train(model, trainloader)

        print(f"[Client {client_id}] Training finished")

        return get_parameters(model), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(model, parameters)

        loss, accuracy = evaluate(model, testloader)

        print(f"[Client {client_id}] Accuracy: {accuracy:.4f}")

        return loss, len(testloader.dataset), {"accuracy": accuracy}


# ==============================
# Start Client
# ==============================
if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=IDSClient(),
    )