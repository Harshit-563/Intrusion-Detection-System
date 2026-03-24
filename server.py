from collections import OrderedDict
import json
from pathlib import Path
import pickle

import flwr as fl
import torch

from nsl_kdd import Net, prepare_datasets

# ==============================
# Config
# ==============================
BASE_DIR = Path(__file__).resolve().parent
NUM_ROUNDS = 3


# ==============================
# Custom Strategy
# ==============================
class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.accuracy_history = []

    # --------------------------
    # Save Accuracy History
    # --------------------------
    def save_metrics(self):
        file_path = BASE_DIR / "accuracy_history.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(self.accuracy_history, f, indent=4)
        print(f"[Server] Accuracy history saved -> {file_path}")

    # --------------------------
    # Aggregate Evaluation
    # --------------------------
    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        accuracies = [
            res.metrics["accuracy"]
            for _, res in results
            if "accuracy" in res.metrics
        ]

        if accuracies:
            avg_acc = sum(accuracies) / len(accuracies)
            self.accuracy_history.append(avg_acc)
            print(f"\n[Server] Round {server_round} Global Accuracy: {avg_acc:.4f}\n")

        return super().aggregate_evaluate(server_round, results, failures)

    # --------------------------
    # Aggregate Training
    # --------------------------
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            self._save_model(aggregated_parameters, server_round)

        return aggregated_parameters, metrics

    # --------------------------
    # Save Model
    # --------------------------
    def _save_model(self, parameters, round_num):
        print(f"[Server] Saving global model (round {round_num})...")

        ndarrays = fl.common.parameters_to_ndarrays(parameters)
        model = Net()

        state_dict = OrderedDict(
            (k, torch.tensor(v))
            for k, v in zip(model.state_dict().keys(), ndarrays)
        )
        model.load_state_dict(state_dict)

        model_dir = BASE_DIR / "models"
        model_dir.mkdir(exist_ok=True)

        round_path = model_dir / f"global_model_round_{round_num}.pth"
        final_path = model_dir / "global_model_final.pth"

        torch.save(model.state_dict(), round_path)
        torch.save(model.state_dict(), final_path)

        print(f"[Server] Model saved -> {round_path}")
        print(f"[Server] Updated latest model -> {final_path}")


# ==============================
# Save Preprocessing Objects
# ==============================
def save_preprocessing():
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)

    scaler_path = model_dir / "scaler.pkl"
    encoder_path = model_dir / "label_encoder.pkl"

    _, _, _, label_encoder, scaler = prepare_datasets()

    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)

    with encoder_path.open("wb") as f:
        pickle.dump(label_encoder, f)

    print(f"[Saved] Scaler -> {scaler_path}")
    print(f"[Saved] Label Encoder -> {encoder_path}")


# ==============================
# Main
# ==============================
def main():
    save_preprocessing()

    strategy = CustomStrategy()
    print("[Server] Starting Flower server...")

    fl.server.start_server(
        server_address="127.0.0.1:8081",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    strategy.save_metrics()
    print("[Server] Training complete.")


if __name__ == "__main__":
    main()
