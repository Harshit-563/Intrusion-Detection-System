from pathlib import Path
import subprocess
import sys
import signal

# ==============================
# Config
# ==============================
NUM_CLIENTS = 3
BASE_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = BASE_DIR / "main.py"


# ==============================
# Launch Clients
# ==============================
def start_clients():
    processes = []

    for client_id in range(NUM_CLIENTS):
        print(f"[Launcher] Starting client {client_id}...")

        process = subprocess.Popen(
            [sys.executable, str(MAIN_SCRIPT), str(client_id)],
            cwd=BASE_DIR,
        )

        processes.append(process)

    return processes


# ==============================
# Wait & Handle Exit
# ==============================
def wait_for_clients(processes):
    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        print("\n[Launcher] Stopping all clients...")
        for process in processes:
            process.send_signal(signal.SIGINT)


# ==============================
# Main
# ==============================
def main():
    processes = start_clients()
    wait_for_clients(processes)
    print("[Launcher] All clients finished.")


if __name__ == "__main__":
    main()