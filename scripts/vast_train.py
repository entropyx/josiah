"""Launch and manage GPU training on vast.ai.

Usage:
    # Set API key (one-time)
    export VASTAI_API_KEY=your_key

    # Launch training run
    python scripts/vast_train.py launch

    # Check status
    python scripts/vast_train.py status

    # View logs
    python scripts/vast_train.py logs

    # Pull results back
    python scripts/vast_train.py pull

    # Destroy instance (stop billing)
    python scripts/vast_train.py destroy
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/entropyx/josiah.git"
BRANCH = "feature/implement-v1"
STATE_FILE = Path("neural_output/.vast_instance.json")

# Default training command
TRAIN_CMD = (
    "python scripts/train_neural.py --decomposition "
    "--n-train 50000 --n-epochs 50 --fixed-channels 5 "
    "--scenario realistic_brand"
)


def get_client():
    from dotenv import load_dotenv
    from vastai_sdk import VastAI

    load_dotenv()
    api_key = os.environ.get("VASTAI_API_KEY")
    if not api_key:
        print("VASTAI_API_KEY not found in .env or environment.")
        sys.exit(1)
    return VastAI(api_key=api_key)


def save_state(instance_id: int):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"instance_id": instance_id}))


def load_state() -> int | None:
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
        return data.get("instance_id")
    return None


def cmd_launch(args):
    """Launch a GPU instance and start training."""
    client = get_client()

    gpu = args.gpu
    print(f"Launching {gpu} instance...")

    # Setup script: clone repo, install, run training
    setup_script = f"""#!/bin/bash
set -e
cd /root
echo "=== Cloning repo ==="
git clone -b {BRANCH} {REPO_URL} josiah
cd josiah
echo "=== Installing dependencies ==="
pip install -e ".[neural]" -q
echo "=== Starting training ==="
{args.cmd}
echo "=== Training complete ==="
"""

    result = client.launch_instance(
        gpu_name=gpu,
        num_gpus="1",
        image="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
        disk=args.disk,
        onstart_cmd=setup_script,
        ssh=True,
        label="demantiq-training",
    )
    print(result)

    # Try to parse instance ID from result
    try:
        # Result format varies — try to extract ID
        if isinstance(result, str):
            data = json.loads(result)
            if "new_contract" in data:
                instance_id = data["new_contract"]
                save_state(instance_id)
                print(f"\nInstance ID: {instance_id}")
                print(f"Saved to {STATE_FILE}")
                print(f"\nTraining will start automatically.")
                print(f"Check progress: python scripts/vast_train.py logs")
                print(f"When done:      python scripts/vast_train.py pull")
                print(f"Stop billing:   python scripts/vast_train.py destroy")
    except (json.JSONDecodeError, KeyError):
        print("\nCouldn't parse instance ID from response.")
        print("Run 'python scripts/vast_train.py status' to find your instance.")


def cmd_status(args):
    """Show running instances."""
    client = get_client()
    result = client.show_instances()
    print(result)


def cmd_logs(args):
    """View instance logs."""
    client = get_client()
    instance_id = args.id or load_state()
    if not instance_id:
        print("No instance ID. Pass --id or run 'launch' first.")
        sys.exit(1)
    result = client.logs(INSTANCE_ID=instance_id, tail=args.tail)
    print(result)


def cmd_pull(args):
    """Pull training results from instance."""
    instance_id = args.id or load_state()
    if not instance_id:
        print("No instance ID. Pass --id or run 'launch' first.")
        sys.exit(1)

    client = get_client()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pulling results to {out_dir}/...")
    # Copy results and plots
    for remote_dir in ["neural_output/results", "neural_output/plots", "neural_output/decomp_model"]:
        try:
            result = client.copy(
                src=f"{instance_id}:/root/josiah/{remote_dir}/",
                dst=str(out_dir / remote_dir.split("/")[-1]) + "/",
            )
            print(f"  {remote_dir}: {result}")
        except Exception as e:
            print(f"  {remote_dir}: {e}")

    print(f"\nResults saved to {out_dir}/")


def cmd_destroy(args):
    """Destroy instance (stops billing)."""
    client = get_client()
    instance_id = args.id or load_state()
    if not instance_id:
        print("No instance ID. Pass --id or run 'launch' first.")
        sys.exit(1)

    result = client.destroy_instance(id=instance_id)
    print(result)

    if STATE_FILE.exists():
        STATE_FILE.unlink()
    print("Instance destroyed. Billing stopped.")


def cmd_search(args):
    """Browse available GPUs (free — no charges)."""
    client = get_client()
    min_ram = args.min_ram
    query = f"gpu_ram >= {min_ram} num_gpus == 1 rentable == true"
    if args.gpu:
        query += f" gpu_name={args.gpu}"
    result = client.search_offers(query=query, order="dph_total", limit=args.limit)

    if not result:
        print("No offers found.")
        return

    print(f"\n{'ID':<12} {'GPU':<16} {'VRAM':>6} {'$/hr':>7} {'RAM':>7} {'Disk':>6} {'Location':<20} {'Reliability':>11}")
    print("-" * 95)
    for offer in result:
        print(
            f"{offer['id']:<12} "
            f"{offer.get('gpu_name', '?'):<16} "
            f"{offer.get('gpu_ram', 0) // 1024:>4}GB "
            f"${offer.get('dph_total', 0):>6.3f} "
            f"{offer.get('cpu_ram', 0) // 1024:>5}GB "
            f"{offer.get('disk_space', 0):>5.0f}G "
            f"{offer.get('geolocation', '?'):<20} "
            f"{offer.get('reliability', 0):>10.1%}"
        )


def cmd_ssh(args):
    """Print SSH command to connect."""
    client = get_client()
    result = client.show_instances()
    print("Running instances:")
    print(result)
    print("\nTo SSH manually, use the connection info above.")


def main():
    parser = argparse.ArgumentParser(description="Manage vast.ai GPU training")
    sub = parser.add_subparsers(dest="command")

    # Launch
    p_launch = sub.add_parser("launch", help="Launch GPU instance and start training")
    p_launch.add_argument("--gpu", default="RTX_3060", help="GPU type (default: RTX_3060)")
    p_launch.add_argument("--disk", type=float, default=30.0, help="Disk GB (default: 30)")
    p_launch.add_argument("--cmd", default=TRAIN_CMD, help="Training command to run")
    p_launch.set_defaults(func=cmd_launch)

    # Status
    p_status = sub.add_parser("status", help="Show running instances")
    p_status.set_defaults(func=cmd_status)

    # Logs
    p_logs = sub.add_parser("logs", help="View instance logs")
    p_logs.add_argument("--id", type=int, help="Instance ID (auto-detected if launched via this script)")
    p_logs.add_argument("--tail", default="100", help="Number of log lines (default: 100)")
    p_logs.set_defaults(func=cmd_logs)

    # Pull
    p_pull = sub.add_parser("pull", help="Pull training results")
    p_pull.add_argument("--id", type=int, help="Instance ID")
    p_pull.add_argument("--output", default="vast_results", help="Local output directory")
    p_pull.set_defaults(func=cmd_pull)

    # Destroy
    p_destroy = sub.add_parser("destroy", help="Destroy instance (stop billing)")
    p_destroy.add_argument("--id", type=int, help="Instance ID")
    p_destroy.set_defaults(func=cmd_destroy)

    # SSH
    p_ssh = sub.add_parser("ssh", help="Show SSH connection info")
    p_ssh.set_defaults(func=cmd_ssh)

    # Search
    p_search = sub.add_parser("search", help="Browse cheapest available GPUs (free)")
    p_search.add_argument("--gpu", default=None, help="Filter by GPU name (e.g. RTX_3060)")
    p_search.add_argument("--min-ram", type=int, default=4, help="Minimum GPU RAM in GB (default: 4)")
    p_search.add_argument("--limit", type=int, default=10, help="Number of results")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
