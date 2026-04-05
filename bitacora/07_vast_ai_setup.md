# Vast.ai Setup & Workflow

## Account

- Edwin has an account on vast.ai
- API key stored in `.env` as `VASTAI_API_KEY` (org-level key)
- SSH key: `~/.ssh/id_ed25519_vastai` (personal, separate from other SSH keys)
- SSH key registered in vast.ai dashboard under personal view (not team)
- Credits: pre-paid, instances stop when credit hits $0

## Tested GPU

RTX A2000 — $0.044/hr, sufficient for our ~100K param model.
Also available: GTX 1080 ($0.045), RTX 3060 ($0.047).

## Workflow

```bash
# 1. Push code locally
git push origin feature/implement-v1

# 2. Rent instance from vast.ai web UI
#    Template: "PyTorch (vast)" 
#    Launch mode: SSH
#    Disk: 30GB

# 3. SSH in
ssh -p PORT root@HOST -i ~/.ssh/id_ed25519_vastai -L 8080:localhost:8080

# 4. Clone (first time) or pull (subsequent)
cd /workspace
git clone https://TOKEN@github.com/entropyx/josiah.git -b feature/implement-v1
# OR
cd /workspace/josiah && git remote set-url origin https://TOKEN@github.com/entropyx/josiah.git && git pull

# 5. Install
pip install -e ".[neural]"

# 6. Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 7. Train
python scripts/train_neural.py --decomposition --n-train 50000 --n-epochs 50 --fixed-channels 5 --random-eval 10

# 8. Copy results back (from LOCAL machine)
scp -P PORT -i ~/.ssh/id_ed25519_vastai -r root@HOST:/workspace/josiah/neural_output/results ./vast_results/

# 9. DESTROY (not stop — stop still charges storage at $0.11/day)
# Use dashboard trash icon or: python scripts/vast_train.py destroy
```

## GitHub Token

Private repo requires fine-grained personal access token:
- github.com → Settings → Developer Settings → Fine-grained tokens
- Resource owner: entropyx (org)
- Select only josiah repo
- Permissions: Contents (read-only) + Metadata (read-only)
- Expiration: 1 day recommended

## Billing Safety

- Per-second billing (not rounded to hours)
- On-demand = fixed rate, no surprises
- DESTROY to stop ALL charges (stop still bills storage)
- ~$0.04/hr → 1hr training = $0.04
- Pre-paid credits only

## Script Commands

```bash
python scripts/vast_train.py search                    # browse GPUs (free)
python scripts/vast_train.py search --min-ram 4        # cheapest GPUs
python scripts/vast_train.py status                    # running instances
python scripts/vast_train.py destroy                   # kill instance
python scripts/vast_train.py destroy --id 12345678     # kill by ID
```
