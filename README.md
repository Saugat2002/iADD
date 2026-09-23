# iADD: Improving Alignment and Diversity in Diffusion Policy Optimization



## Environment Setup

### Install Miniforge

```bash
INSTALLER="Miniforge3-24.11.3-0-Linux-x86_64.sh"
wget -q --show-progress \
    "https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/${INSTALLER}" \
    -O "${INSTALLER}"

bash "${INSTALLER}" -b -p "${CONDA_DIR}"
```

### Create Conda Environment

Python version: 3.10.19

```bash
CONDA_ENV_NAME=iadd
conda create -n "${CONDA_ENV_NAME}" python=3.10.19 -y
conda activate "${CONDA_ENV_NAME}"
```

### Install Dependencies

```bash
pip install uv==0.9.26
uv pip install -r requirements.txt

# Fix setuptools version
pip uninstall setuptools -y
pip install setuptools==80.9.0
```

---

## Usage

### Training

```bash
python3 ./scripts/training/train_pipeline.py \
    exp_name="${run_name}" \
    train.incremental_training=true \
    sample.fk=true \
    sample.num_particles=4 \
    sample.only_best_fk=true \
    sample.potential_type=max \
    sample.fk_lambda=2.0 \
    sample.resample_frequency=4 \
    sample.resampling_t_start=8 \
    sample.resampling_t_end=16 \
    sample.brach_at_before_fk=5 \
    seed=42 \
    sample.batch_size=12 \
    train.batch_size=16 \
    sample.num_batches_per_epoch=16 \
    train.learning_rate=3e-4 \
    train.max_grad_norm=0.005 \
    train.incremental_timesteps=[4,8,12,16] \
    train.num_stages_per_increment=10 \
    train.use_kl_div_loss=false
```

### Inference

```bash
run_name="test"
stage_number=1

python3 ./scripts/inference/inference_lora_clip_reward.py \
    --checkpoint_path /path/to/model/lora/${run_name}/stage${stage_number}/checkpoints/checkpoint_1/ \
    --output_dir ./outputs/${run_name}/stage${stage_number} \
    --num_images 1080 \
    --batch_size 32
```


---

**Note:** Additional documentation and instructions will be provided in the public release.