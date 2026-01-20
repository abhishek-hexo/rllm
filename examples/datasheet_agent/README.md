# Datasheet Agent

Train and evaluate a retrieval-augmented language agent for answering questions about electronic component datasheets using reinforcement learning.

## Overview

This project trains a Qwen3-4B model using RL (RLOO algorithm) to answer complex questions about datasheets by learning to use three retrieval tools:
- **Text Search**: Dense retrieval over text chunks from datasheet PDFs
- **Table Search**: Dense retrieval over extracted tables
- **Figure Search**: Vision-augmented retrieval that renders PDF pages and uses a vision LLM to extract relevant information

The agent learns to formulate queries, select appropriate tools, and synthesize information from multiple sources to answer questions about block diagrams, pin maps, and register maps.

## Dataset

<!-- TODO: Add HuggingFace dataset link -->
**Dataset**: [To be uploaded]

The dataset contains QA pairs extracted from electronic component datasheets:
- **Block Diagram Questions**: Questions about blocks, pins, and their descriptions in block diagrams
- **Pin Map Questions**: Questions about pin configurations, types, and descriptions
- **Register Map Questions**: Questions about register configurations

**Dataset Format**:
```json
{
  "question": "List all pins in the power management block",
  "ground_truth": {"pins": [...], "blocks": [...], "description": "..."},
  "datasheet_id": "component_xyz",
  "data_source": "block_diagram"
}
```

## Training Infrastructure

### Hardware Requirements
- **GPUs**: 4x NVIDIA GPUs (tested on A100/H100)
- **Memory**: ~70% GPU memory utilization per GPU
- **Storage**: Sufficient space for datasheet indices and model checkpoints

### Software Stack
- **Framework**: rLLM with verl backend
- **Inference**: vLLM with tensor parallelism (TP=4)
- **Retrieval**: FAISS indices with E5-base-v2 embeddings
- **Vision**: Gemma-27B-IT for figure extraction (via local endpoint or OpenRouter)

### Cluster Configuration
The training script is configured for a SLURM/HPC environment with Lustre filesystem. Key paths in `train_datasheet_search_agent.sh`:
```bash
RLLM_DIR=/lustre/scratch/users/<username>/datasheet_RL_v2
```

Modify this path to match your storage location.

## Replication Steps

### 1. Environment Setup

```bash
# Clone rLLM with submodules
git clone --recurse-submodules https://github.com/rllm-org/rllm.git
cd rllm

# Create conda environment
conda create -n rllm python=3.10 -y
conda activate rllm

# Install verl and dependencies
bash scripts/install_verl.sh

# Install rLLM
pip install -e .

# Additional dependencies for retrieval servers
pip install faiss-cpu flask sentence-transformers PyMuPDF python-dotenv
```

### 2. Data Preparation

#### Download Pre-built Indices
<!-- TODO: Add download instructions for indices -->
Download the pre-built FAISS indices and place them in your data directory:
```
<data_dir>/
├── <datasheet_id_1>/
│   ├── text_index.faiss
│   ├── text_index.metadata.json
│   ├── table_index.faiss
│   ├── table_index.metadata.json
│   ├── figure_index.faiss
│   ├── figure_index.metadata.json
│   └── datasheet.pdf
├── <datasheet_id_2>/
│   └── ...
```

#### Prepare Dataset Splits
Create train/test split files:
```bash
# Create train.txt and test.txt with paths to datasheet directories
# Each line should be the full path to a datasheet folder
```

#### Register Dataset
```bash
cd examples/datasheet_agent
python prepare_dataset.py
```

### 3. Launch Retrieval Servers

Open three separate terminals and launch each retrieval server:

**Terminal 1 - Text Retrieval Server (port 8001)**:
```bash
export TEXT_RETRIEVAL_SERVER_URL="http://127.0.0.1:8001"
python -m examples.datasheet_agent.text_retrieval.server \
    --data_dir /path/to/your/datasheet_data \
    --port 8001
```

**Terminal 2 - Table Retrieval Server (port 8002)**:
```bash
export TABLE_RETRIEVAL_SERVER_URL="http://127.0.0.1:8002"
python -m examples.datasheet_agent.table_retriever.server \
    --data_dir /path/to/your/datasheet_data \
    --port 8002
```

**Terminal 3 - Figure Retrieval Server (port 8003)**:
```bash
# Set up vision LLM endpoint (either local vLLM or OpenRouter)
export OPENROUTER_API_KEY="your_api_key"  # If using OpenRouter
export FIGURE_RETRIEVAL_SERVER_URL="http://127.0.0.1:8003"

python -m examples.datasheet_agent.figure_retrieval.server \
    --data_dir /path/to/your/datasheet_data \
    --port 8003
```

For the figure retrieval server, you also need a vision-capable LLM. Either:
- **Local**: Start a vLLM server with Gemma-27B-IT on port 8005
- **API**: Use OpenRouter API (set `OPENROUTER_API_KEY`)

### 4. Download Base Model

```bash
# Download Qwen3-4B to your model directory
huggingface-cli download Qwen/Qwen3-4B --local-dir /path/to/hf_ckpts/Qwen3-4B
```

### 5. Run Training

```bash
cd rllm

# Set environment variables for retrieval servers
export TEXT_RETRIEVAL_SERVER_URL="http://127.0.0.1:8001"
export TABLE_RETRIEVAL_SERVER_URL="http://127.0.0.1:8002"
export FIGURE_RETRIEVAL_SERVER_URL="http://127.0.0.1:8003"

# Edit train_datasheet_search_agent.sh to update paths:
# - RLLM_DIR: Output directory for checkpoints and logs
# - actor_rollout_ref.model.path: Path to Qwen3-4B weights

# Launch training
bash examples/datasheet_agent/train_datasheet_search_agent.sh
```

### Training Configuration

Key hyperparameters in `train_datasheet_search_agent.sh`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `algorithm.adv_estimator` | `rloo` | REINFORCE Leave-One-Out advantage estimation |
| `data.train_batch_size` | 16 | Batch size for training |
| `data.max_prompt_length` | 2048 | Maximum prompt length |
| `data.max_response_length` | 2048 | Maximum response length |
| `actor_rollout_ref.actor.optim.lr` | 1e-4 | Learning rate |
| `actor_rollout_ref.rollout.n` | 8 | Number of rollouts per prompt |
| `actor_rollout_ref.rollout.temperature` | 0.7 | Sampling temperature |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | 4 | Tensor parallelism |
| `rllm.agent.max_steps` | 10 | Maximum agent steps per episode |
| `trainer.total_epochs` | 100 | Total training epochs |
| `trainer.save_freq` | 40 | Checkpoint save frequency |
| `trainer.test_freq` | 10 | Validation frequency |

## Inference

### Option 1: Using vLLM Server

Start a vLLM server with your trained checkpoint:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/checkpoint \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 \
    --tensor-parallel-size 4
```

### Option 2: Using rLLM AgentExecutionEngine

```python
import asyncio
from transformers import AutoTokenizer
from rllm.engine import AgentExecutionEngine
from rllm.agents.tool_agent import ToolAgent
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.agents.system_prompts import DATASHEET_AGENT_SYSTEM_PROMPT
from rllm.rewards.reward_fn import datasheet_reward_fn
from examples.datasheet_agent.local_retrieval_tools import (
    LocalDatasheetFigureRetriever,
    LocalDatasheetTableRetriever,
    LocalDatasheetTextRetriever,
)

# Setup
model_path = "/path/to/checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_path)

tool_map = {
    "local_figure_search": LocalDatasheetFigureRetriever,
    "local_text_search": LocalDatasheetTextRetriever,
    "local_table_search": LocalDatasheetTableRetriever,
}

engine = AgentExecutionEngine(
    agent_class=ToolAgent,
    agent_args={
        "system_prompt": DATASHEET_AGENT_SYSTEM_PROMPT,
        "tool_map": tool_map,
        "parser_name": "qwen"
    },
    env_class=ToolEnvironment,
    env_args={
        "max_steps": 10,
        "tool_map": tool_map,
        "reward_fn": datasheet_reward_fn,
    },
    engine_name="openai",
    rollout_engine_args={"base_url": "http://localhost:30000/v1"},
    tokenizer=tokenizer,
    sampling_params={"temperature": 0.7, "top_p": 0.8, "model": model_path},
    max_response_length=2048,
    max_prompt_length=2048,
    n_parallel_agents=8,
)

# Run inference
tasks = [
    {
        "question": "What are the pins in the power management block?",
        "datasheet_id": "component_xyz",
        "data_source": "block_diagram",
    }
]

results = asyncio.run(engine.execute_tasks(tasks))
for result in results:
    print(result.trajectory)
```

## Tensorboard Logs

<!-- TODO: Add tensorboard logs path or screenshots -->
Tensorboard logs are saved to:
```
<RLLM_DIR>/datasheet-agent/7b-loop-drgrpo-datasheet_agent/
```

To view logs:
```bash
tensorboard --logdir /path/to/RLLM_DIR/datasheet-agent/
```

## Project Structure

```
examples/datasheet_agent/
├── README.md                      # This file
├── train_search_agent.py          # Main training script
├── train_datasheet_search_agent.sh # Training launcher with hyperparameters
├── prepare_dataset.py             # Dataset preparation and registration
├── local_retrieval_tools.py       # Tool implementations for retrieval
├── text_retrieval/
│   └── server.py                  # Text chunk retrieval server
├── table_retriever/
│   └── server.py                  # Table retrieval server
└── figure_retrieval/
    └── server.py                  # Vision-augmented figure retrieval server
```

## Troubleshooting

### Retrieval Server Connection Errors
Ensure all three retrieval servers are running and accessible:
```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8003/health
```

### CUDA Out of Memory
- Reduce `data.train_batch_size`
- Enable gradient checkpointing (already enabled)
- Reduce `actor_rollout_ref.rollout.gpu_memory_utilization`

### Vision LLM Errors
For the figure retrieval server, ensure the vision LLM endpoint is accessible:
```bash
curl http://localhost:8005/v1/models
```

## Citation

If you use this work, please cite:

```bibtex
@misc{rllm2025,
  title={rLLM: A Framework for Post-Training Language Agents},
  author={rLLM Team},
  year={2025},
  howpublished={\url{https://github.com/rllm-org/rllm}}
}
```
