import hydra

from rllm.agents.system_prompts import DATASHEET_AGENT_SYSTEM_PROMPT
from rllm.agents.tool_agent import ToolAgent
from rllm.data import DatasetRegistry
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import datasheet_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer

from .local_retrieval_tools import (
    LocalDatasheetFigureRetriever,
    LocalDatasheetTableRetriever,
    LocalDatasheetTextRetriever,
)
    

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("datasheet_agent", "train")
    val_dataset = DatasetRegistry.load_dataset("datasheet_agent", "test")

    tool_map = {"local_figure_search": LocalDatasheetFigureRetriever, "local_text_search": LocalDatasheetTextRetriever, "local_table_search": LocalDatasheetTableRetriever}

    env_args = {
        "max_steps": 20,
        "tool_map": tool_map,
        "reward_fn": datasheet_reward_fn,
    }

    agent_args = {"system_prompt": DATASHEET_AGENT_SYSTEM_PROMPT, "tool_map": tool_map, "parser_name": "qwen"}

    # Use the registry-based approach (comment out the other approach)
    trainer = AgentTrainer(
        agent_class=ToolAgent,
        env_class=ToolEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_args=agent_args,
        env_args=env_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
