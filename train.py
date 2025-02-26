import argparse
import yaml
import os
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from qwen_vl_utils import process_vision_info
import logging
import random

import utils
from dataset.digirl_dataset import ReplayBuffer, DummyDataset
from models.autoui_model import ImageFeatureExtractor, AutoUIAgent
from eval_tools.metrix import compute_matrix
from data_preprocess.utils import update_trajectory


class DigiRLTrainer:
    def __init__(self,
        agent,
        accelerator,
        tokenizer,
        config
    ):
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr=float(config["lm_lr"]))

        self.criterion = torch.nn.CrossEntropyLoss()
        self.grad_accum_steps = config["grad_accum_steps"]
        self.gamma = config["gamma"]

        self.epochs = config["epochs"]

        self.step = 0
        self.tau = config["tau"]
        self.max_grad_norm = config["max_grad_norm"]
        self.accelerator = accelerator
        self.softmax = torch.nn.Softmax(dim = -1)
        self.image_process = ImageFeatureExtractor()


    def prepare(self):
        self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)


    def actor_loss(
        self,
        critic_images,
        critic_inputs,
        policy_inputs,
        policy_outputs,
        policy_images,
        validation=False,
        **kwargs
    ):
        dtype = self.accelerator.unwrap_model(self.agent.model).dtype
        device = self.accelerator.unwrap_model(self.agent.model).device
        
        messages = []
        for critic_input, critic_image in zip(critic_inputs, critic_images):
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "text", "text": critic_input},
                    {"type": "image", "image": critic_image, "max_pixels": 56000}
                ]
            }])

        texts = self.agent.critic_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        vision_inputs = []
        for message in messages:
            vision_input, _ = process_vision_info(message)
            vision_inputs.append(vision_input)

        inputs = self.agent.critic_processor(
            text=texts,
            images=vision_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        generated_ids = self.agent.critic.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        q_values = self.agent.critic_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        q_values = [int(val) for val in q_values]
        q_values = torch.tensor(q_values, dtype=dtype, requires_grad=True).to(device)

        q_values = q_values / 2

        policy_image_features = []
        for policy_image in policy_images:
            policy_image_feature = self.image_process.to_feat(image_path=policy_image).to(device, dtype=dtype)
            policy_image_features.append(policy_image_feature)
        policy_image_features = torch.stack(policy_image_features)

        log_prob = self.agent.get_log_prob(policy_inputs, policy_image_features, policy_outputs).sum(dim=1).flatten()

        pg_loss = - torch.mean(log_prob * q_values)

        if not validation:
            self.accelerator.backward(pg_loss)

        return pg_loss.detach().cpu().item(), torch.mean(q_values).detach().cpu().item()


    def update_policy(self, buffer, is_validation, batch_size):
        logs = []

        self.step += 1
        data = [buffer.sample(1) for _ in range(self.grad_accum_steps * batch_size)]

        for d in data:
            for k, v in d.items():
                d[k] = v[0]

        keys = ["ep_id", "step_id", "policy_inputs", "policy_outputs", "policy_images", "critic_inputs", "critic_images"]
        dataloader = self.accelerator.prepare(DataLoader(DummyDataset(data, keys), batch_size=batch_size, shuffle=False))

        self.lm_optimizer.zero_grad()
        losses, q_values = [], []
        if is_validation:
            with torch.no_grad():
                for batch in dataloader:
                    loss, q_value = self.actor_loss(**batch, validation=True)
                    losses.append(loss)
                    q_values.append(q_value)
            logging.info(f"[val] step: {self.step}\tloss: {sum(losses) / len(losses):.2f}\tQ-values: {sum(q_values) / len(q_values):.4f}")
            logs.append({"step": self.step, "val loss": sum(losses) / len(losses), "val Q value": sum(q_values) / len(q_values)})
        else:
            for batch in dataloader:
                loss, q_value = self.actor_loss(**batch)
                losses.append(loss)
                q_values.append(q_value)
            logging.info(f"step: {self.step}\tloss: {sum(losses) / len(losses):.2f}\tQ-values: {sum(q_values) / len(q_values):.4f}")
            logs.append({"step": self.step, "train loss": sum(losses) / len(losses), "train Q value": sum(q_values) / len(q_values)})

            self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.lm_optimizer.step()

        return logs


    def infer(self, data, batch_size):
        dtype = self.accelerator.unwrap_model(self.agent.model).dtype
        device = self.accelerator.unwrap_model(self.agent.model).device
        keys = ["ep_id", "step_id", "policy_input", "policy_output", "policy_image"]
        dataloader = DataLoader(DummyDataset(data, keys), batch_size=batch_size, shuffle=False)
        dataloader = self.accelerator.prepare(dataloader)

        results = []
        for batch in dataloader:
            ep_ids, step_ids = batch["ep_id"], batch["step_id"]
            texts, groundtruths, image_paths = batch["policy_input"], batch["policy_output"], batch["policy_image"]
            image_features = []
            for image_path in image_paths:
                image_feature = self.image_process.to_feat(image_path=image_path).to(device, dtype=dtype)
                image_features.append(image_feature)
            image_features = torch.stack(image_features)

            outputs = self.agent.get_action(texts, image_features)

            for (output, groundtruth, ep_id, step_id) in zip(outputs, groundtruths, ep_ids, step_ids):
                results.append({"output": output, "groundtruth": groundtruth, "ep_id": ep_id, "step_id": step_id.item()})

        return results


    def save(self, path):
        self.accelerator.save_state(path, safe_serialization=False)


    def load(self, path):
        self.accelerator.load_state(path)


def train(
    agent,
    accelerator,
    config
):
    trainer = DigiRLTrainer(
        agent=agent,
        accelerator=accelerator,
        tokenizer=agent.tokenizer,
        config=config
    )

    batch_size = config["batch_size"]
    all_trajectories = utils.read_jsonl(config["data_path"])

    agent.prepare()
    trainer.prepare()

    print(f"### all trajectories: {len(all_trajectories)}")

    logs = []
    train_trajectories = all_trajectories[:int(len(all_trajectories) * 0.95)]
    val_trajectories = all_trajectories[int(len(all_trajectories) * 0.95):]
    random.shuffle(train_trajectories)
    sample_num = config["batch_size"] * config["grad_accum_steps"]
    for epoch in range(config["epochs"]):
        print(f"### epoch {epoch}")
        for train_step in range(len(train_trajectories) // sample_num):
            sample_trajectories = train_trajectories[train_step * sample_num: (train_step + 1) * sample_num]

            results = trainer.infer(sample_trajectories, batch_size)
            sample_trajectories = update_trajectory(sample_trajectories, results)
            
            replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(sample_trajectories))

            for d in sample_trajectories:
                replay_buffer.insert(**d)

            logs.extend(trainer.update_policy(replay_buffer, is_validation=False, batch_size=batch_size))

        results = trainer.infer(val_trajectories, batch_size)
        val_trajectories = update_trajectory(val_trajectories, results)
        validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(val_trajectories))
        for d in val_trajectories:
            validation_buffer.insert(**d)
        logs.extend(trainer.update_policy(validation_buffer, is_validation=True, batch_size=batch_size))

        if accelerator.is_main_process:
            save_path = config["save_path"]
            print("### saving")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if not os.path.exists(os.path.join(save_path, f"epoch_{epoch}")):
                os.mkdir(os.path.join(save_path, f"epoch_{epoch}"))
            trainer.save(os.path.join(save_path, f"epoch_{epoch}"))
            utils.write_jsonl(logs, os.path.join(save_path, f"epoch_{epoch}", "train_log.jsonl"))
            utils.plot_loss(os.path.join(save_path, f"epoch_{epoch}"), keys=["train loss", "train Q value", "val loss", "val Q value"])


def evaluation(
    agent,
    accelerator,
    config
):
    
    result_dir = f"checkpoints/{config['test_task']}_result"
    result_wpath = os.path.join(result_dir, f"{config['test_task']}_{config['model_name']}_results.jsonl")

    print(f"### result path: {result_wpath}")
    
    anns = utils.read_jsonl(config['eval_data'])
    
    position_dict = {}
    for ann in anns:
        position_dict[f"{ann['ep_id']}_{ann['step_id']}"] = ann["position"]

    if not os.path.exists(result_wpath):
        trainer = DigiRLTrainer(
            agent=agent,
            accelerator=accelerator,
            tokenizer=agent.tokenizer,
            config=config,
        )

        results = trainer.infer(anns, config["batch_size"])
        utils.write_jsonl(results, result_wpath)
    else:
        results = utils.read_jsonl(result_wpath)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    for file in os.listdir(result_dir):
        result_wpath = os.path.join(result_dir, file)
        results = utils.read_jsonl(result_wpath)
        print(f"================{result_wpath.split('/')[2]}================")
        compute_matrix(results, position_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--eval', action='store_true')  
    args = parser.parse_args()

    if args.task == "webshop":
        config = "configs/policy/rl_qwen2.5.yaml"
    else:
        config = "configs/policy/rl_qwen2.5.yaml"

    with open(config, 'r') as file:
        config = yaml.safe_load(file)
        
    print(f"### config:")
    for k, v in config.items():
        print(f"\t{k}: {v}")
    
    accelerator = Accelerator()

    print("### load SeeclickAgent")

    agent = AutoUIAgent(
        device=accelerator.device,
        accelerator=accelerator,
        temperature=config["temperature"],
        do_sample=config["do_sample"],
        policy_lm=config["policy_lm"],
        critic_lm=config["critic_lm"],
        max_new_tokens=config["max_new_tokens"]
    )

    if args.eval:
        evaluation(agent=agent, accelerator=accelerator, config=config)
    else:
        train(agent=agent, accelerator=accelerator, config=config)


