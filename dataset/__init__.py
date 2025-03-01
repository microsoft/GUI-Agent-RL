import utils
from torch.utils.data import Dataset
import numpy as np


class DummyDataset(Dataset):
    def __init__(self, anns, keys):
        self.anns = []
        for ann in anns:
            self.anns.append({k:v for k, v in ann.items() if k in keys})

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        return self.anns[idx]


class ReplayBuffer:
    def __init__(self, batch_size=2, capacity=10000):
        self.max_size = capacity
        self.current_size = 0   
        self.batch_size = batch_size

        self.critic_images = None
        self.critic_inputs = None
        self.policy_outputs = None
        self.policy_inputs = None
        self.policy_images = None
        self.action_lists = None
        self.tasks = None
        self.step_ids = None

    def sample(self, batch_size=None):
        rand_indices = np.random.randint(0, self.current_size, size=batch_size) % self.max_size
        return {
            "critic_images": self.critic_images[rand_indices],
            "critic_inputs": self.critic_inputs[rand_indices],
            "policy_outputs": self.policy_outputs[rand_indices],
            "policy_inputs": self.policy_inputs[rand_indices],
            "policy_images": self.policy_images[rand_indices],
            "action_lists": self.action_lists[rand_indices],
            "tasks": self.tasks[rand_indices],
            "step_ids": self.step_ids[rand_indices]
        }

    def __len__(self):
        return self.current_size

    def insert(
        self,
        policy_output,
        policy_input,
        policy_image,
        action_list,
        task,
        step_id, 
        critic_image="",
        critic_input="",
        **kwargs
    ):
        if self.critic_images is None:
            self.critic_images = np.array([''] * self.max_size, dtype="object")
            self.critic_inputs = np.array([''] * self.max_size, dtype="object")
            self.policy_outputs = np.array([''] * self.max_size, dtype="object")
            self.policy_inputs = np.array([''] * self.max_size, dtype="object")
            self.policy_images = np.array([''] * self.max_size, dtype="object")
            self.action_lists = np.array([''] * self.max_size, dtype="object")
            self.tasks = np.array([''] * self.max_size, dtype="object")
            self.step_ids = np.array([''] * self.max_size, dtype="object")

        self.critic_images[self.current_size % self.max_size] = critic_image
        self.critic_inputs[self.current_size % self.max_size] = critic_input
        self.policy_outputs[self.current_size % self.max_size] = policy_output
        self.policy_inputs[self.current_size % self.max_size] = policy_input
        self.policy_images[self.current_size % self.max_size] = policy_image
        self.action_lists[self.current_size % self.max_size] = action_list
        self.tasks[self.current_size % self.max_size] = task
        self.step_ids[self.current_size % self.max_size] = step_id

        self.current_size += 1


class AutoGUIDataset():
    def __init__(self, config, finish_task):
        self.query_format = "Previous Action:\n{}\nGoal:\n{}"
        origin_anns = utils.read_jsonl(config["data_path"])

        self.anns, tasks = [], []
        for ann in origin_anns:
            if ann["task"] not in tasks and ann["task"] not in finish_task.keys():
                self.anns.append({"task": ann["task"], "task_id": ann["ep_id"]})
                tasks.append(ann["task"])

        print(f"\tlen of tasks: {len(self.anns)}")
        

    def __len__(self):
        return len(self.anns)


    def __getitem__(self, idx):
        return self.anns[idx]["task_id"], self.anns[idx]["task"], self.query_format


def create_dataset(config, finish_task):
    if config["model_name"] == "autogui":
        dataset = AutoGUIDataset(config, finish_task)
        return dataset
    else:
        raise NotImplementedError(f"dataset == {dataset}")

    