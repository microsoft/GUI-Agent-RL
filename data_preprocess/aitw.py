import os
import sys
sys.path.insert(0, os.getcwd())

from tqdm import tqdm
import json
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import utils
from data_preprocess.utils import action_type_dict, to_autoui, action_dict_to_class
from data_preprocess.prompt import prompt_critic_system, prompt_critic_user
from data_preprocess.gpt import GPTScorer


def get_unfinish_anns(anns, rpath):
    if os.path.exists(rpath):
        unfinish_anns = []
        finish_anns = utils.read_jsonl(rpath)
        finish_ids = [f"{ann['ep_id']}_{ann['step_id']}" for ann in finish_anns]
        for ann in anns:
            if f"{ann['ep_id']}_{ann['step_id']}" in finish_ids:
                pass
            else:
                unfinish_anns.append(ann)
        print(f"### finish anns: {len(finish_anns)} unfinish: {len(unfinish_anns)}")
        return unfinish_anns
    else:
        return anns


class AITW:
    def __init__(self, split: str, part: str):
        self.image_dir = "images/aitw_images"
        self.split = split

        self.part = part
        if not os.path.exists(f"data/aitw_anns"):
            os.mkdir(f"data/aitw_anns")
        self.gpt = GPTScorer()
    

    def get_unfold_data(self):
        anns = utils.read_json(f"data/aitw_anns/aitw_{self.split}.json")[self.part]
        steps = []
        for episode in tqdm(anns):
            action_list, action_type_list, image_list, add_point_image_list = [], [], [], []
            action_desc_list, action_desc_all_list = [], []
            for step_id, step in enumerate(episode):
                image_filename = f"{step['img_filename']}.png"
                image_path = os.path.join(self.image_dir, image_filename).replace("\\", "/")
                if not os.path.exists(image_path):
                    print(f"{image_path} image not found")
                    continue
                
                # all the models need
                action_dict = {
                    "action_type": action_type_dict[step["action_type_text"]], 
                    "touch_point": step["touch"], 
                    "lift_point": step["lift"], 
                    "typed_text": step["type_text"]
                }
                action_list.append(action_dict)
                action_type_list.append(action_dict["action_type"])
                image_list.append(image_path)
                action = action_dict_to_class(action_dict)

                # for different model input
                action_desc_list.append(f"step {step_id}: " + to_autoui(action, all_dict=False))
                action_desc_all_list.append(to_autoui(action, all_dict=True))
                
                add_point_image_list.append(utils.add_visilize2screenshot(image_path, action_list[-1], "score"))
            
            for step_id, step in enumerate(episode):
                steps.append({
                    "ep_id": step["ep_id"], 
                    "step_id": step_id,
                    "task": step["goal"], 
                    "action_list": action_list,
                    "action_type_list": action_type_list,
                    "image_list": image_list,
                    "add_point_image_list": add_point_image_list,
                    "position": step["annot_position"],
                    "action_type": action_type_dict[step["action_type_text"]], 
                    "touch_point": step["touch"], 
                    "lift_point": step["lift"], 
                    "typed_text": step["type_text"],
                    "action_desc_list": action_desc_list,
                    "action_desc_all_list": action_desc_all_list
                })
        
        utils.write_jsonl(steps, f"data/aitw_anns/{self.part}_{self.split}.jsonl")

    
    def get_gpt_label(self):
        anns = utils.read_jsonl(f"data/aitw_anns/{self.part}_{self.split}.jsonl")
        ann_wpath = f"data/aitw_anns/{self.part}_{self.split}_critic.jsonl"
        unfinish_anns = get_unfinish_anns(anns, ann_wpath)

        write_lock = threading.Lock()

        def process_ann(ann):
            response = self.gpt.get_score(ann)
            response = utils.parse_response(response)
            ann["critic_output"], ann["critic_explanation"] = response["rating"], response["explanation"]

            conversations = [
                {"from": "human", "value": prompt_critic_system + prompt_critic_user.format(ann["task"], "\n".join(ann["action_desc_list"][:ann["step_id"]]), ann["action_desc_list"][ann["step_id"]])},
                {"from": "gpt", "value": str(response["rating"])}
            ]
            ann["critic_inputs"] = conversations
            ann["critic_images"] = ann["add_point_image_list"][ann["step_id"]].replace("\\", "/")

            return ann

        # Open the file in append mode outside of the threads
        with open(ann_wpath, "a") as fout:
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_ann = {executor.submit(process_ann, ann): ann for ann in unfinish_anns}
                for future in tqdm(as_completed(future_to_ann), total=len(future_to_ann)):
                    ann = future_to_ann[future]
                    try:
                        result = future.result()
                        with write_lock:
                            fout.writelines(json.dumps(result) + "\n")
                    except Exception as exc:
                        print(f'Error processing annotation {ann}: {exc}')
            fout.close()


    def get_rl_data(self):
        ann_rpath = f"data/aitw_anns/{self.part}_{self.split}.jsonl"
        ann_wpath = f"data/aitw_anns/{self.part}_{self.split}_policy.jsonl"
        anns = utils.read_jsonl(ann_rpath)

        for ann in tqdm(anns):
            previous_actions = "\n".join(ann["action_desc_all_list"][:ann["step_id"]])
            ann["policy_image"] = ann["image_list"][ann["step_id"]]
            ann["policy_input"] = f"Previous Action:\n{previous_actions}\nGoal:\n{ann['task']}".replace("'", "")
            ann["policy_output"] = f"Action Plan: {ann['action_type_list'][ann['step_id']:]} ; Action Decision: {ann['action_desc_all_list'][ann['step_id']]}".replace("'", "")

        utils.write_jsonl(anns, ann_wpath)


if __name__ == "__main__":
    aitw_data = AITW(split="train", part="general")
    aitw_data.get_unfold_data()
    aitw_data.get_gpt_label()
    aitw_data.get_rl_data()

    aitw_data = AITW(split="val", part="general")
    aitw_data.get_unfold_data()
    aitw_data.get_gpt_label()
    aitw_data.get_rl_data()

    aitw_data = AITW(split="train", part="webshopping")
    aitw_data.get_unfold_data()
    aitw_data.get_gpt_label()
    aitw_data.get_rl_data()

    aitw_data = AITW(split="val", part="webshopping")
    aitw_data.get_unfold_data()
    aitw_data.get_gpt_label()
    aitw_data.get_rl_data()