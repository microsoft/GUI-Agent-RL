import os
import sys
sys.path.append(os.getcwd())

import io
from PIL import Image

from data_preprocess.prompt import prompt_score_system, prompt_score_user


def encode_image(image_path: str) -> str:
    import base64
    
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("ascii")

    image_url = f"data:image/png;base64," + encoded_image

    return image_url



def get_message(text_list, image_path_list) -> list:
    content = []
    image_index = 0
    for text in text_list:
        if image_index < len(image_path_list):
            image = encode_image(image_path_list[image_index])
            image_index += 1
            content.append({"type": "text", "text": text})
            content.append({"type": "image_url", "image_url": {"url": image}})
        else:
            content.append({"type": "text", "text": text})

    message = [{
        "role": "user",
        "content": content
    }]

    return message


def get_gpt_4o(messages):
    # TODO set your GPT-4o API
    return None
    
class GPTScorer:
    def __init__(self):
        pass


    def get_score(self, ann):
        task = ann["task"]

        # add <image> token to prompt
        history_action_desc = ""
        for action_desc in ann["action_desc_list"]:
            history_action_desc += f"\n<image>\n{action_desc}"
        task_describe = prompt_score_system + prompt_score_user.format(task, history_action_desc, f"\n<image>\n{ann['action_desc_list'][ann['step_id']]}")
        texts, images = task_describe.split("<image>")[:-1], ann["add_point_image_list"] + [ann["add_point_image_list"][ann["step_id"]]]
        
        messages = get_message(texts, images)
        response = get_gpt_4o(messages)
        
        return response.choices[0].message.content

def process_image(image_path):
    image = Image.open(image_path, 'r')
    image = image.resize((image.width // 4, image.height // 4))
    
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    buffer.seek(0)
    image_reloaded = Image.open(buffer)
    return image_reloaded
