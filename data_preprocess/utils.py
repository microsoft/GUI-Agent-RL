from enum import Enum
from typing import Tuple
from dataclasses import dataclass
import re

import utils
from data_preprocess.prompt import prompt_critic_system, prompt_critic_user


class ActionType(Enum):
    Idle=0
    DualPoint=1
    Type=2
    GoBack=3
    GoHome=4
    Enter=5
    TaskComplete=6
    TaskImpossible=7
    Up=8
    Down=9
    Left=10
    Right=11


@dataclass
class AndroidAction():
    action_type: ActionType
    touch_point: Tuple[float, float] = None
    lift_point: Tuple[float, float] = None
    typed_text: str = None


action_type_dict = {
    "type": "TYPE",
    "click": "DUAL_POINT",
    "press back": "PRESS_BACK",
    "press home": "PRESS_HOME",
    "press enter": "PRESS_ENTER",
    "status task complete": "STATUS_TASK_COMPLETE",
    "status task impossible": "STATUS_TASK_IMPOSSIBLE",
    "scroll down": "SCROLL_DOWN",
    "scroll up": "SCROLL_UP",
    "scroll left": "SCROLL_LEFT",
    "scroll right": "SCROLL_RIGHT",
}


scroll_map = {
    "up": [[0.8000, 0.5000], [0.2000, 0.5000]],
    "down": [[0.2000, 0.5000], [0.8000, 0.5000]],
    "left": [[0.5000, 0.8000], [0.5000, 0.2000]],
    "right": [[0.5000, 0.2000], [0.5000, 0.8000]]
}


def extract_scroll(action):
    if action.touch_point == action.lift_point:
        return action
    
    drag_delta_x, drag_delta_y = action.lift_point[0] - action.touch_point[0], action.lift_point[1] - action.touch_point[1]
    if drag_delta_y == 0:
        if drag_delta_x < 0: action.action_type = ActionType.Up
        else: action.action_type = ActionType.Down
    elif drag_delta_x == 0:
        if drag_delta_y < 0: action.action_type = ActionType.Left
        else: action.action_type = ActionType.Right
    
    return action


def update_trajectory(anns, results):
    for (result, ann) in zip(results, anns):
        new_action = autoui_translate_action(result["output"])
        try:
            new_action = extract_scroll(new_action)
        except:
            print(f"error get new action: {new_action}")
        
        new_action_desc = to_autoui(new_action, all_dict=False)
        
        history_action_desc = "\n".join(ann["action_desc_list"][:ann["step_id"] - 1])
        
        ann["critic_input"] = prompt_critic_system + prompt_critic_user.format(ann["task"], history_action_desc, new_action_desc)
        ann["policy_output"] = new_action_desc
        ann["critic_image"] = utils.add_visilize2screenshot(ann["policy_image"], new_action, "policy")

    return anns


def action_dict_to_class(action_dict):
    action_type = action_dict["action_type"]
    
    if action_type == 'DUAL_POINT':
        action_class = AndroidAction(action_type=ActionType.DualPoint, touch_point=action_dict["touch_point"][::-1], lift_point=action_dict["lift_point"][::-1])
    elif action_type == 'TYPE':
        action_class = AndroidAction(action_type=ActionType.Type, typed_text=action_dict["typed_text"])
    elif action_type == 'SCROLL_UP':
        return AndroidAction(action_type=ActionType.Up, touch_point=(0.5, 0.5), lift_point=(0.5, 0.2))
    elif action_type == 'SCROLL_DOWN':
        return AndroidAction(action_type=ActionType.Down, touch_point=(0.5, 0.2), lift_point=(0.5, 0.5))
    elif action_type == 'SCROLL_LEFT':
        return AndroidAction(action_type=ActionType.Left, touch_point=(0.8, 0.5), lift_point=(0.2, 0.5))
    elif action_type == 'SCROLL_RIGHT':
        return AndroidAction(action_type=ActionType.Right, touch_point=(0.2, 0.5), lift_point=(0.8, 0.5))
    elif action_type == 'PRESS_HOME':
        action_class = AndroidAction(action_type=ActionType.GoHome)
    elif action_type == 'PRESS_BACK':
        action_class = AndroidAction(action_type=ActionType.GoBack)
    elif action_type == 'PRESS_ENTER':
        action_class = AndroidAction(action_type=ActionType.Enter)
    elif action_type == 'STATUS_TASK_COMPLETE':
        action_class = AndroidAction(action_type=ActionType.TaskComplete)
    elif action_type == 'STATUS_TASK_IMPOSSIBLE':
        action_class = AndroidAction(action_type=ActionType.TaskImpossible)
    else:
        print(f"Action {action_dict} not supported yet.")
        action_class = AndroidAction(action_type=ActionType.Idle)
    
    return action_class


def autoui_translate_action(raw_action):
    try:
        action_str = raw_action.split("Action Decision: ")[1]
        action_type, touch_point_1, touch_point_2, lift_point_1, lift_point_2, typed_text = action_str.split(", ")
        touch_point = touch_point_1 + ", " + touch_point_2
        lift_point = lift_point_1 + ", " + lift_point_2
        action_type = action_type.split(": ")[1].strip('"')
        if action_type == 'DUAL_POINT':
            touch_point_yx = touch_point.split(": ")[1].strip('[]"')
            touch_point_yx = [float(num) for num in touch_point_yx.split(", ")]
            lift_point_yx = lift_point.split(": ")[1].strip('[]"')
            lift_point_yx = [float(num) for num in lift_point_yx.split(", ")]
            action_class = AndroidAction(action_type=ActionType.DualPoint, touch_point=touch_point_yx[::-1], lift_point=lift_point_yx[::-1])
        elif action_type == 'TYPE':
            text = typed_text.split(": ")[1].strip('"')
            action_class = AndroidAction(action_type=ActionType.Type, typed_text=text)
        elif action_type == 'PRESS_HOME':
            action_class = AndroidAction(action_type=ActionType.GoHome)
        elif action_type == 'PRESS_BACK':
            action_class = AndroidAction(action_type=ActionType.GoBack)
        elif action_type == 'PRESS_ENTER':
            action_class = AndroidAction(action_type=ActionType.Enter)
        elif action_type == 'STATUS_TASK_COMPLETE':
            action_class = AndroidAction(action_type=ActionType.TaskComplete)
        elif action_type == 'TASK_IMPOSSIBLE':
            action_class = AndroidAction(action_type=ActionType.TaskImpossible)
        else:
            print(f"Action {raw_action} not supported yet.")
            action_class = AndroidAction(action_type=ActionType.Idle)
    except:
        return AndroidAction(action_type=ActionType.GoHome)
    
    return action_class


def to_autoui(act: AndroidAction, all_dict):
    if all_dict:
        if act.action_type in [ActionType.DualPoint, ActionType.Up, ActionType.Down, ActionType.Left, ActionType.Right]:
            return f'"action_type": "DUAL_POINT", "touch_point": "[{act.touch_point[1]:.4f}, {act.touch_point[0]:.4f}]", "lift_point": "[{act.lift_point[1]:.4f}, {act.lift_point[0]:.4f}]", "typed_text": ""'
        elif act.action_type == ActionType.Type:
            return f'"action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "{act.typed_text}"'
        elif act.action_type == ActionType.GoBack:
            return f'"action_type": "PRESS_BACK", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
        elif act.action_type == ActionType.GoHome:
            return f'"action_type": "PRESS_HOME", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
        elif act.action_type == ActionType.Enter:
            return f'"action_type": "PRESS_ENTER", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
        elif act.action_type == ActionType.TaskComplete or act.action_type == ActionType.TaskImpossible:
            return f'"action_type": "STATUS_TASK_COMPLETE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
        else:
            print(f"Action {act} not supported yet.")
            return ""
    else:
        if act.action_type == ActionType.DualPoint:
            return f'"action_type": "DUAL_POINT", "click_point": "[{act.touch_point[1]:.4f}, {act.touch_point[0]:.4f}]"'
        elif act.action_type == ActionType.Type:
            return f'"action_type": "TYPE", "typed_text": "{act.typed_text}"'
        elif act.action_type == ActionType.GoBack:
            return f'"action_type": "PRESS_BACK"'
        elif act.action_type == ActionType.GoHome:
            return f'"action_type": "PRESS_HOME"'
        elif act.action_type == ActionType.Enter:
            return f'"action_type": "PRESS_ENTER"'
        elif act.action_type == ActionType.TaskComplete or act.action_type == ActionType.TaskImpossible:
            return f'"action_type": "STATUS_TASK_COMPLETE"'
        elif act.action_type == ActionType.Up:
            return f'"action_type": "SCROLL_UP"'
        elif act.action_type == ActionType.Down:
            return f'"action_type": "SCROLL_DOWN"'
        elif act.action_type == ActionType.Left:
            return f'"action_type": "SCROLL_LEFT"'
        elif act.action_type == ActionType.Right:
            return f'"action_type": "SCROLL_RIGHT"'
        else:
            print(f"Action {act} not supported yet.")
            return ""


