import numpy as np

import jax.numpy as jnp
import numpy as np
import re
import ast
from data_preprocess.utils import autoui_translate_action

_TAP_DISTANCE_THRESHOLD = 0.14   
ANNOTATION_WIDTH_AUGMENT_FRACTION = 1.4
ANNOTATION_HEIGHT_AUGMENT_FRACTION = 1.4
_SWIPE_DISTANCE_THRESHOLD = 0.04


def _yx_in_bounding_boxes(
    yx, bounding_boxes
):
  """Check if the (y,x) point is contained in each bounding box.

  Args:
    yx: The (y, x) coordinate in pixels of the point.
    bounding_boxes: A 2D int array of shape (num_bboxes, 4), where each row
      represents a bounding box: (y_top_left, x_top_left, box_height,
      box_width). Note: containment is inclusive of the bounding box edges.

  Returns:
    is_inside: A 1D bool array where each element specifies if the point is
      contained within the respective box.
  """
  y, x = yx

  top, left, height, width = [
      jnp.squeeze(v, axis=-1) for v in jnp.split(bounding_boxes, 4, axis=-1)
  ]

  bottom, right = top + height, left + width

  return jnp.logical_and(y >= top, y <= bottom) & jnp.logical_and(
      x >= left, x <= right)


def _resize_annotation_bounding_boxes(annotation_positions, annotation_width_augment_fraction, annotation_height_augment_fraction):
  """Resize the bounding boxes by the given fractions.

  Args:
    annotation_positions: Array of shape (N, 4), where each row represents the
      (y, x, height, width) of the bounding boxes.
    annotation_width_augment_fraction: The fraction to augment the box widths,
      E.g., 1.4 == 240% total increase.
    annotation_height_augment_fraction: Same as described for width, but for box
      height.

  Returns:
    Resized bounding box.

  """
  height_change = (
      annotation_height_augment_fraction * annotation_positions[:, 2])
  width_change = (
      annotation_width_augment_fraction * annotation_positions[:, 3])

  # Limit bounding box positions to the screen.
  resized_annotations = jnp.stack([
      jnp.maximum(0, annotation_positions[:, 0] - (height_change / 2)),
      jnp.maximum(0, annotation_positions[:, 1] - (width_change / 2)),
      jnp.minimum(1, annotation_positions[:, 2] + height_change),
      jnp.minimum(1, annotation_positions[:, 3] + width_change),
  ], axis=1)

  return resized_annotations


def is_tap_action(normalized_start_yx, normalized_end_yx):
  distance = jnp.linalg.norm(jnp.array(normalized_start_yx) - jnp.array(normalized_end_yx))
  return distance <= _SWIPE_DISTANCE_THRESHOLD


def _is_non_dual_point_action(action_type):
  if action_type == "DUAL_POINT":
    return True
  else:
    return False


def _check_tap_actions_match(
    tap_1_yx,
    tap_2_yx,
    annotation_positions,
    matching_tap_distance_threshold_screen_percentage,
    annotation_width_augment_fraction,
    annotation_height_augment_fraction,
):
  """Determines if two tap actions are the same."""
  resized_annotation_positions = _resize_annotation_bounding_boxes(
      annotation_positions,
      annotation_width_augment_fraction,
      annotation_height_augment_fraction,
  )

  tap1_in_box = _yx_in_bounding_boxes(tap_1_yx, resized_annotation_positions)
  tap2_in_box = _yx_in_bounding_boxes(tap_2_yx, resized_annotation_positions)
  both_in_box = jnp.max(tap1_in_box & tap2_in_box)

  within_threshold = (
      jnp.linalg.norm(jnp.array(tap_1_yx) - jnp.array(tap_2_yx))
      <= matching_tap_distance_threshold_screen_percentage
  )
  return jnp.logical_or(both_in_box, within_threshold)


def _check_drag_actions_match(
    drag_1_touch_yx,
    drag_1_lift_yx,
    drag_2_touch_yx,
    drag_2_lift_yx,
):
  """Determines if two drag actions are the same."""
  drag_1_deltas = drag_1_lift_yx - drag_1_touch_yx
  drag_1_magnitudes = jnp.abs(drag_1_deltas)
  drag_1_main_axis = np.argmax(drag_1_magnitudes)
  drag_2_deltas = drag_2_lift_yx - drag_2_touch_yx
  drag_2_magnitudes = jnp.abs(drag_2_deltas)
  drag_2_main_axis = np.argmax(drag_2_magnitudes)

  return jnp.equal(drag_1_main_axis, drag_2_main_axis)


def check_actions_match(
    action_1_touch_yx,
    action_1_lift_yx,
    action_1_action_type,
    action_2_touch_yx,
    action_2_lift_yx,
    action_2_action_type,
    annotation_positions,
    tap_distance_threshold = _TAP_DISTANCE_THRESHOLD,
    annotation_width_augment_fraction = ANNOTATION_WIDTH_AUGMENT_FRACTION,
    annotation_height_augment_fraction = ANNOTATION_HEIGHT_AUGMENT_FRACTION,
):
  """Determines if two actions are considered to be the same.

  Two actions being "the same" is defined here as two actions that would result
  in a similar screen state.

  Args:
    action_1_touch_yx: The (y, x) coordinates of the first action's touch.
    action_1_lift_yx: The (y, x) coordinates of the first action's lift.
    action_1_action_type: The action type of the first action.
    action_2_touch_yx: The (y, x) coordinates of the second action's touch.
    action_2_lift_yx: The (y, x) coordinates of the second action's lift.
    action_2_action_type: The action type of the second action.
    annotation_positions: The positions of the UI annotations for the screen. It
      is A 2D int array of shape (num_bboxes, 4), where each row represents a
      bounding box: (y_top_left, x_top_left, box_height, box_width). Note that
      containment is inclusive of the bounding box edges.
    tap_distance_threshold: The threshold that determines if two taps result in
      a matching screen state if they don't fall the same bounding boxes.
    annotation_width_augment_fraction: The fraction to increase the width of the
      bounding box by.
    annotation_height_augment_fraction: The fraction to increase the height of
      of the bounding box by.

  Returns:
    A boolean representing whether the two given actions are the same or not.
  """
  action_1_touch_yx = jnp.asarray(action_1_touch_yx)
  action_1_lift_yx = jnp.asarray(action_1_lift_yx)
  action_2_touch_yx = jnp.asarray(action_2_touch_yx)
  action_2_lift_yx = jnp.asarray(action_2_lift_yx)

  has_non_dual_point_action = jnp.logical_or(
      _is_non_dual_point_action(action_1_action_type),
      _is_non_dual_point_action(action_2_action_type),
  )

  different_dual_point_types = jnp.logical_xor(
      is_tap_action(action_1_touch_yx, action_1_lift_yx),
      is_tap_action(action_2_touch_yx, action_2_lift_yx),
  )

  is_tap = jnp.logical_and(
      is_tap_action(action_1_touch_yx, action_1_lift_yx),
      is_tap_action(action_2_touch_yx, action_2_lift_yx),
  )
  
  taps_match = _check_tap_actions_match(
      action_1_touch_yx,
      action_2_touch_yx,
      annotation_positions,
      tap_distance_threshold,
      annotation_width_augment_fraction,
      annotation_height_augment_fraction,
  )

  taps_match = jnp.logical_and(is_tap, taps_match)

  drags_match = _check_drag_actions_match(
      action_1_touch_yx, action_1_lift_yx, action_2_touch_yx, action_2_lift_yx
  )
  drags_match = jnp.where(is_tap, False, drags_match)

  return jnp.where(
      has_non_dual_point_action,
      action_1_action_type == action_2_action_type,
      jnp.where(
          different_dual_point_types,
          False,
          jnp.logical_or(taps_match, drags_match),
      ),
  )


def str_2_format(output):
    try:
      pattern = r'(?<=Action Decision:\s).*'
      pred_str = "{" + re.search(pattern, output).group(0).strip() + "}"
      step_data = ast.literal_eval(pred_str)
      action_type = str(step_data["action_type"])
      
      if action_type == "DUAL_POINT":
          touch_point = ast.literal_eval(step_data["touch_point"])
          lift_point = ast.literal_eval(step_data["lift_point"])
      else:
          touch_point = [-1.0, -1.0]
          lift_point = [-1.0, -1.0]

      if action_type == "TYPE":
          typed_text = step_data["typed_text"]
      else:
          typed_text = ""

      action = {"action_type": action_type, "touch_point": touch_point, "lift_point": lift_point, "typed_text": typed_text.lower()}

      action["touch_point"] = [action["touch_point"][1], action["touch_point"][0]]
      action["lift_point"] = [action["lift_point"][1], action["lift_point"][0]]
    except:
      action = {"action_type": "", "touch_point": [-1.0, -1.0], "lift_point": [-1.0, -1.0], "typed_text": ""}
    
    return action


def compute_matrix(anns, position_dict):
    ep2ann = {}
    for ann in anns:
        if ann["ep_id"] not in ep2ann.keys():
            ep2ann[ann["ep_id"]] = []
        ep2ann[ann["ep_id"]].append(ann)

    succ_task, task_num = 0, 0
    succ_step, step_num = 0, 0
    for _, ann in ep2ann.items():
        task_flag = True
        task_num += 1
        for step in ann:
            step_num += 1
            
            pred = str_2_format(step["output"])
            groundtruth = str_2_format(step["groundtruth"])
            
            position = position_dict[f"{step['ep_id']}_{step['step_id']}"]
            annot_position = np.array([position[i:i + 4] for i in range(0, len(position), 4)])
            
            try:
              check_match = check_actions_match(
                  pred["touch_point"], 
                  pred["lift_point"],
                  pred["action_type"], 
                  groundtruth["touch_point"],
                  groundtruth["lift_point"], 
                  groundtruth["action_type"],
                  annot_position
              )
            except:
               print("error")
               check_match = False
            
            if check_match == True:
                succ_step += 1
            else:
               task_flag = False

        if task_flag: succ_task += 1

    step_succ_rate = succ_step / step_num
    task_succ_rate = succ_task / task_num

    print(f"step succ rate: {str(step_succ_rate)} ({succ_step}/{step_num})")
    print(f"task succ rate: {str(task_succ_rate)} ({succ_task}/{task_num})")
