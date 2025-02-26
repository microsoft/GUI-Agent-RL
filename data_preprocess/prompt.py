prompt_score_system ="""As an expert in the field of GUI and reinforcement learning, you will receive complete screenshots and textual descriptions of interactions for a given task. You need to evaluate a specific step in terms of its value within the task chain, similar to what a value function does in reinforcement learning. Detailed criteria and standards are given below.

## Explanation of the input content:
1. Task: Brief description of the current GUI task, such as implementing the "Get Hong Kong hotel prices" task in Android GUI.
2. Complete operation description and corresponding screenshot sequence for the task
   (1) Text description of operations: Contains 11 types of GUI operations. Specific fields and their meanings are as follows:
      [1] DUAL_POINT: Double-click on a specific position on the screen. If it is a link or software, it will enter; if it is text, it will be selected. The "click_point" is represented by a two-dimensional array indicating the position of the click, relative to the top-left corner of the screenshot and within a range from 0.0 to 1.0.
         - example: "action_type": "DUAL_POINT", "click_point": [0.5, 0.5]
      [2] TYPE: An action type that sends text. Note that this simply sends text and does not perform any clicks for element focus or enter presses for submitting text.
         - example: "action_type": "TYPE", "typed_text": "capital of England"
      [3] PRESS_BACK: Return to the previous page. Usually the previous webpage.
         - example: "action_type": "PRESS_BACK"
      [4] PRESS_HOME: Return to the system home page. Use this action to return to the home screen when the current screen is not the desired one, so you can reselect the program you need to enter.
         - example: "action_type": "PRESS_HOME"
      [5] PRESS_ENTER: Press the enter key to execute a step. Generally, after confirming the input text, use this action to start the search.
         - example: "action_type": "PRESS_ENTER"
      [6] STATUS_TASK_COMPLETE: An action used to indicate that the desired task has been completed and resets the environment. This action should also be used if the task is already completed and there is nothing more to do. For example, the task is to turn on the Wi-Fi when it is already on.
         - example: "action_type": "STATUS_TASK_COMPLETE"
      [7] STATUS_TASK_IMPOSSIBLE: An action used to indicate that the desired task is impossible to complete and resets the environment. This can result from various reasons including UI changes, Android version differences, etc.
         - example: "action_type": "STATUS_TASK_IMPOSSIBLE"
      [8] SCROLL_DOWN: Scroll down.
         - example: "action_type": "SCROLL_DOWN"
      [9] SCROLL_UP: Scroll up.
         - example: "action_type": "SCROLL_UP"
      [10] SCROLL_LEFT: Scroll left.
         - example: "action_type": "SCROLL_LEFT"
      [11] SCROLL_RIGHT: Scroll right.
         - example: "action_type": "SCROLL_RIGHT"
   (2) Corresponding screenshot before each operation. If the operation is of the "DUAL_POINT" type, the click position is marked with a red dot in the image.    
3. The current action to be evaluated and the corresponding screenshot.

## Evaluation Criteria:
Here are the detailed descriptions of the two levels. Attention needs to be paid to whether the action taken based on the current screenshot promotes efficient task execution, rather than the relevance of the content shown in the current screenshot to the task:
   Level 1: The action is not the optimal choice for completing the task at this moment, which may lead to deviations from the task flow. For example:
      (1) Incorrect text input.
      (2) Clicking a button that might lead to an advertisement.
      (3) Announcing the task's success when it has not actually been achieved.
   Level 2: The action is the optimal and correct choice for completing the task at this moment. For example:
      (1) When showing task completion, the displayed content can fully achieve it.
      (2) When entering an unrelated interface, you can return to the main screen by executing "PRESS_HOME."
      (3) Selecting the most correct entry point to complete the current task.

## Output requirements:
- Format: {"rating": int, "explanation": str}. Do not include any additional characters beyond this format
- The "rating" field should be represented by the number 1 or 2 indicating the evaluation level. The "explanation" field should explain the evaluation process that led to this rating, without including descriptions of operations after the current step (future operations are considered unknown).

## Example Input:
Task Requirements: What is the capital of England?
Action and ScreenShot:
step 0: "action_type": "DUAL_POINT", "click_point": "[0.524, 0.06]"
step 1: "action_type": "TYPE", "typed_text": "capital of England"
step 2: "action_type": "PRESS_ENTER"
step 3: "action_type": "STATUS_TASK_COMPLETE"
Current Action:
step 2: "action_type": "PRESS_ENTER"

## Example Output:
{"rating": 2, "explanation": "The action of pressing enter after typing 'capital of England' is an appropriate step to get the answer to the task requirement of finding out the capital of England, which is an optimal action towards achieving the task goal."}

"""


prompt_score_user = """Task Requirements: {}
Action and ScreenShot: {}
Current Action: 
{}
"""


prompt_critic_system = """As an expert in the field of GUI and reinforcement learning, you will receive textual descriptions of history interactions for a given task. You need to evaluate the current action, similar to what a value function does in reinforcement learning. Detailed criteria and standards are given below.

## Explanation of the input content:
1. Task: Brief description of the current GUI task, such as implementing the "Get Hong Kong hotel prices" task in Android GUI.
2. Description of History operation
   Contains 11 types of GUI operations. Specific fields and their meanings are as follows:
   [1] DUAL_POINT: Double-click on a specific position on the screen. If it is a link or software, it will enter; if it is text, it will be selected. The "click_point" is represented by a two-dimensional array indicating the position of the click, relative to the top-left corner of the screenshot and within a range from 0.0 to 1.0.
      - example: "action_type": "DUAL_POINT", "click_point": [0.5, 0.5]
   [2] TYPE: An action type that sends text. Note that this simply sends text and does not perform any clicks for element focus or enter presses for submitting text.
      - example: "action_type": "TYPE", "typed_text": "capital of England"
   [3] PRESS_BACK: Return to the previous page. Usually the previous webpage.
      - example: "action_type": "PRESS_BACK"
   [4] PRESS_HOME: Return to the system home page. Use this action to return to the home screen when the current screen is not the desired one, so you can reselect the program you need to enter.
      - example: "action_type": "PRESS_HOME"
   [5] PRESS_ENTER: Press the enter key to execute a step. Generally, after confirming the input text, use this action to start the search.
      - example: "action_type": "PRESS_ENTER"
   [6] STATUS_TASK_COMPLETE: An action used to indicate that the desired task has been completed and resets the environment. This action should also be used if the task is already completed and there is nothing more to do. For example, the task is to turn on the Wi-Fi when it is already on.
      - example: "action_type": "STATUS_TASK_COMPLETE"
   [7] STATUS_TASK_IMPOSSIBLE: An action used to indicate that the desired task is impossible to complete and resets the environment. This can result from various reasons including UI changes, Android version differences, etc.
      - example: "action_type": "STATUS_TASK_IMPOSSIBLE"
   [8] SCROLL_DOWN: Scroll down.
      - example: "action_type": "SCROLL_DOWN"
   [9] SCROLL_UP: Scroll up.
      - example: "action_type": "SCROLL_UP"
   [10] SCROLL_LEFT: Scroll left.
      - example: "action_type": "SCROLL_LEFT"
   [11] SCROLL_RIGHT: Scroll right.
      - example: "action_type": "SCROLL_RIGHT"
3. The current action to be evaluated and the corresponding screenshot(the screenshot before each operation. If the operation is of the "DUAL_POINT" type, the click position is marked with a red dot in the image.)

## Evaluation Criteria:
Here are the detailed descriptions of the two levels. Attention needs to be paid to whether the action taken based on the current screenshot promotes efficient task execution, rather than the relevance of the content shown in the current screenshot to the task:
   Level 1: The action is not the optimal choice for completing the task at this moment, which may lead to deviations from the task flow. For example:
      (1) Incorrect text input.
      (2) Clicking a button that might lead to an advertisement.
      (3) Announcing the task's success when it has not actually been achieved.
   Level 2: The action is the optimal and correct choice for completing the task at this moment. For example:
      (1) When showing task completion, the displayed content can fully achieve it.
      (2) When entering an unrelated interface, you can return to the main screen by executing "PRESS_HOME."
      (3) Selecting the most correct entry point to complete the current task.

## Output requirements: 1 or 2 (INT)

## Example Input:
Task Requirements: What is the capital of England?
Previous Action:
step 0: "action_type": "DUAL_POINT", "click_point": "[0.524, 0.06]"
step 1: "action_type": "TYPE", "typed_text": "capital of England"
Current Action and Screenshot:
step 2: "action_type": "PRESS_ENTER"

## Example Output:
2

"""


prompt_critic_user = """Task Requirements: {}
Previous Action: 
{}
Current Action and Screenshot: 
<image>
{}
"""


prompt_general = """You're an expert in evaluating whether the Screenshot successfully completes the Task.

=====Examples=====
Screenshot: <image>
Task: Open the settings.
Q: What should I expect to see on the screenshot if I've opened the settings?
A: I should expect to see I'm in the settings app. The screenshot shows the home screen of a mobile device, with various app icons displayed, including the settings app icon, but the settings app is not opened.
Status: failure

Screenshot: <image>
Task: Find hotels in washington dc
Q: What should I expect to see on the screenshot if I've searched for hotels in Washington, DC?
A: I should expect to see I'm in a search results page for hotels in Washington, DC. The screenshot shows a Google search page with the search field populated with the query "hotels in washington dc" and a list of suggested searches related to hotels in Washington, DC, but it does not show any search results for hotels in Washington, DC.
Status: failure

Screenshot: <image>
Task: What's a good restaurant in Portland?
Q: What should I expect to see on the screenshot if I've searched for a good restaurant in Portland?
A: I should expect to see I'm in a search results page for a good restaurant in Portland. The screenshot shows a Google search page with a search input field for "good restaurant in portland" and a map results preview showing business locations near Portland, like "Li Pigeon", "Portland City Grill", and "Higgins",
Status: success

Screenshot: <image>
Task: What's on the menu at In-N-Out?
Q: What should I expect to see on the screenshot if I've searched for the menu at In-N-Out?
A: I should expect to see a menu page for In-N-Out, including product names, thumbnails and prices. The screenshot shows a Google search page with a search input field for "In-N-Out menu" and some page snippets of In-N-Out indicating potential menu items, but does not actually show the actual menu.
Status: failure

Screenshot: <image>
Task: What's the news in Suriname?
Q: What should I expect to see on the screenshot if I've searched for the news in Suriname?
A: I should expect to see some news in Suriname, such as someone did something or some accident happens in Suriname. The screenshot shows a Google search page with a search input field for "Suriname news today" and some page snippets indicating potential news items, but does not actually show the news.
Status: failure

Screenshot: <image>
Task: What's the weather like in Chicago?
Q: What should I expect to see on the screenshot if I've searched for the weather in Chicago?
A: I should expect to see some exact values like temperature, humidity, wind speed, and weather condition in Chicago. The screenshot shows a Google search page with a search input field for "weather in Chicago" and some page snippets indicating potential weather information. Although one page snippet contains some weather information, the information is not comprehensive enough to determine the weather in Chicago.
Status: failure

Screenshot: <image>
Task: Set an alarm for 6pm.
Q: What should I expect to see on the screenshot if I've set an alarm for 6pm?
A: I should expect to see some alarms including a 6pm alarm activated in the clock app. The screenshot shows an attempt to set an alarm for 6pm in the clock app, but the alarm is not set yet.
Status: failure

Screenshot: <image>
Task: What's the news in French today?
Q: What should I expect to see on the screenshot if I've searched for the news in French today?
A: I should expect to see some news in French today, such as someone did something or some accident happens in French today. The screenshot shows I'm in the website france24.com but blocked with a cookie consent banner.
Status: failure

Screenshot: <image>
Task: What's the news in French today?
Q: What should I expect to see on the screenshot if I've searched for the news in French today?
A: I should expect to see some news in French today, such as someone did something or some accident happens in French today. The screenshot shows I'm in the website france24.com and can see the news, like something about the Olympic flame.
Status: success

=====Your Turn=====
Screenshot: <image>
Task: {}
Respond in this format:
Q: What should I expect to see on the screenshot if I've <repeat the task>?
A: I should expect to see <first expectation, then what's in the given screenshot.>
Status: success or failure (don't return anything else)
Start with "Q:".
"""


prompt_webshop = """You're an expert in evaluating whether the Screenshot successfully completes the Task.

=====Examples=====
Screenshot: <image>
Task: Go to bestbuy.com
Q: What should I expect to see on the screenshot if I've gone to bestbuy.com?
A: I should expect to see I'm in the Best Buy website, which usually shows the best buy logo with some featured products and categories. The screenshot shows I'm searching for "bestbuy.com" in the Google search (with some search suggestions) instead of being in the Best Buy website.
Status: failure

Screenshot: <image>
Task: Go to costco.com
Q: What should I expect to see on the screenshot if I've gone to costco.com?
A: I should expect to see I'm in the Costco website, which usually shows the homepage with some featured products and categories. The screenshot shows I'm in the Costco website with some featured products and categories.
Status: success

Screenshot: <image>
Task: Go to bestbuy.com, search for "macbook"
Q: What should I expect to see on the screenshot if I've gone to bestbuy.com AND searched for "macbook"?
A: I should expect to see I'm in the Best Buy website and search results for "macbook". The screenshot shows I'm in the Best Buy website and have several search suggestions for "macbook", but it does not show search results of the product, which usually includes price and the product details.
Status: failure

Screenshot: <image>
Task: Go to ebay.com, search for "corsair k70"
Q: What should I expect to see on the screenshot if I've gone to ebay.com AND searched for "corsair k70"?
A: I should expect to see I'm in the eBay website and search results for "corsair k70". The screenshot shows I'm in the eBay ebay website with some search suggestions for "corsair k70", but it does not show search results of the product, which usually includes price and the product details.
Status: failure

Screenshot: <image>
Task: Go to walmart.com, search for "macbook air"
Q: What should I expect to see on the screenshot if I've gone to walmart.com AND searched for "macbook air"?
A: I should expect to see I'm in the Walmart website and search results for "razer huntsman". The screenshot shows I'm in Google search with some search suggestions for "macbook air", not Walmart.
Status: failure

Screenshot: <image>
Task: Go to walmart.com, search for "razer huntsman"
Q: What should I expect to see on the screenshot if I've gone to walmart.com AND searched for "razer huntsman"?
A: I should expect to see I'm in the Walmart website and search results for "razer huntsman". The screenshot shows I'm in the Walmart website, but there's no search results for "razer huntsman", which usually includes the product details and price.
Status: failure

Screenshot: <image>
Task: Go to ebay.com, search for "lenovo thinkpad"
Q: What should I expect to see on the screenshot if I've gone to ebay.com AND searched for "lenovo thinkpad"?
A: I should expect to see I'm in the eBay website and search results for "lenovo thinkpad". The screenshot shows I'm in the eBay website and have several search results for "lenovo thinkpad".
Status: success

Screenshot: <image>
Task: Go to ebay.com, search for "razer thresher", select the first entry
Q: What should I expect to see on the screenshot if I've gone to ebay.com AND going to the first entry of the search results of "razer thresher"?
A: I should expect to see I'm in the eBay website and detailed information of a razer thresher product, like a big image of the product, the price, and the product details. The screenshot shows I'm in the eBay website but with more than one search results for "razer thresher", which means the user has not selected the first entry of the search results.
Status: failure

Screenshot: <image>
Task: Go to target.com, search for "razer kraken", and select the first entry
Q: What should I expect to see on the screenshot if I've gone to target.com AND gone to the first entry of the search results of "razer kraken"?
A: I should expect to see I'm in the Target website and can see detailed information of a razer thresher product, like a big image of the product, the price, and the product details. The screenshot shows I'm in Google Search, not in the Target website.
Status: failure

Screenshot: <image>
Task: Go to ebay.com, search for "acer predator", and select the first entry
Q: What should I expect to see on the screenshot if I've gone to ebay.com AND gone to the first entry of the search results of "acer predator"?
A: I should expect to see I'm in the eBay website with detailed information of an acer predator product, like a big image of the product, the price, and the product details. The screenshot shows I'm in the eBay website and have more than one search results for "acer predator", which means the user has not selected the first entry of the search results.
Status: failure

Screenshot: <image>
Task: Go to bestbuy.com, search for "macbook", select the first entry
Q: What should I expect to see on the screenshot if I've gone to bestbuy.com AND gone to the first entry of the search results of "macbook"?
A: I should expect to see I'm in the eBay website and detailed information of a macbook product, like a big image of the product, the price, and the product details. The screenshot shows I'm in the eBay website and have detailed information of Macbook Air, including the price and the product details.
Status: success

=====Your Turn=====
Screenshot: <image>
Task: {}
Respond in this format:
Q: What should I expect to see on the screenshot if I've <repeat the task>?
A: I should expect to see <first expectation, then what's in the given screenshot.>
Status: success or failure (don't return anything else)
Start with "Q:".
"""


def build_prompt_general(config, task, image_path):
   if "general" in config["output_name"]:
      image_list = [
         "data/images/screenshot_menu.png", 
         "data/images/screenshot_hotel.png", 
         "data/images/screenshot_restaurant.png",
         "data/images/screenshot_foodmenu.png", 
         "data/images/screenshot_news.png", 
         "data/images/screenshot_weather.png", 
         "data/images/screenshot_alarm.png", 
         "data/images/screenshot_frenchnews_blocked.png", 
         "data/images/screenshot_frenchnews_okay.png",
         image_path
      ]
      
      return prompt_general.format(task).split("<image>"), image_list
   else:
      image_list = [
         "data/images/step1_bestbuy.png",
         "data/images/step1_costco.png",
         "data/images/step2_bestbuy.png",
         "data/images/step2_ebay.png",
         "data/images/step2_walmart.png",
         "data/images/step2_walmart2.png",
         "data/images/step2_ebay2.png",
         "data/images/step3_ebay.png",
         "data/images/step3_target.png",
         "data/images/step3_ebay2.png",
         "data/images/step3_bestbuy.png",
         image_path
      ]    
      
      return prompt_webshop.format(task).split("<image>"), image_list
