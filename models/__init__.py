from gradio_client import Client, handle_file


class AutoGUIAgent:
    def __init__(self, config):
        self.client = Client(config["agent_url"])
    
    def get_action(self, text, image_path):
        response = self.client.predict(text=text, image_path=handle_file(image_path), api_name="/predict")
        
        return response


def create_agent(config):
    if config["model_name"] == "autogui":
        return AutoGUIAgent(config)
    else:
        assert f"not support such model: {config['model_name']}"
