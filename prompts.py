import os

def load_prompt(name):
    with open(os.path.join("prompts", f"{name}.txt"), "r", encoding="utf-8") as f:
        return f.read()