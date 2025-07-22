import os


def load_prompt(name):
    with open(os.path.join("prompts", f"{name}.txt"), "r", encoding="utf-8") as f:
        return f.read()


def load_persona(persona_name):
    with open(
        os.path.join("persona", f"{persona_name}.txt"), "r", encoding="utf-8"
    ) as f:
        return f.read()
