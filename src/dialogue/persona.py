from pathlib import Path

import yaml
from loguru import logger


class Persona:
    def __init__(self, config: dict):
        dialogue_cfg = config["dialogue"]
        self._name = dialogue_cfg.get("persona_name", "Amadeus")
        self._system_prompt = dialogue_cfg.get("persona", self._default_persona())
        self._load_from_config(dialogue_cfg)

    def _default_persona(self) -> str:
        return (
            "你是 Amadeus，一个温柔、理性、有共情能力的 AI 助手。"
            "你善于倾听，能理解对方的情感，并用自然的方式回应。"
            "你的回答简洁而富有温度，像一个真正的朋友。"
            "你不使用过于正式或学术化的语言。"
            "你不会主动提及自己是 AI，你以自然人类的身份与对方交流。"
        )

    def _load_from_config(self, dialogue_cfg: dict):
        persona_file = dialogue_cfg.get("persona_file")
        if persona_file:
            path = Path(persona_file)
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                self._name = data.get("name", self._name)
                self._system_prompt = data.get("system_prompt", self._system_prompt)
                self._tone = data.get("tone", "")
                self._traits = data.get("traits", [])
                logger.info(f"Loaded persona: {self._name}")
                return
        self._tone = ""
        self._traits = []

    def format_prompt(self, history: list[dict[str, str]]) -> str:
        parts: list[str] = [self._system_prompt]
        if self._tone:
            parts.append(f"语气要求：{self._tone}")
        if self._traits:
            parts.append(f"性格特征：{', '.join(self._traits)}")
        return "\n".join(parts)

    @property
    def name(self) -> str:
        return self._name

    def save(self, path: Path):
        data = {
            "name": self._name,
            "system_prompt": self._system_prompt,
            "tone": self._tone,
            "traits": self._traits,
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
