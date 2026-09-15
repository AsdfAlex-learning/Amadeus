import threading
from collections.abc import Generator
from pathlib import Path

from loguru import logger


class DialogueModel:
    def __init__(self, config: dict):
        dialogue_cfg = config["dialogue"]
        self.model_type = dialogue_cfg["model_type"]
        self.model_path = Path(dialogue_cfg["model_path"])
        self.model_size = dialogue_cfg.get("model_size", "3B")
        self.quantization = dialogue_cfg.get("quantization", "int8")
        self.max_tokens = int(dialogue_cfg.get("max_context_tokens", 4096))
        # Generation parameters. max_new_tokens was previously hardcoded to 512
        # inside generate() while self.max_tokens (the context budget) went unused.
        self.max_new_tokens = int(dialogue_cfg.get("max_new_tokens", 512))
        self.temperature = float(dialogue_cfg.get("temperature", 0.7))
        self.top_p = float(dialogue_cfg.get("top_p", 0.9))
        self.repetition_penalty = float(dialogue_cfg.get("repetition_penalty", 1.1))

        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._loaded = False

    def _detect_device(self) -> str:
        """Auto-detect best available device: CUDA > MPS > CPU."""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> bool:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = self._resolve_model_id()
            logger.info(f"Loading dialogue model: {model_id}")

            self._device = self._detect_device()
            load_kwargs = {"device_map": self._device, "torch_dtype": torch.float16}
            if self.quantization == "int8":
                load_kwargs["load_in_8bit"] = True
            elif self.quantization == "int4":
                load_kwargs["load_in_4bit"] = True

            self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, **load_kwargs
            )
            self._loaded = True
            logger.info(f"Dialogue model loaded on {self._device}")
            return True
        except ImportError:
            logger.warning("Transformers not available. Dialogue model will use API fallback.")
            return False
        except Exception as e:
            logger.error(f"Failed to load dialogue model: {e}")
            return False

    def generate(self, messages: list[dict[str, str]]) -> Generator[str, None, None]:
        if not self._loaded:
            yield from self._generate_api(messages)
            return

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        result_chunks: list[str] = []

        def _run():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            response = self._tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            result_chunks.append(response)

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()

        # NOTE (C-7, unresolved): this is not real streaming. The whole generation
        # completes before the first chunk is yielded, and `yield from` a string
        # emits one character at a time. Fixing it needs an incremental token
        # stream, but the pinned transformers (4.32.1, see environment.yml) does not
        # provide one: its TextIteratorStreamer buffers and flushes only at end().
        # See docs/PROJECT_REVIEW_REPORT.md.
        full_text = "".join(result_chunks)
        yield from full_text

    def _generate_api(self, messages: list[dict[str, str]]) -> Generator[str, None, None]:
        import json
        from urllib import request

        api_url = "http://localhost:11434/api/chat"
        payload = json.dumps(
            {
                "model": f"qwen2.5:{self.model_size.lower()}",
                "messages": messages,
                "stream": True,
            }
        ).encode("utf-8")

        req = request.Request(api_url, data=payload)
        req.add_header("Content-Type", "application/json")

        try:
            with request.urlopen(req) as resp:
                for line in resp:
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"API fallback failed: {e}")
            yield "（对话模型暂时不可用，请确认 Ollama 已启动）"

    def _resolve_model_id(self) -> str:
        model_map = {
            "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
            "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
            "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        }
        return model_map.get(f"{self.model_type}-{self.model_size.lower()}", self.model_type)

    @property
    def is_loaded(self) -> bool:
        return self._loaded
