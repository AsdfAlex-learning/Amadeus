class ConversationContext:
    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self._messages: list[dict[str, str]] = []
        self._token_count = 0
        self._summary = ""

    def add_user_message(self, text: str):
        self._messages.append({"role": "user", "content": text})
        self._token_count += self._estimate_tokens(text)
        self._trim()

    def add_assistant_message(self, text: str):
        self._messages.append({"role": "assistant", "content": text})
        self._token_count += self._estimate_tokens(text)
        self._trim()

    def add_system_message(self, text: str):
        self._messages.insert(0, {"role": "system", "content": text})
        self._token_count += self._estimate_tokens(text)

    def get_messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def get_user_messages(self) -> list[str]:
        return [m["content"] for m in self._messages if m["role"] == "user"]

    def clear(self):
        self._messages.clear()
        self._token_count = 0
        self._summary = ""

    def _trim(self):
        while self._token_count > self.max_tokens and len(self._messages) > 1:
            removed = self._messages.pop(0)
            if removed["role"] == "system":
                self._messages.insert(0, removed)
                if len(self._messages) <= 1:
                    break
                removed = self._messages.pop(0)
            self._token_count -= self._estimate_tokens(removed["content"])
            self._token_count = max(0, self._token_count)
            self._summary = (
                f"[早期对话摘要: {removed['content'][:200]}...]"
                if len(removed["content"]) > 200
                else ""
            )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 2)

    @property
    def is_empty(self) -> bool:
        return len(self._messages) == 0

    @property
    def token_count(self) -> int:
        return self._token_count
