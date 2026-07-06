from __future__ import annotations


class ArithmeticTokenizer:
    BASE_TOKENS = ["<pad>", *list("0123456789"), "+", "=", "\n"]
    EXTENDED_TOKENS = ["<pad>", *list("0123456789"), "+", "-", "*", "/", "=", "R", "E", "\n"]

    def __init__(self, tokens: list[str] | None = None) -> None:
        self.tokens = list(tokens) if tokens is not None else list(self.EXTENDED_TOKENS)
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<pad>"]
        self.newline_id = self.token_to_id["\n"]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.token_to_id[character] for character in text]
        except KeyError as error:
            raise ValueError(f"unsupported character: {error.args[0]!r}") from error

    def decode(self, token_ids: list[int], skip_pad: bool = True) -> str:
        characters: list[str] = []
        for token_id in token_ids:
            if skip_pad and token_id == self.pad_id:
                continue
            characters.append(self.id_to_token[int(token_id)])
        return "".join(characters)
