class ChatResponseTraits:
    def to_string(self) -> str:
        first_choice = self.choices[0]
        return first_choice.message.content
