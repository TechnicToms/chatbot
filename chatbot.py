import numpy as np


class TerminalColor:
    info = "\u001b[1m\u001b[34m[INFO] \u001b[0m"
    success = "\u001b[1m\u001b[32;1m[SUCCESS] \u001b[0m"
    error = "\u001b[1m\u001b[31m[ERROR] \u001b[0m"
    warning = "\u001b[1m\u001b[33;1m[WARNING] \u001b[0m"


class ChatBot:
    def __init__(self, name="Eve"):
        self.tc = TerminalColor
        print(self.tc.info + f"Starting Chatbot, \u001b[4m{name}\u001b[0m!")


if __name__ == "__main__":
    ai = ChatBot()
    print(TerminalColor.success + "finished!")
