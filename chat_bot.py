import openai
import yaml


class Chatbot:
    with open("secret.yaml", encoding="UTF-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    openai.api_key = cfg["openai_api"]
    memory_size = 5
    gpt_standard_messages = [
        {
            "role": "system",
            "content": "Your task is to very accurately calculate the quantities of ingredients and provide the ingredients and cooking instructions.\
            The calculated quantities of ingredients should be rounded to the first decimal place.\
            When responding, do not show the calculation process or formula. Respond in the same structure as the original input, for example: [Ingredients] Apples 1 each Beef 200g [Seasoning] Salt 1t. Respond in Korean.",
        }
    ]

    def set_memory_size(self, memory_size):
        self.memory_size = memory_size

    def get_send_msg(self, question: str):
        self.gpt_standard_messages.append({"role": "user", "content": question})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=self.gpt_standard_messages, temperature=0.5
        )

        answer = response["choices"][0]["message"]["content"]
        self.gpt_standard_messages.append({"role": "assistant", "content": answer})
        if self.memory_size * 2 < len(self.gpt_standard_messages):
            self.gpt_standard_messages.pop(1)
            self.gpt_standard_messages.pop(1)

        return answer
