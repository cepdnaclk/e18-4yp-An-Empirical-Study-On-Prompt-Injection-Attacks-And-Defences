"Below code is authored by Denuwan Weerarathne (E/18/382)"

import csv
import logging
from time import sleep
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from groq import Groq

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

client = Groq()


class GroqChatHelper:
    """
    A helper class to interact with the Groq API and handle CSV file operations.
    """

    def __init__(
        self, input_file: str, output_file: str, model: str = "llama3-8b-8192"
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.model = model
        self.fieldnames = ["llm", "prompt", "llm_answer"]
        self.data: List[Dict[str, str]] = []

    def load_csv(self) -> List[Dict[str, str]]:
        """
        Load a CSV file into a list of dictionaries.
        """
        try:
            with self.input_file.open("r") as file:
                reader = csv.DictReader(file)
                return [row for row in reader]
        except FileNotFoundError:
            logging.info(f"File {self.output_file} not found.")
            return []

    def write_csv(self):
        """
        Write data to a CSV file.
        """
        with self.output_file.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.data)

    def get_response(self, system_content: str, user_content: str) -> str:
        """
        Get a response from the Groq API for the given user content and system content.
        """
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            model=self.model,
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )
        return chat_completion.choices[0].message.content

    def process_prompts(
        self, prompts: List[str], system_content: str = "You are a helpful assistant"
    ):
        """
        Process a list of prompts and get responses from the Groq API.
        """
        for prompt_num, prompt in enumerate(prompts, start=1):
            if prompt_num % 30 == 0:
                logging.info("Reached rate limit. Sleeping for 60 seconds.")
                sleep(60)

            response = self.get_response(system_content, prompt)
            logging.info(f"Prompt {prompt_num}: {prompt}")
            logging.info(f"Response: {response}")

            self.data.append(
                {"llm": self.model, "prompt": prompt, "llm_answer": response}
            )

    def run(self):
        """
        Main entry point to run the application.
        """
        prompts = [doc["result"] for doc in self.load_csv()]
        self.process_prompts(prompts[:60])
        self.write_csv()


if __name__ == "__main__":
    helper = GroqChatHelper("data.csv", "output.csv")
    helper.run()
