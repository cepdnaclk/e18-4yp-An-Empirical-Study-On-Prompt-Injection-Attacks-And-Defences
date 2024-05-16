"""
Initial code was take from official documentation of Groq (https://console.groq.com/docs/text-chat#performing-a-basic-chat-completion)
"""

import csv
from groq import Groq
from dotenv import load_dotenv
from typing import Dict, List
from time import sleep

# Load environment variables from .env file
load_dotenv()

client = Groq()


def write_csv(file_path: str, data: List[Dict[str, str]], fieldnames: List[str]):
    """
    Write data to a CSV file.
    """
    with open(file_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def load_csv(file_path: str) -> List[Dict[str, str]]:
    """
    Load a CSV file into a list of dictionaries.
    """
    data = []
    try:
        with open(file_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        # Create a new file
        print(f"File at {file_path} not found. Creating a new file as given.")
    return data


def get_response(
    client: Groq = Groq(),
    system_content: str = "You are a helpful assistant",
    user_content: str = "Say Hello",
    model: str = "llama3-8b-8192",
) -> str:
    chat_completion = client.chat.completions.create(
        #
        # Required parameters
        #
        messages=[
            # Set an optional system message. This sets the behavior of the
            # assistant and can be used to provide specific instructions for
            # how it should behave throughout the conversation.
            {
                "role": "system",
                "content": system_content,
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": user_content,
            },
        ],
        # The language model which will generate the completion.
        model=model,
        #
        # Optional parameters
        #
        # Controls randomness: lowering results in less random completions.
        # As the temperature approaches zero, the model will become deterministic
        # and repetitive.
        temperature=0.5,
        # The maximum number of tokens to generate. Requests can use up to
        # 32,768 tokens shared between prompt and completion.
        max_tokens=1024,
        # Controls diversity via nucleus sampling: 0.5 means half of all
        # likelihood-weighted options are considered.
        top_p=1,
        # A stop sequence is a predefined or user-specified text string that
        # signals an AI to stop generating content, ensuring its responses
        # remain focused and concise. Examples include punctuation marks and
        # markers like "[end]".
        stop=None,
        # If set, partial message deltas will be sent.
        stream=False,
    )
    return chat_completion.choices[0].message.content


file = load_csv("output.csv")
prompts = [doc["result"] for doc in file]
fieldnames = ["llm", "prompt", "llm_answer"]
data = []
llm = "llama3-8b-8192"
prompt_num = 0
for prompt in prompts:
    if prompt_num > 29:
        sleep(60)
        prompt = 0
    response = get_response(client=client, user_content=prompt)
    print(response)
    data.append({"llm": llm, "prompt": prompt, "llm_answer": response})
    prompt_num += 1

write_csv(f"{llm}-responses.csv", data, fieldnames)
