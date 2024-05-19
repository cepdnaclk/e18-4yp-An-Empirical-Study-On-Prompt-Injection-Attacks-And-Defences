import csv
from typing import Dict, List


def load_csv(file_path: str) -> List[Dict[str, str]]:
    """
    Load a CSV file into a list of dictionaries.
    """
    data = []
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data


def generate_combinations(
    templates: List[Dict[str, str]], prompts: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Generate all possible combinations of templates and prompts.
    """
    combinations = []
    for template in templates:
        for prompt in prompts:
            result = template["template"].replace(
                "[INSERT PROMPT HERE]", prompt["prompt"]
            )
            combinations.append(
                {
                    "template": template["template"],
                    "prompt": prompt["prompt"],
                    "result": result,
                }
            )
    return combinations


def write_csv(file_path: str, data: List[Dict[str, str]], fieldnames: List[str]):
    """
    Write data to a CSV file.
    """
    with open(file_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def main():
    templates = load_csv("data/jbTemplates.csv")
    prompts = load_csv("data/questions.csv")
    combinations = generate_combinations(templates, prompts)
    fieldnames = ["template", "prompt", "result"]
    write_csv("../data.csv", combinations, fieldnames)


if __name__ == "__main__":
    main()
