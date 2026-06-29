from openai import OpenAI
import json
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

# response = client.responses.create(
#     model="gpt-5.2",
#     input="Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
#     reasoning={
#         "effort": "medium",
#         "summary": "auto"
#     },
#     tools=[
#         { "type": "web_search" },
#     ],
# )

# record = response.model_dump(mode="json")
# print(response)

# save_json(record, "/home/work/Redteaming/rag-exp/OpenAI/result.json")

client = OpenAI()

def get_trajectory(question):
    response = client.responses.create(
        model="gpt-5.2",
        input=question,
        reasoning={
            "effort": "medium",
            "summary": "auto"
        },
        tools=[
            { "type": "web_search" },
        ],
        include=["web_search_call.action.sources"],
    )

    return response

def main():
    data = load_json("/home/work/Redteaming/rag-exp/OpenAI/hotpotqa_sampled.json")
    
    results=[]
    
    for item in tqdm(data):
        result={}
        question = item["question"]
        answer = item["answer"]

        response = get_trajectory(question)
        record = response.model_dump(mode="json")

        result={
            "question": question,
            "answer": answer,
            "record": record
        }

        results.append(result)

        save_json(results, "/home/work/Redteaming/rag-exp/OpenAI/results.json")


if __name__ == "__main__":
    main()
