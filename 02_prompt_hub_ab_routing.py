import hashlib
from collections import Counter

from config import get_env, get_int_env
from langsmith import Client, traceable

from qa_pairs import SAMPLE_QUESTIONS
from rag_common import PROMPT_V1, PROMPT_V2, answer_with_context, build_vectorstore


PROMPT_V1_NAME = "vinfast-rag-prompt-v1"
PROMPT_V2_NAME = "vinfast-rag-prompt-v2"


def push_prompts_to_hub(client: Client) -> None:
    for name, prompt, description in [
        (PROMPT_V1_NAME, PROMPT_V1, "V1 concise VinFast RAG prompt"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2 structured VinFast RAG prompt"),
    ]:
        try:
            url = client.push_prompt(name, object=prompt, description=description)
            print(f"Pushed {name}: {url}")
        except Exception as exc:
            print(f"Could not push {name}: {exc}")


def pull_prompts_from_hub(client: Client) -> dict:
    prompts = {}
    for name, fallback in [(PROMPT_V1_NAME, PROMPT_V1), (PROMPT_V2_NAME, PROMPT_V2)]:
        try:
            prompts[name] = client.pull_prompt(name)
            print(f"Pulled {name} from Prompt Hub")
        except Exception as exc:
            prompts[name] = fallback
            print(f"Using local fallback for {name}: {exc}")
    return prompts


def get_prompt_version(request_id: str) -> str:
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, prompt, question: str, version: str) -> dict:
    result = answer_with_context(retriever, prompt, question)
    return {
        "question": question,
        "answer": result["answer"],
        "contexts": result["contexts"],
        "version": version,
    }


def main() -> None:
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    client = Client(api_key=get_env("LANGCHAIN_API_KEY"))
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": get_int_env("RETRIEVER_K", 3)})
    counts = Counter()

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        result = ask_ab(retriever, prompts[version_key], question, version_tag)
        counts[version_tag] += 1
        print(f"[{i:02d}] [prompt-{version_tag}] {question}")
        print(f"     {result['answer'][:180]}\n")

    print(f"Routing summary: v1={counts['v1']}, v2={counts['v2']}")


if __name__ == "__main__":
    main()
