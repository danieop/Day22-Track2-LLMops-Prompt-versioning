from langsmith import traceable

from qa_pairs import SAMPLE_QUESTIONS
from rag_common import PROMPT_V1, build_rag_chain, build_vectorstore


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    return chain.invoke(question)


def main() -> None:
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    vectorstore = build_vectorstore()
    chain, _ = build_rag_chain(vectorstore, PROMPT_V1)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question}")
        print(f"       A: {answer[:220]}\n")

    print(f"Sent {len(SAMPLE_QUESTIONS)} traced RAG calls to LangSmith.")
    print("Open https://smith.langchain.com to verify traces.")


if __name__ == "__main__":
    main()
