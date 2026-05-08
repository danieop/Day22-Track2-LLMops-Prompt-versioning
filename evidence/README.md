# Evidence Summary

## LangSmith and Prompt Hub

Task 1 generated 50 `rag-query` traces in the LangSmith project `day22-vinfast-rag-local`.
Task 2 added 50 A/B routed traces and pushed two Prompt Hub versions: `vinfast-rag-prompt-v1` and `vinfast-rag-prompt-v2`.

## RAGAS Results

Prompt V1 achieved stronger faithfulness and context precision:

- Faithfulness: V1 = 0.9630, V2 = 0.8817
- Context precision: V1 = 0.8333, V2 = 0.7887

Prompt V2 achieved slightly stronger answer relevancy and context recall:

- Answer relevancy: V1 = 0.9195, V2 = 0.9294
- Context recall: V1 = 0.9762, V2 = 0.9783

Overall, V1 is the safer production choice because it is more faithful to the retrieved context. V2 is useful when a slightly more detailed answer style is preferred.

## Guardrails

The PII validator redacts emails, phone numbers, SSNs, and credit card numbers. The JSON validator formats valid JSON, repairs common malformed JSON patterns, and returns a safe fallback JSON object for unrecoverable input.
