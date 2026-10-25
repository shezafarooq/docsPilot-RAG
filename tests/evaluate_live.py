"""Explicit live acceptance checks; consumes Gemini quota. Not a unit test."""

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

QUESTIONS = [
    ("greeting", "Hi!"),
    ("container", "what is a docker container"),
    ("docker", "How do I mount a named volume in Docker Compose? Show a small example."),
    ("fastapi", "How do I declare a request body in FastAPI using a Pydantic model?"),
    ("missing", "How do I configure Kubernetes ClusterRoleBindings for RBAC?"),
]


def main():
    chunks = {chunk["id"]: chunk for chunk in (
        json.loads(line) for line in Path("data/chunks.jsonl").read_text().splitlines()
    )}
    examples = []
    for label, question in QUESTIONS:
        print("ASKING:", label, flush=True)
        request = Request("http://localhost:8000/ask", data=json.dumps({"question": question}).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=180) as response:
                result = json.load(response)
        except HTTPError as error:
            print("HTTP", error.code, error.read().decode(), flush=True)
            raise SystemExit(1)
        examples.append({"case": label, "question": question, "response": result})
        Path("data/ask_examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n")
        for source in result["sources"]:
            original = chunks[source["chunk_id"]]
            assert source["url"] == original["url"] and source["passage"] == original["text"]
            assert f"[{source['id']}]" in result["answer"]
        if label == "greeting":
            assert result["status"] == "direct" and result["retrieval_attempts"] == 0
        elif label == "missing":
            assert result["status"] == "not_found" and result["retrieval_attempts"] == 3 and result["rewrite_count"] == 2
        else:
            assert result["status"] == "answered" and result["sources"]
            if label == "container":
                assert any("/what-is-a-container/" in source["url"] for source in result["sources"])
        print(json.dumps({
            "case": label, "status": result["status"], "attempts": result["retrieval_attempts"],
            "rewrites": result["rewrite_count"], "trace_id": result["trace_id"],
            "answer": result["answer"][:2200], "sources": [source["url"] for source in result["sources"]],
        }, indent=2), flush=True)
    print("All live acceptance checks passed.", flush=True)


if __name__ == "__main__":
    main()
