#!/usr/bin/env python
import sys

# Test various imports
tests = [
    "from langchain import chains",
    "from langchain.chains import RetrievalQA",
    "from langchain_community.chains.retrieval_qa.base import RetrievalQA",
    "from langchain.chains.retrieval_qa import RetrievalQA",
]

for test in tests:
    try:
        exec(test)
        print(f"✓ {test}")
    except Exception as e:
        print(f"✗ {test} - {e}")

# Check what's in langchain
print("\nLangchain modules:")
import langchain
for attr in dir(langchain):
    if not attr.startswith('_'):
        print(f"  - {attr}")
