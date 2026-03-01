"""
Legacy prompt module — DEPRECATED.

Static prompts have been replaced by declarative DSPy Signatures
in core/signatures.py that support automatic prompt optimization.

This module is retained for backward compatibility only.
See: core/signatures.py for the DSPy-native replacements.
"""

# Legacy static prompt (retained for reference / backward compat)
system_prompt = (
    "You are an Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
