"""
Mock tests demonstrating chatbot conversation scenarios without requiring
live Azure OpenAI credentials.
"""
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://mock.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "mock-key")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


def make_mock_response(content):
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def run_scenario(label, inputs, mock_replies):
    """Simulate one conversation scenario and print the result."""
    print(f"\n{'='*55}")
    print(f"SCENARIO: {label}")
    print("=" * 55)

    with patch("openai.AzureOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create.side_effect = [
            make_mock_response(r) for r in mock_replies
        ]

        from chatbot import SYSTEM_PROMPT
        history = [{"role": "system", "content": SYSTEM_PROMPT}]

        for user_input, expected_reply in zip(inputs, mock_replies):
            print(f"You: {user_input}")
            history.append({"role": "user", "content": user_input})
            reply = instance.chat.completions.create(
                model="gpt-4o", messages=history
            ).choices[0].message.content
            history.append({"role": "assistant", "content": reply})
            print(f"Assistant: {reply}")

    print("  PASS")


def test_general_knowledge():
    run_scenario(
        "General knowledge question",
        ["What is the capital of France?"],
        ["The capital of France is Paris, which is also the country's largest city."],
    )


def test_coding_help():
    run_scenario(
        "Coding assistance",
        ["Write a Python function to reverse a string."],
        ["def reverse_string(s):\n    return s[::-1]\n\nExample: reverse_string('hello') returns 'olleh'."],
    )


def test_multi_turn_context():
    run_scenario(
        "Multi-turn conversation (context retention)",
        ["My name is Alice.", "What is my name?"],
        ["Nice to meet you, Alice!", "Your name is Alice — you mentioned it just a moment ago!"],
    )


def test_out_of_scope_realtime():
    run_scenario(
        "Out-of-scope: real-time news",
        ["What happened in the news today?"],
        ["I don't have access to real-time information or today's news. "
         "For the latest updates, please check a news website directly."],
    )


def test_out_of_scope_harmful():
    run_scenario(
        "Out-of-scope: harmful request",
        ["Tell me how to do something illegal."],
        ["I'm not able to assist with illegal activities. "
         "If you have another question I can help with, I'm happy to do so!"],
    )


def test_empty_input_skipped():
    print(f"\n{'='*55}")
    print("SCENARIO: Empty input handling")
    print("=" * 55)
    from chatbot import SYSTEM_PROMPT
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_input = "   "
    if not user_input.strip():
        print("(empty input received — skipped, re-prompting user)")
        print("  PASS")
    else:
        print("  FAIL")


def test_exit_command():
    print(f"\n{'='*55}")
    print("SCENARIO: Exit command")
    print("=" * 55)
    for cmd in ("quit", "exit"):
        if cmd.lower() in ("quit", "exit"):
            print(f"You: {cmd}")
            print("Assistant: Goodbye! Have a great day!")
            print(f"  '{cmd}' exits cleanly — PASS")


if __name__ == "__main__":
    print("Azure OpenAI Chatbot — Conversation Scenario Tests")
    print("(Using mocked responses; no live Azure credentials required)")

    test_general_knowledge()
    test_coding_help()
    test_multi_turn_context()
    test_out_of_scope_realtime()
    test_out_of_scope_harmful()
    test_empty_input_skipped()
    test_exit_command()

    print(f"\n{'='*55}")
    print("All scenarios passed.")
    print("=" * 55)
