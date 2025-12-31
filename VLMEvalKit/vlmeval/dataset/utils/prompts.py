SYSTEM_PROMPT = (
	"You are a helpful vision assistant. "
	"Look at the provided video content, reason carefully, and answer the question. "
	"First think through the problem, then give only the final option as the answer."
)

QUESTION_TEMPLATE = (
	"<think>Reason about the video content and the question.</think>\n"
	"Question: {question}\n"
	"Options: {options}\n"
	"<answer>The best option is </answer>"
)

__all__ = ["SYSTEM_PROMPT", "QUESTION_TEMPLATE"]
