"""Convert test_data.jsonl to LLaMA-Factory chat format with qwen3_vl QUESTION_TEMPLATE.

Input: test_data.jsonl (each line JSON with fields: problem, process, solution, path, ...)
Output: test_data_converted.jsonl where each line has messages list with system/user/assistant.

Usage:
  python convert_to_llamafactory.py --input test_data.jsonl --output test_data_converted.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


QUESTION_TEMPLATE = (
	"{Question}\n\n"
	"Please analyze the video carefully by identifying key segments and their important visual clues within "
	"`<time> </time>`, `<caption> </caption>`, `<think> </think>` tags "
	"then conduct deep analysis and reasoning to arrive at your answer to the question, "
	"finally provide only the single option letter (e.g., A, B, C, D, E, F etc.) within the `<answer> </answer>` tags."
	"Follow the format specified in the instructions."
)


def convert_example(raw: Dict[str, Any]) -> Dict[str, Any]:
	question = raw.get("problem", "").strip()
	options = raw.get("options")
	if options:
		# append options text
		opts_text = "\n".join(options)
		question = f"{question}\n\nOptions:\n{opts_text}"

	user_prompt = QUESTION_TEMPLATE.format(Question=question)

	messages = [
		{"role": "user", "content": user_prompt},
	]

	solution = raw.get("solution")
	if solution:
		messages.append({"role": "assistant", "content": solution})

	# carry over media path if present for VL training
	res: Dict[str, Any] = {"messages": messages}
	if "path" in raw:
		res["videos"] = [raw["path"]]  # single video path wrapped as list
	return res


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", type=Path, default=Path("test_data.jsonl"))
	parser.add_argument("--output", type=Path, default=Path("test_data_converted.jsonl"))
	args = parser.parse_args()

	out_lines = []
	with args.input.open() as fin:
		for line in fin:
			line = line.strip()
			if not line:
				continue
			raw = json.loads(line)
			out_lines.append(convert_example(raw))

	with args.output.open("w") as fout:
		for item in out_lines:
			fout.write(json.dumps(item, ensure_ascii=False) + "\n")

	print(f"Wrote {len(out_lines)} examples to {args.output}")


if __name__ == "__main__":
	main()
