import os
import json
import warnings

import pandas as pd
from huggingface_hub import HfFileSystem

from ..smp import *
from .video_base import VideoBaseDataset
from .utils import DEBUG_MESSAGE, build_judge
from .utils.prompts import SYSTEM_PROMPT, QUESTION_TEMPLATE
from .utils.videoholmes import extract_option
from ..smp.file import get_file_extension, get_intermediate_file_path

FAIL_MSG = 'Failed to obtain answer via API.'


class TestVideoDataset(VideoBaseDataset):
	SYS = SYSTEM_PROMPT
	QUESTION_TMPL = QUESTION_TEMPLATE
	TYPE = 'Video-MCQ'
	DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4o-mini']

	def __init__(self, dataset='TestVideoDataset', nframe=16, fps=-1):
		super().__init__(dataset=dataset, nframe=nframe, fps=fps)
		self.dataset_name = dataset

	@classmethod
	def supported_datasets(cls):
		return ['TestVideoDataset']

	def _load_jsonl(self, jsonl_path):
		rows = []
		with open(jsonl_path, 'r', encoding='utf-8') as f:
			for idx, line in enumerate(f):
				if not line.strip():
					continue
				try:
					item = json.loads(line)
				except json.JSONDecodeError as e:
					warnings.warn(f"Line {idx} is not valid JSON: {e}")
					continue

				video_path = item.get('path') or item.get('video_path')
				if video_path is None:
					warnings.warn(f"Line {idx} missing 'path'; skipping")
					continue

				video_abs = video_path if os.path.isabs(video_path) else os.path.join(self.data_root, video_path)
				if not os.path.exists(video_abs):
					warnings.warn(f"Video not found at {video_abs}; ensure files are placed correctly.")

				video_id = os.path.splitext(os.path.basename(video_path))[0]

				options = item.get('options', [])
				if isinstance(options, dict):
					# sort by key to keep deterministic order
					options = [options[k] for k in sorted(options.keys())]
				candidates = [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)]

				rows.append({
					'index': idx,
					'video': video_id,
					'video_path': video_path,
					'candidates': candidates,
					'question': item.get('problem', ''),
					'answer': (item.get('solution', '') or '').strip().upper(),
					'question_id': item.get('question_id', idx),
					'question_type': item.get('question_type', ''),
					'explanation': item.get('explanation', ''),
				})
		return pd.DataFrame(rows)

	def prepare_dataset(self, dataset='TestVideoDataset', jsonl_name='longvideo_test.jsonl'):
		# locate jsonl either via env or local data folder
		dataset_root = os.environ.get('TEST_VIDEO_DATA_ROOT', os.path.join(LMUDataRoot(), 'longvideo_test'))
		os.makedirs(dataset_root, exist_ok=True)
		jsonl_path = os.environ.get('TEST_VIDEO_JSONL', os.path.join(dataset_root, jsonl_name))

		if not os.path.exists(jsonl_path):
			raise FileNotFoundError(
				f"JSONL file not found at {jsonl_path}. Provide a longvideo-style jsonl via TEST_VIDEO_JSONL or place it under {dataset_root}."
			)

		df = self._load_jsonl(jsonl_path)
		data_file = os.path.join(dataset_root, f'{dataset}.tsv')
		df.to_csv(data_file, sep='\t', index=False)

		return dict(data_file=data_file, root=dataset_root)

	def _resolve_video_path(self, video_path):
		return video_path if os.path.isabs(video_path) else os.path.join(self.data_root, video_path)

	def save_video_frames(self, line, video_llm=False):
		"""Extract frames for a single line; similar to VideoBase but respects stored video_path."""
		import decord
		video_path = self._resolve_video_path(line['video_path'])
		vid = decord.VideoReader(video_path)
		video_info = {
			'fps': vid.get_avg_fps(),
			'n_frames': len(vid),
		}

		if self.nframe > 0 and self.fps < 0:
			step_size = len(vid) / (self.nframe + 1)
			indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
			frame_paths = self.frame_paths(line['video'])
		elif self.fps > 0:
			total_duration = video_info['n_frames'] / video_info['fps']
			required_frames = int(total_duration * self.fps)
			step_size = video_info['fps'] / self.fps
			indices = [int(i * step_size) for i in range(required_frames)]
			frame_paths = self.frame_paths_fps(line['video'], len(indices))
		else:
			raise ValueError('fps and nframe are both invalid')

		flag = np.all([osp.exists(p) for p in frame_paths])
		if not flag:
			lock_path = osp.splitext(video_path)[0] + '.lock'
			with portalocker.Lock(lock_path, 'w', timeout=30):
				if not np.all([osp.exists(p) for p in frame_paths]):
					images = [vid[i].asnumpy() for i in indices]
					images = [Image.fromarray(arr) for arr in images]
					for im, pth in zip(images, frame_paths):
						if not osp.exists(pth):
							im.save(pth)

		return frame_paths, indices, video_info, video_path

	def build_prompt(self, line, video_llm):
		if isinstance(line, int):
			assert line < len(self)
			line = self.data.iloc[line]

		message = [dict(type='text', value=self.SYS)]

		frames, indices, video_info, video_path = self.save_video_frames(line, video_llm)

		if video_llm:
			message.append(dict(type='video', value=video_path))
		else:
			for im in frames:
				message.append(dict(type='image', value=im))

		text_prompt = self.QUESTION_TMPL.format(
			question=line['question'],
			options=line['candidates'],
		)
		message.append(dict(type='text', value=text_prompt))
		return message

	@classmethod
	def evaluate(self, eval_file, **judge_kwargs):
		from ..smp.file import dump, load

		assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], 'data file should be xlsx/json/tsv'

		score_file = get_intermediate_file_path(eval_file, '_score')
		tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')

		if not osp.exists(score_file):
			data = load(eval_file)
			data['score'] = -1

			res = data[~pd.isna(data['prediction'])]
			for idx in data['index']:
				ans = str(data.loc[data['index'] == idx, 'answer'].values[0]).strip().upper()
				pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])
				predicted_answer = extract_option(pred)
				data.loc[idx, 'score'] = int(predicted_answer == ans)

			dump(data, score_file)

		data = load(score_file)
		valid_scores = [s for s in data['score'] if s >= 0]
		acc = float(np.mean(valid_scores)) if len(valid_scores) else 0.0
		rating = {'overall_acc': acc, 'total': len(data), 'valid': len(valid_scores)}
		dump(rating, tgt_file)
		return rating

