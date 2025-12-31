export TEST_VIDEO_DATA_ROOT=videothinker/vsi
export TEST_VIDEO_JSONL=/home/wangxingjian/videothinker/vsitest.jsonl

export FORCE_QWENVL_VIDEO_READER=decord

export CUDA_VISIBLE_DEVICES=0

python /home/wangxingjian/OneMLLM/VLMEvalKit/run.py --config /home/wangxingjian/OneMLLM/VLMEvalKit/test/test_config.json 