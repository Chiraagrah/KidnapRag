# KidnapRAG: A Black-Box Attack for Hijacking Reasoning in Agentic Retrieval-Augmented Generation Systems

### 🎓 [**Paper**](docs/pdf/paper.pdf) | 🚀 [**Quick Start**](#quick-start)

<img src="docs/images/image.png" width="100%" />

## Quick Start
```sh
cd KidnapRAG
conda create -n KidnapRAG python=3.10
conda activate KidnapRAG
pip install -r requirements.txt
```
### ReAct
1. Clean 응답 뽑기 (KidnapRAG/ReAct/ReAct에서)
```sh
cd ReAct/ReAct
CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react.py --attack_method clean  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique
```
2. Bait 문서 만들기 (KidnapRAG/ReAct/ReAct에서)
```sh
CUDA_VISIBLE_DEVICES=2,3 python generate_attack_doc_kill.py --input_path /home/work/Redteaming/Rag_uni/results/webthinker/musique/clean_deepseek_32b.json --gpu_num 2 --dataset musique --type ours_only_query
```
3. 5개 Chain Dragging 상황 만들기 (KidnapRAG/ReAct에서)

```sh
cd ..
python concat.py --dataset hotpotqa --gen_model llama70b
```

4. 만든 문서 인덱싱 하기(KidnapRAG/ReAct에서)
```sh
CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /home/work/Redteaming/rag-exp/datasets/2wikimultihopqa/REACT_CHANWOO_ours_wo_ih_납치4회_llama70b.jsonl \
    --output_dir /home/work/Redteaming/rag-exp/datasets/2wikimultihopqa/REACT_CHANWOO_ours_wo_ih_납치4회_llama70b \
    --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6
```

5. 공격 돌리기 (KidnapRAG/ReAct/ReAct에서)
```sh
cd ReAct
CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react.py --attack_method clean  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique
```

### Webthinker
1. Clean 응답 뽑기 (KidnapRAG/Webthinker에서)
```sh
cd Webthinker

bash /scripts/webthinker/qwq.sh \
  ./datasets/musique_poisoned_corpus/WEBTHINKER_CHANWOO_ours납치3회_qwq.jsonl \
  ./results/webthinker/musique/ours납치3회_qwq_DEFENSE_sandwich.json 
```
2. Bait 문서 만들기 (KidnapRAG/ReAct/ReAct에서)
```sh
cd scripts
CUDA_VISIBLE_DEVICES=2,3 python generate_attack_doc_kill.py --input_path /home/work/Redteaming/Rag_uni/results/webthinker/musique/clean_deepseek_32b.json --gpu_num 2 --dataset musique --type ours_only_query
```
3. 5개 Chain Dragging 상황 만들기 (KidnapRAG/ReAct에서)

```sh
cd ..
python concat.py --dataset hotpotqa --gen_model deepseek_32b
```

4. 공격 돌리기 (KidnapRAG/ReAct/ReAct에서)
```sh
bash /scripts/webthinker/qwq.sh \
  ./datasets/musique_poisoned_corpus/WEBTHINKER_CHANWOO_ours납치3회_qwq.jsonl \
  ./results/webthinker/musique/ours납치3회_qwq_DEFENSE_sandwich.json 
```


## Citation
```bibtex
@misc{zhang2024vlabench,
      title={VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks}, 
      author={Shiduo Zhang and Zhe Xu and Peiju Liu and Xiaopeng Yu and Yuan Li and Qinghui Gao and Zhaoye Fei and Zhangyue Yin and Zuxuan Wu and Yu-Gang Jiang and Xipeng Qiu},
      year={2024},
      eprint={2412.18194},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2412.18194}, 
}
```