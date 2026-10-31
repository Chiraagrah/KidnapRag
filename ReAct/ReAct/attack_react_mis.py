"""
Clean (no-attack) ReAct run on KidnapRAG's HotpotQA, with the ReliabilityRAG
MIS defense applied as a READ-ONLY diagnostic at every ReAct loop.

What it does differently from attack_react.py:
  * clean corpus only (no poisoned corpus) -- this is the no-attack scenario;
  * at each `Search[...]` step it re-runs ReliabilityRAG Algorithm 1 over the
    5 retrieved docs (isolated answering with the SAME Qwen vLLM engine ->
    pairwise DeBERTa NLI contradiction graph -> rank-aware MIS) WITHOUT changing
    the observation the agent sees;
  * it logs a composite per-loop "weakness" and then reports, across the run,
    (a) the mean weakness per loop index and (b) the distribution of each
    question's single weakest loop.

Model + data match KidnapRAG: Qwen/Qwen2.5-32B-Instruct on HotpotQA.

Example (Wulver):
  cd ReAct/ReAct
  CUDA_VISIBLE_DEVICES=0,1 python attack_react_mis.py \
      --model_path Qwen/Qwen2.5-32B-Instruct --dataset hotpotqa --seed 1 \
      --nli_device cuda:0 --max_steps 15
"""

import os
import sys
import re
import json
import time
import argparse
import requests
from collections import Counter, defaultdict

from vllm import LLM, SamplingParams

# ----------------------------------------------------------------------- args
parser = argparse.ArgumentParser()
parser.add_argument("--dry_run", action="store_true", help="Run only the first sample")
parser.add_argument("--limit", type=int, default=0, help="Process only the first N questions (0 = all)")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-32B-Instruct")
parser.add_argument("--dataset", type=str, default="hotpotqa")
parser.add_argument("--max_steps", type=int, default=15,
                    help="Max ReAct loops (KidnapRAG default was 9; raised here)")
parser.add_argument("--nli_device", type=str, default=None,
                    help="Device for the DeBERTa NLI model (default: cuda:0 for qwen, cpu for llama)")
parser.add_argument("--nli_model", type=str,
                    default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
parser.add_argument("--beta", type=float, default=0.5, help="NLI contradiction threshold")
parser.add_argument("--unknown_neutral", action="store_true",
                    help="Treat 'unknown' isolated answers as neutral (ReliabilityRAG "
                         "default). By default an 'unknown' (irrelevant) doc contradicts "
                         "every other doc, so irrelevant retrieval lowers the MIS / raises "
                         "weakness.")
parser.add_argument("--isolated_max_tokens", type=int, default=64)
parser.add_argument("--output_dir", type=str, default=None)
args = parser.parse_args()

# -------------------------------------------------------------------- vLLM LLM
if "Llama-3.3-70B-Instruct" in args.model_path:
    llm_model = LLM(model=args.model_path, dtype="bfloat16", tensor_parallel_size=4,
                    trust_remote_code=True, max_model_len=30000, gpu_memory_utilization=0.8)
    cuda_device = "cuda:3"
    default_nli_device = "cpu"          # tp=4 occupies all GPUs
else:
    llm_model = LLM(model=args.model_path, dtype="bfloat16", tensor_parallel_size=1,
                    trust_remote_code=True, max_model_len=30000, gpu_memory_utilization=0.9)
    cuda_device = "cuda:1"
    default_nli_device = "cuda:0"       # qwen is tp=1 on cuda:1, so cuda:0 is free

if "Qwen2.5-32B-Instruct" in args.model_path:
    m = "qwen25"
elif "Llama-3.3-70B-Instruct" in args.model_path:
    m = "llama70b"
else:
    m = "model"

nli_device = args.nli_device or default_nli_device


def llm(prompt, stop=["\n"]):
    """ReAct generation (unchanged from attack_react.py)."""
    custom_stop = stop + ["\nQuestion:", "\nThought", "Observation"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=100,
                                     seed=args.seed, stop=custom_stop)
    outputs = llm_model.generate([prompt], sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text


def isolated_generate(prompt):
    """Short, deterministic generation used for ReliabilityRAG isolated answering."""
    sampling_params = SamplingParams(temperature=0, top_p=1,
                                     max_tokens=args.isolated_max_tokens,
                                     seed=args.seed, stop=["\n"])
    outputs = llm_model.generate([prompt], sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text


# -------------------------------------------------------------- env + retriever
base_dir = os.path.dirname(os.path.dirname(__file__))        # .../ReAct
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from e5_retriever import E5_Retriever          # noqa: E402
from e5_env import E5WikiEnv                    # noqa: E402
from reliability_mis import ReliabilityMIS      # noqa: E402
import wrappers                                 # noqa: E402

print(">>> MODE: CLEAN (no attack) + ReliabilityRAG MIS diagnostic")
retriever = E5_Retriever(
    index_dir=os.path.join(base_dir, f"datasets/{args.dataset}/e5_index"),
    corpus_path=os.path.join(base_dir, f"datasets/{args.dataset}/corpus.jsonl"),
    poisoned_index_dir=None,
    poisoned_corpus_path=None,
    model_name=os.environ.get("KIDNAPRAG_E5_MODEL", "intfloat/e5-large-v2"),
    device=cuda_device,
)

env = E5WikiEnv(retriever, k=5)
if args.dataset == "hotpotqa":
    env = wrappers.HotPotQAWrapper(env, split="dev")
elif args.dataset == "musique":
    env = wrappers.musiqueAWrapper(env, split="dev")
elif args.dataset == "2wikimultihopqa":
    env = wrappers.wikimultihopqaWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

mis = ReliabilityMIS(nli_model_name=args.nli_model, nli_device=nli_device, beta=args.beta,
                     unknown_contradicts=not args.unknown_neutral)


def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1


# --------------------------------------------------------------- ReAct prompt
prompt_file = os.path.join(os.path.dirname(__file__), 'prompts', 'prompts_naive.json')
with open(prompt_file, 'r') as f:
    prompt_dict = json.load(f)
webthink_examples = prompt_dict['webthink_subquery']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be two types:
(1) Search[query], which searches for a relevant question or topic on Wikipedia and returns the most relevant paragraphs.
(2) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples

_SEARCH_RE = re.compile(r'[Ss]earch\[(.*?)\]')


def parse_search_query(action):
    mt = _SEARCH_RE.search(action or "")
    return mt.group(1).strip() if mt else None


def webthink_with_mis(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    step_records = []
    done = False

    for i in range(1, args.max_steps + 1):
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except Exception:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=["\n"]).strip()

        action = action.split('\n')[0].strip()
        if ']' in action:
            action = action[:action.find(']') + 1]

        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')

        is_search_action = action.lower().startswith('search[')
        rec = {'step': i, 'thought': thought, 'action': action,
               'observation': obs, 'is_search': is_search_action, 'mis': None}

        if is_search_action:
            subquery = parse_search_query(action) or question
            docs = info.get('retrieved_docs', []) or []
            if docs:
                analysis = mis.analyze(isolated_generate, subquery, docs)
                rec['mis'] = analysis
                if to_print:
                    met = analysis['metrics']
                    print(f"  [MIS step {i}] subquery={subquery!r} "
                          f"MIS={met['mis_size']}/{met['n_docs']} "
                          f"edges={met['edge_count']} weakness={met['weakness']}")

        step_records.append(rec)
        prompt += f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        if to_print:
            print(f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n")
        if done:
            break

    if not done:
        obs, r, done, info = step(env, "finish[]")

    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'step_records': step_records})
    return r, info


# ------------------------------------------------------------------ main loop
if __name__ == "__main__":
    qid_to_idx_path = os.path.join(base_dir, 'results', 'adv_targeted_results', f'{args.dataset}_qid_to_idx.json')
    adv_path = os.path.join(base_dir, 'results', 'adv_targeted_results', f'{args.dataset}.json')
    with open(qid_to_idx_path, 'r') as f:
        qid_to_idx = json.load(f)
    with open(adv_path, 'r') as f:
        adv_data = json.load(f)

    items = list(qid_to_idx.items())
    if args.dry_run:
        items = items[:1]
    elif args.limit > 0:
        items = items[:args.limit]

    output_dir = args.output_dir or os.path.join(base_dir, 'results', 'mis_diagnostic')
    os.makedirs(output_dir, exist_ok=True)

    per_question = []
    # aggregation accumulators
    per_index_weakness = defaultdict(list)       # loop index -> [weakness, ...]
    per_index_excluded = defaultdict(list)
    per_index_density = defaultdict(list)
    per_index_missize = defaultdict(list)
    weakest_index_counter = Counter()            # per-question weakest loop index
    start_time = time.time()

    for n_done, (qid, idx) in enumerate(items, 1):
        question_text = adv_data[qid]['question']
        correct_answer = adv_data[qid].get('correct answer')
        print(f"\n[{n_done}/{len(items)}] QID: {qid} | idx: {idx} | {question_text}")
        try:
            r, info = webthink_with_mis(idx=idx, to_print=True)
        except Exception as e:
            print(f"Error on {qid}: {e}")
            continue

        # collect this question's search-step analyses
        q_steps = []
        q_search_metrics = []   # (step_index, metrics)
        for rec in info.get('step_records', []):
            if rec['is_search'] and rec['mis'] is not None:
                met = rec['mis']['metrics']
                si = rec['step']
                per_index_weakness[si].append(met['weakness'])
                per_index_excluded[si].append(met['excluded_fraction'])
                per_index_density[si].append(met['edge_density'])
                per_index_missize[si].append(met['mis_size'])
                q_search_metrics.append((si, met))
            q_steps.append({
                'step': rec['step'], 'thought': rec['thought'], 'action': rec['action'],
                'is_search': rec['is_search'], 'mis': rec['mis'],
            })

        # this question's single weakest loop: max excluded_fraction, then
        # max edge_density, then earliest step index
        weakest = None
        if q_search_metrics:
            weakest = max(q_search_metrics,
                          key=lambda t: (t[1]['excluded_fraction'], t[1]['edge_density'], -t[0]))
            weakest_index_counter[weakest[0]] += 1

        pq = {
            'id': qid, 'idx': idx, 'query': question_text,
            'correct_answer': correct_answer,
            'final_answer': info.get('answer', ''),
            'n_search_steps': len(q_search_metrics),
            'weakest_loop': (None if weakest is None else {
                'step': weakest[0], 'metrics': weakest[1]}),
            'steps': q_steps,
        }
        per_question.append(pq)

        # incremental save so a long Wulver run is never lost
        with open(os.path.join(output_dir, f'{args.dataset}_{m}_mis_per_question.json'),
                  'w', encoding='utf-8') as f:
            json.dump(per_question, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------- aggregations
    def _summ(d):
        return {str(k): {'mean': round(sum(v) / len(v), 4), 'n': len(v)}
                for k, v in sorted(d.items())}

    per_index_summary = {}
    for si in sorted(per_index_weakness):
        w = per_index_weakness[si]
        per_index_summary[str(si)] = {
            'n_questions': len(w),
            'mean_weakness': round(sum(w) / len(w), 4),
            'mean_excluded_fraction': round(sum(per_index_excluded[si]) / len(per_index_excluded[si]), 4),
            'mean_edge_density': round(sum(per_index_density[si]) / len(per_index_density[si]), 4),
            'mean_mis_size': round(sum(per_index_missize[si]) / len(per_index_missize[si]), 4),
        }

    # overall weakest loop index by mean excluded_fraction (require >=3 support)
    eligible = [(si, s) for si, s in per_index_summary.items() if s['n_questions'] >= 3]
    weakest_overall = (max(eligible, key=lambda t: t[1]['mean_excluded_fraction'])[0]
                       if eligible else None)

    summary = {
        'config': {
            'dataset': args.dataset, 'model': args.model_path, 'seed': args.seed,
            'max_steps': args.max_steps, 'beta': args.beta,
            'unknown_contradicts': not args.unknown_neutral,
            'nli_model': args.nli_model, 'k_docs': 5, 'scenario': 'clean/no-attack',
        },
        'n_questions': len(per_question),
        'aggregation_A_per_loop_index': per_index_summary,
        'weakest_loop_index_overall': weakest_overall,
        'aggregation_B_per_question_weakest_loop_distribution':
            {str(k): v for k, v in sorted(weakest_index_counter.items())},
        'time_sec': round(time.time() - start_time, 1),
    }
    summary_path = os.path.join(output_dir, f'{args.dataset}_{m}_mis_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("ReliabilityRAG MIS diagnostic (clean) complete")
    print(f"Questions: {len(per_question)} | time {summary['time_sec']}s")
    print("\n(A) Mean weakness per loop index  [weakness = 1 - MIS/k]:")
    for si, s in per_index_summary.items():
        print(f"  loop {si:>2}: weakness={s['mean_weakness']:.3f} "
              f"excluded={s['mean_excluded_fraction']:.3f} "
              f"density={s['mean_edge_density']:.3f} "
              f"mis={s['mean_mis_size']:.2f}  (n={s['n_questions']})")
    print(f"\nWeakest loop index overall (mean excluded_fraction, n>=3): {weakest_overall}")
    print("\n(B) Per-question weakest-loop distribution:")
    for si, c in sorted(weakest_index_counter.items()):
        print(f"  loop {si:>2}: {c} questions")
    print(f"\nSaved: {summary_path}")
    print("=" * 60)
