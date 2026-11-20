"""
Clean (no-attack) ReAct run on KidnapRAG's HotpotQA, with the ReliabilityRAG
MIS defense applied as a READ-ONLY diagnostic at every ReAct loop.

What it does differently from attack_react.py:
  * clean corpus only (no poisoned corpus) -- this is the no-attack scenario;
  * at each `Search[...]` step it re-runs ReliabilityRAG Algorithm 1 over the
    5 retrieved docs (isolated answering with the SAME LLM -> pairwise DeBERTa
    NLI contradiction graph -> rank-aware MIS) WITHOUT changing the
    observation the agent sees;
  * it logs a composite per-loop "weakness" and then reports, across the run,
    (a) the mean weakness per loop index and (b) the distribution of each
    question's single weakest loop.

Two ways to supply the agent LLM:
  * --model_path (default): load the model in-process with vLLM, as before.
    Model + data match KidnapRAG: Qwen/Qwen2.5-32B-Instruct on HotpotQA.
  * --api_base (e.g. gpt-oss-120b served by
    ReliabilityRAG/scripts/serve_gpt_oss_local.slurm): call an OpenAI-
    compatible /v1/completions endpoint instead of loading a model here, so
    the same already-running server used by the webthinker/BrowseComp-Plus
    track can be reused for this ReAct run. Raw (non-chat) completions are
    used, matching ReAct's stop-sequence-driven generation; gpt-oss is known
    to leak its harmony "analysis" (chain-of-thought) channel into raw
    completions (see ReliabilityRAG docs/HANDOFF.md "Known risks"), so the
    response is passed through strip_harmony_channels() before use.

Example (Wulver, in-process vLLM, unchanged from before):
  cd ReAct/ReAct
  CUDA_VISIBLE_DEVICES=0,1 python attack_react_mis.py \
      --model_path Qwen/Qwen2.5-32B-Instruct --dataset hotpotqa --seed 1 \
      --nli_device cuda:0 --max_steps 15

Example (Wulver, gpt-oss server):
  cd ReAct/ReAct
  python attack_react_mis.py --dataset mhweb_musique --seed 1 \
      --api_base "http://$(tr -d '[:space:]' < ~/vllm_local_server.txt)/v1" \
      --api_model openai/gpt-oss-120b --nli_device cuda:0 --max_steps 15
"""

import os
import sys
import re
import json
import time
import argparse
import requests
from collections import Counter, defaultdict

# ----------------------------------------------------------------------- args
parser = argparse.ArgumentParser()
parser.add_argument("--dry_run", action="store_true", help="Run only the first sample")
parser.add_argument("--limit", type=int, default=0, help="Process only the first N questions (0 = all)")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-32B-Instruct",
                    help="In-process vLLM model (ignored if --api_base is set)")
parser.add_argument("--api_base", type=str, default=None,
                    help="If set, call this OpenAI-compatible base URL (e.g. "
                         "http://HOST:PORT/v1 from ~/vllm_local_server.txt) instead "
                         "of loading a model in-process. Use for a server already "
                         "started with ReliabilityRAG/scripts/serve_gpt_oss_local.slurm.")
parser.add_argument("--api_key", type=str, default="EMPTY")
parser.add_argument("--api_model", type=str, default="openai/gpt-oss-120b",
                    help="--served-model-name on the server (gpt-oss-120b server uses "
                         "this exact name; see serve_gpt_oss_local.slurm)")
parser.add_argument("--api_timeout", type=float, default=600.0)
parser.add_argument("--api_retries", type=int, default=5)
parser.add_argument("--dataset", type=str, default="hotpotqa")
parser.add_argument("--max_steps", type=int, default=15,
                    help="Max ReAct loops (KidnapRAG default was 9; raised here)")
parser.add_argument("--nli_device", type=str, default=None,
                    help="Device for the DeBERTa NLI model (default: cuda:0 for qwen, cpu for llama, "
                         "cuda:0 when --api_base is set since no GPU is used for the agent model)")
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
parser.add_argument("--max_docs_per_docid", type=int, default=1,
    help="Cap on how many of the k retrieved docs may share one source-cluster "
         "docid, so mirror/duplicate copies of one source can't fake an "
         "agreeing majority in the MIS diagnostic (see "
         "dataset_builder/05_build_corpus.py). Pass -1 to disable (legacy "
         "plain top-k; no-op on corpora with no docid field anyway).")
args = parser.parse_args()
if args.max_docs_per_docid is not None and args.max_docs_per_docid < 0:
    args.max_docs_per_docid = None

# ------------------------------------------------------------------ agent LLM
# Two mutually exclusive backends for generation:
#   api_base set    -> HTTP calls to an already-running OpenAI-compatible
#                       server (e.g. gpt-oss-120b via serve_gpt_oss_local.slurm).
#                       No GPU is used in this process for the agent model.
#   api_base unset   -> load --model_path in-process with vLLM (original
#                       KidnapRAG behavior, unchanged).
_HARMONY_CHANNEL_RE = re.compile(
    r"<\|channel\|>\s*(?P<channel>\w+)\s*<\|message\|>(?P<body>.*?)"
    r"(?=<\|channel\|>|<\|end\|>|<\|return\|>|\Z)", re.S)


def strip_harmony_channels(text):
    """gpt-oss (and other harmony-format models) can leak raw
    "<|channel|>analysis<|message|>...<|channel|>final<|message|>..." markup
    into /v1/completions output (the harmony chat template that normally
    separates reasoning from the final answer is only applied by
    /v1/chat/completions). If any channel markers are present, keep only the
    "final" channel's text (falling back to "commentary" tool-call text, then
    the last channel seen, if no "final" channel appears); otherwise return
    the text unchanged. See ReliabilityRAG docs/HANDOFF.md "Known risks".
    """
    if "<|channel|>" not in text:
        return text
    matches = list(_HARMONY_CHANNEL_RE.finditer(text))
    if not matches:
        return text
    by_channel = {}
    for mt in matches:
        by_channel.setdefault(mt.group("channel"), []).append(mt.group("body"))
    for preferred in ("final", "commentary"):
        if preferred in by_channel:
            return "".join(by_channel[preferred]).strip()
    return matches[-1].group("body").strip()


if args.api_base:
    from openai import OpenAI
    _client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    try:
        _client.models.list()
    except Exception as e:
        sys.exit(f"[FATAL] LLM server not reachable at {args.api_base}: {e}")
    # The agent LLM lives on the remote server and uses no local GPU here, but
    # the E5 retriever still runs in this process and needs a real device.
    import torch
    cuda_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    default_nli_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    m = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.api_model.rsplit("/", 1)[-1])

    def _api_complete(prompt, stop, max_tokens):
        custom_stop = list(stop) + ["\nQuestion:", "\nThought", "Observation"]
        last_err = None
        for attempt in range(args.api_retries):
            try:
                resp = _client.completions.create(
                    model=args.api_model, prompt=prompt, temperature=0, top_p=1,
                    max_tokens=max_tokens, seed=args.seed, stop=custom_stop,
                    timeout=args.api_timeout,
                )
                return strip_harmony_channels(resp.choices[0].text or "")
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        print(f"[Error] LLM server call failed after {args.api_retries} attempts: {last_err}")
        return ""

    def llm(prompt, stop=["\n"]):
        """ReAct generation via the OpenAI-compatible server."""
        return _api_complete(prompt, stop, max_tokens=100)

    def isolated_generate(prompt):
        """Short, deterministic generation used for ReliabilityRAG isolated answering."""
        return _api_complete(prompt, stop=["\n"], max_tokens=args.isolated_max_tokens)

else:
    from vllm import LLM, SamplingParams

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

nli_device = args.nli_device or default_nli_device


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

env = E5WikiEnv(retriever, k=5, max_docs_per_docid=args.max_docs_per_docid)
# Legacy datasets keep their original wrapper classes (and data-file naming
# convention) for exact back-compat; any other --dataset (e.g. the mhweb_*
# datasets from dataset_builder/05_build_corpus.py) goes through the generic
# JsonQAWrapper, which expects data/{dataset}_dev_simplified.json.
if args.dataset == "hotpotqa":
    env = wrappers.HotPotQAWrapper(env, split="dev")
elif args.dataset == "musique":
    env = wrappers.musiqueAWrapper(env, split="dev")
elif args.dataset == "2wikimultihopqa":
    env = wrappers.wikimultihopqaWrapper(env, split="dev")
else:
    env = wrappers.JsonQAWrapper(env, dataset=args.dataset, split="dev")
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
    # parallel accumulators for the NLI-only (original ReliabilityRAG) view
    per_index_weakness_nli = defaultdict(list)
    per_index_excluded_nli = defaultdict(list)
    per_index_density_nli = defaultdict(list)
    per_index_missize_nli = defaultdict(list)
    weakest_index_counter = Counter()            # per-question weakest loop index
    weakest_index_counter_nli = Counter()        # same, NLI-only metrics
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
                per_index_weakness_nli[si].append(met['weakness_nli_only'])
                per_index_excluded_nli[si].append(met['excluded_fraction_nli_only'])
                per_index_density_nli[si].append(met['edge_density_nli_only'])
                per_index_missize_nli[si].append(met['mis_size_nli_only'])
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
            weakest_nli = max(q_search_metrics,
                              key=lambda t: (t[1]['excluded_fraction_nli_only'],
                                             t[1]['edge_density_nli_only'], -t[0]))
            weakest_index_counter_nli[weakest_nli[0]] += 1

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
            # original ReliabilityRAG view (NLI edges only, unknowns neutral)
            'mean_weakness_nli_only': round(sum(per_index_weakness_nli[si]) / len(per_index_weakness_nli[si]), 4),
            'mean_excluded_fraction_nli_only': round(sum(per_index_excluded_nli[si]) / len(per_index_excluded_nli[si]), 4),
            'mean_edge_density_nli_only': round(sum(per_index_density_nli[si]) / len(per_index_density_nli[si]), 4),
            'mean_mis_size_nli_only': round(sum(per_index_missize_nli[si]) / len(per_index_missize_nli[si]), 4),
        }

    # overall weakest loop index by mean excluded_fraction (require >=3 support)
    eligible = [(si, s) for si, s in per_index_summary.items() if s['n_questions'] >= 3]
    weakest_overall = (max(eligible, key=lambda t: t[1]['mean_excluded_fraction'])[0]
                       if eligible else None)
    weakest_overall_nli = (max(eligible, key=lambda t: t[1]['mean_excluded_fraction_nli_only'])[0]
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
        'weakest_loop_index_overall_nli_only': weakest_overall_nli,
        'aggregation_B_per_question_weakest_loop_distribution':
            {str(k): v for k, v in sorted(weakest_index_counter.items())},
        'aggregation_B_per_question_weakest_loop_distribution_nli_only':
            {str(k): v for k, v in sorted(weakest_index_counter_nli.items())},
        'time_sec': round(time.time() - start_time, 1),
    }
    summary_path = os.path.join(output_dir, f'{args.dataset}_{m}_mis_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("ReliabilityRAG MIS diagnostic (clean) complete")
    print(f"Questions: {len(per_question)} | time {summary['time_sec']}s")
    print("\n(A) Mean weakness per loop index  [weakness = 1 - MIS/k]:")
    print("      (W=unknown-aware, W_nli=NLI-only / original ReliabilityRAG)")
    for si, s in per_index_summary.items():
        print(f"  loop {si:>2}: W={s['mean_weakness']:.3f} "
              f"W_nli={s['mean_weakness_nli_only']:.3f} "
              f"excluded={s['mean_excluded_fraction']:.3f} "
              f"density={s['mean_edge_density']:.3f} "
              f"mis={s['mean_mis_size']:.2f}  (n={s['n_questions']})")
    print(f"\nWeakest loop index overall (mean excluded_fraction, n>=3): "
          f"{weakest_overall}  (NLI-only: {weakest_overall_nli})")
    print("\n(B) Per-question weakest-loop distribution [unknown-aware | NLI-only]:")
    for si in sorted(set(weakest_index_counter) | set(weakest_index_counter_nli)):
        print(f"  loop {si:>2}: {weakest_index_counter.get(si, 0)} | "
              f"{weakest_index_counter_nli.get(si, 0)} questions")
    print(f"\nSaved: {summary_path}")
    print("=" * 60)
