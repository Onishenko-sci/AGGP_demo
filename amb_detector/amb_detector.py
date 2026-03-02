import os
import spacy
import re
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from functools import lru_cache
import json
import nltk
import yaml
import pickle

from .llm import llm_response

import nltk
import os

EMBEDING_CACHE_FILE = "./amb_detector/embedding_cache.pkl"

# C:\Users\Anatoly\AppData\Roaming\nltk_data
# Автоопределение пути NLTK
nltk.download('wordnet')

# try:
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     nltk.download('wordnet')

nlp = spacy.load("en_core_web_trf")

def load_config(config_path="./amb_detector/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

matching_config = config["matching"]
weight_map = config["weights"]
manual_synonyms = config["manual_synonyms"]

embedding_cache = {}

def load_embedding_cache(cache_file=EMBEDING_CACHE_FILE):
    """Загружает кеш эмбеддингов из pkl файла"""
    global embedding_cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                embedding_cache = pickle.load(f)
            print(f"Loaded {len(embedding_cache)} embeddings from cache")
        except Exception as e:
            print(f"Failed to load embedding cache: {e}")
            embedding_cache = {}
    else:
        print("No embedding cache found, starting fresh")
        embedding_cache = {}

def save_embedding_cache(cache_file="./amb_detector/embedding_cache.pkl"):
    """Сохраняет кеш эмбеддингов в pkl файл"""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding_cache, f)
        print(f"Saved {len(embedding_cache)} embeddings to cache")
    except Exception as e:
        print(f"Failed to save embedding cache: {e}")


# Загружаем кеш эмбеддингов при импорте модуля
def _init_cache():
    load_embedding_cache()

try:
    _init_cache()
except Exception as e:
    print(f"Warning: Could not initialize embedding cache: {e}")

def preprocess_text(text):
    doc = nlp(text)
    return doc

def extract_tokens_info(doc):
    tokens_info = []
    for token in doc:
        tokens_info.append({
            'text': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'dep': token.dep_,
            'head': token.head.text,
            'is_stop': token.is_stop
        })
    return tokens_info

def identify_predicates_from_doc(doc, debug = False):
    predicates = []
    seen = set()

    STRONG_DEPS = {
        'ROOT', 'dobj', 'nsubj', 'nsubjpass', 'attr',
        'acl', 'relcl', 'advcl', 'xcomp', 'ccomp', 'conj'
    }
    LIGHT_VERBS = {'make', 'take', 'have', 'give', 'do'}
    SPATIAL_PREPS = {'on', 'in', 'at', 'under', 'over', 'beside', 'left', 'right'}

    for token in doc:
        if token.pos_ == 'ADP' and token.lemma_ in SPATIAL_PREPS:
            if debug:
                print(f"Skip spatial preposition: {token.text}")
            continue

        if token.pos_ not in {'VERB', 'NOUN', 'ADJ'}:
            if debug:
                print(f"Skip POS: {token.text} ({token.pos_})")
            continue

        # Nouns/adjectives with dobj/nsubj dep are actants, not predicates —
        # they should be matched against environment in extract_actants_with_embeddings
        if token.pos_ in {'NOUN', 'ADJ'} and token.dep_ in {'dobj', 'nsubj', 'nsubjpass'}:
            if debug:
                print(f"Skip noun/adj actant: {token.text} ({token.dep_})")
            continue

        if token.dep_ not in STRONG_DEPS:
            if debug:
                print(f"Skip DEP: {token.text} ({token.dep_})")
            continue

        key = (token.lemma_.lower(), token.dep_, token.head.text)
        if key in seen:
            continue
        seen.add(key)

        particle = next((c.text for c in token.children if c.dep_ == 'prt'), None)
        full_text = f"{token.text} {particle}" if particle else token.text

        modifiers = []
        if any(c.dep_ == 'neg' for c in token.children):
            modifiers.append('negation')
        if any(c.dep_ == 'aux' for c in token.children):
            modifiers.append('auxiliary')

        if token.lemma_ in LIGHT_VERBS:
            for child in token.children:
                if child.dep_ in {'dobj', 'attr'} and child.pos_ == 'NOUN':
                    phrase = f"{token.text} {child.text}"
                    key2 = (phrase.lower(), token.dep_)
                    if key2 not in seen:
                        seen.add(key2)
                        predicates.append({
                            'text': phrase,
                            'lemma': f"{token.lemma_} {child.lemma_}",
                            'pos': 'VERB+NOUN',
                            'dep': token.dep_,
                            'head': token.head.text,
                            'is_predicate': True,
                            'modifiers': modifiers.copy(),
                            'clause_type': token.dep_ if token.dep_ in {'relcl','advcl','xcomp','ccomp'} else None
                        })

        predicates.append({
            'text': full_text,
            'lemma': token.lemma_.lower(),
            'pos': token.pos_,
            'dep': token.dep_,
            'head': token.head.text,
            'is_predicate': True,
            'modifiers': modifiers,
            'clause_type': token.dep_ if token.dep_ in {'relcl','advcl','xcomp','ccomp'} else None
        })

    for token in doc:
        if token.dep_ == 'conj' and token.pos_ in {'VERB','NOUN','ADJ'}:
            key = (token.lemma_.lower(), token.dep_, token.head.text)
            if key in seen:
                continue
            seen.add(key)
            particle = next((c.text for c in token.children if c.dep_=='prt'), None)
            full_text = f"{token.text} {particle}" if particle else token.text
            modifiers = []
            if any(c.dep_=='neg' for c in token.children): modifiers.append('negation')
            if any(c.dep_=='aux' for c in token.children): modifiers.append('auxiliary')
            predicates.append({
                'text': full_text,
                'lemma': token.lemma_.lower(),
                'pos': token.pos_,
                'dep': token.dep_,
                'head': token.head.text,
                'is_predicate': True,
                'modifiers': modifiers,
                'clause_type': None
            })
    return predicates

import os
import requests

def get_emb(text):
    api_key = os.getenv('OPEN_ROUTER_KEY')
    if not api_key:
        raise ValueError("OPEN_ROUTER_KEY environment variable is not set")
    for attempt in range(5):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/text-embedding-3-large",
                    "input": text
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            embedding = data["data"][0]["embedding"]
            return embedding
        except Exception as e:
            print(f"OpenRouter Embedding API error (attempt {attempt+1}): {e}")
            import time; time.sleep(2)
    raise ValueError(f"Failed to get embedding for: {text}")

def get_embedding(text: str):
    if text in embedding_cache:
        return embedding_cache[text]
    tokens = [t.lemma_ for t in nlp(text.lower()) if not t.is_stop]
    normalized = " ".join(tokens)
    if normalized in embedding_cache:
        print(f"Embedding from cache: {normalized}")
        return embedding_cache[normalized]
    print(f"Embedding: {normalized}")
    emb = get_emb(normalized)
    embedding_cache[text] = emb
    embedding_cache[normalized] = emb
    return emb

@lru_cache(maxsize=10000, typed=True)
def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word, pos=wn.NOUN)[:3]:
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if name != word:
                synonyms.add(name)
            if len(synonyms) >= 10:
                break
        if len(synonyms) >= 10:
            break
    manual = manual_synonyms.get(word.lower(), [])
    synonyms.update(m.lower() for m in manual)
    return list(synonyms)[:10]

def get_np_modifiers(span):
    return [t.lemma_.lower() for t in span if t.pos_ == 'ADJ' or t.dep_ == 'amod']

def prepare_environment(environment_list):
    processed = []
    for phrase in environment_list:
        phrase_doc = nlp(phrase)
        lemmas = tuple(t.lemma_.lower() for t in phrase_doc if t.pos_ in {"NOUN", "PROPN", "ADJ"} and not t.is_stop)
        modifiers = get_np_modifiers(phrase_doc)
        embedding = get_embedding(phrase)
        processed.append({
            'original': phrase,
            'lower': phrase.lower(),
            'lemmas': lemmas,
            'modifiers': modifiers,
            'embedding': embedding
        })
    return processed

def normalize_np_text(text):
    return re.sub(r'^(the|a|an)\s+', '', text.lower())

def find_matches(np_text, np_lemmas, np_embedding, env_processed):
    matches = set()
    match_sources = {}

    np_text_norm = normalize_np_text(np_text)

    env_embeddings = np.stack([e['embedding'] for e in env_processed])

    # 1. Full text match
    if matching_config['use_text_match']:
        for env in env_processed:
            env_text_norm = normalize_np_text(env['lower'])
            if np_text_norm == env_text_norm:
                matches.add(env['original'])
                match_sources[env['original']] = 'text_match'

    # 2. Lemma overlap
    if matching_config['use_lemma_overlap']:
        for env in env_processed:
            if set(np_lemmas) & set(env['lemmas']):
                sim = cosine_similarity([np_embedding], [env['embedding']])[0][0]
                if sim > 0.4:
                    matches.add(env['original'])
                    match_sources[env['original']] = 'lemma_overlap'

    # 3. Embedding match
    if matching_config['use_embedding']:
        scores = cosine_similarity([np_embedding], env_embeddings)[0]
        for i, score in enumerate(scores):
            env = env_processed[i]
            threshold = max(0.4, matching_config['embedding_threshold_base'] - matching_config['embedding_threshold_length_factor'] * len(env['original'].split()))
            if score > threshold:
                matches.add(env['original'])
                match_sources[env['original']] = 'embedding'

    # 4. Synonym embedding match
    if matching_config['use_wordnet']:
        synonyms = set()
        for lemma in np_lemmas:
            synonyms.update(get_synonyms(lemma))
        for synonym in synonyms:
            syn_emb = get_embedding(synonym)
            scores = cosine_similarity([syn_emb], env_embeddings)[0]
            for i, score in enumerate(scores):
                if score > matching_config['synonym_threshold']:
                    env = env_processed[i]
                    matches.add(env['original'])
                    match_sources[env['original']] = 'synonym_embedding'

    # 5. Manual synonym match
    if matching_config['use_manual']:
        manual = manual_synonyms.get(np_text, [])
        for m in manual:
            for env in env_processed:
                if normalize_np_text(m) == normalize_np_text(env['lower']):
                    matches.add(env['original'])
                    match_sources[env['original']] = 'manual'

    return list(matches), match_sources


def extract_actants_with_embeddings(instruction_text, predicates, environment_list, doc=None, debug=False):
    if doc is None:
        doc = nlp(instruction_text)

    result = {}
    visited = set()
    env_processed = prepare_environment(environment_list)

    def get_np_lemma_tuple(span):
        return tuple(token.lemma_.lower() for token in span if not token.is_stop)

    modifiers_tokens = set()

    for pred in predicates:
        pred_token = next((t for t in doc if t.lemma_.lower() == pred['lemma'].lower() and t.pos_ in {'VERB', 'ADJ'}), None)
        if not pred_token:
            continue

        actants = []

        for chunk in doc.noun_chunks:
            # Check if this noun chunk is governed by the predicate,
            # either directly or through a chain of prepositions/ancestors
            is_governed = (chunk.root.head == pred_token or
                           pred_token in list(chunk.root.ancestors))
            if debug:
                print(f"  Chunk '{chunk.text}' root='{chunk.root.text}' head='{chunk.root.head.text}' "
                      f"ancestors={[a.text for a in chunk.root.ancestors]} governed_by_{pred['lemma']}={is_governed}")
            if not is_governed:
                continue

            prep = None
            pobj = None

            for child in chunk.root.children:
                if child.dep_ == 'prep':
                    prep = child
                    for pobj_candidate in child.children:
                        if pobj_candidate.dep_ == 'pobj':
                            pobj = pobj_candidate
                            break

            if prep and pobj:
                end = max(chunk.end, pobj.i + 1)
                extended_span = doc[chunk.start:end]
            else:
                extended_span = chunk

            full_np_text = extended_span.text.lower().strip()
            full_np_lemmas = get_np_lemma_tuple(extended_span)
            full_np_embedding = get_embedding(full_np_text)

            modifiers = [token for token in extended_span if token.dep_ in {'amod', 'compound', 'det', 'nummod'} and token != extended_span.root]
            for mod_token in modifiers:
                modifiers_tokens.add(mod_token.i)

            visited.update(range(extended_span.start, extended_span.end))

            explicit_env_match = [
                e for e in environment_list
                if normalize_np_text(e) in normalize_np_text(full_np_text)
            ]

            if explicit_env_match:
                matches = explicit_env_match
                match_sources = {explicit_env_match[0]: 'explicit_modifier'}
                is_ambiguous = False
            else:
                exact_matches = [e for e in environment_list if normalize_np_text(e) == normalize_np_text(full_np_text)]

                if exact_matches:
                    matches = exact_matches
                    match_sources = {exact_matches[0]: 'exact_match'}
                    is_ambiguous = False
                else:
                    matches, match_sources = find_matches(full_np_text, full_np_lemmas, full_np_embedding, env_processed)
                    is_ambiguous = len(matches) > 1

            confidence = max([weight_map.get(src, 0.5) for src in match_sources.values()], default=0)

            actants.append({
                'text': extended_span.text,
                'lemma': full_np_text,
                'dep': extended_span.root.dep_,
                'matches': matches,
                'match_sources': match_sources,
                'modifiers': [t.text for t in modifiers],
                'is_ambiguous': is_ambiguous,
                'is_in_environment': bool(matches),
                'confidence': confidence,
                'is_full_np_match': True,
                'range': (extended_span.start_char, extended_span.end_char),
            })

            if debug:
                print(f"NP match: {extended_span.text} → {matches} | modifiers: {[t.text for t in modifiers]}")

        for token in doc:
            if token.i in visited or token.is_stop or token.pos_ not in {'NOUN', 'PROPN', 'ADJ'}:
                continue
            if token.dep_ in {'amod', 'compound', 'det', 'nummod'} or token.i in modifiers_tokens:
                continue

            lemma = token.lemma_.lower()
            token_emb = get_embedding(lemma)
            matches, match_sources = find_matches(lemma, [lemma], token_emb, env_processed)
            confidence = max([weight_map.get(src, 0.5) for src in match_sources.values()], default=0)

            if matches:
                actants.append({
                    'text': token.text,
                    'lemma': lemma,
                    'dep': token.dep_,
                    'matches': matches,
                    'match_sources': match_sources,
                    'modifiers': [],
                    'is_ambiguous': len(matches) > 1,
                    'is_in_environment': True,
                    'confidence': confidence,
                    'is_full_np_match': False,
                    'range': (token.idx, token.idx + len(token.text)),
                })

                visited.add(token.i)

                if debug:
                    print(f"Token match: {token.text} → {matches}")

        filtered_actants = []
        for a in actants:
            if not a.get("matches"):
                continue
            a_range = a.get("range")
            a_matches = set(a["matches"])
            is_subsumed = any(
                b != a
                and b.get("range")
                and a_range[0] >= b["range"][0]
                and a_range[1] <= b["range"][1]
                and set(b["matches"]) >= a_matches
                for b in actants
            )
            if not is_subsumed:
                filtered_actants.append(a)

        result[pred['lemma'].lower()] = filtered_actants

    return result

def build_prompt_from_actants(actants_dict, environment_list, template_path="amb_detector/prompts/actant_prompt.txt"):
    actant_entries = []
    for actants in actants_dict.values():
        for a in actants:
            if not a["is_in_environment"]:
                continue
            entry = a["lemma"]
            if a.get("modifiers"):
                entry = " ".join(a["modifiers"]) + " " + entry
            actant_entries.append(entry.strip())

    if not actant_entries:
        return "No actants matched with the environment."

    actant_text = ", ".join(sorted(set(actant_entries)))
    environment_text = ", ".join(environment_list)
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    prompt = template.replace("{{actants}}", actant_text).replace("{{environment}}", environment_text)

    return prompt

def parse_llm_json_or_blocks(text, environment_list=None, actants_dict=None, manual_synonyms=None):
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        return data
    except json.JSONDecodeError:
        pass  

    results = []
    current_block = {}

    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue

        variant_match = re.match(r"^-?\s*(.+?):\s*$", line)
        if variant_match:
            if current_block:
                if 'variant' in current_block:
                    current_block['variant'] = current_block['variant'].strip().rstrip(':')
                results.append(current_block)
                current_block = {}

            current_block['variant'] = variant_match.group(1).strip().rstrip(':')
            continue

        key_val_match = re.match(r"^([a-zA-Z_]+)\s*:\s*(.+)$", line)
        if key_val_match:
            key = key_val_match.group(1).strip()
            val = key_val_match.group(2).strip()

            if ',' in val:
                val_list = [v.strip() for v in val.split(',')]
                current_block[key] = val_list
            else:
                current_block[key] = val
            continue

    if current_block:
        if 'variant' in current_block:
            current_block['variant'] = current_block['variant'].strip().rstrip(':')
        results.append(current_block)

    return results

def build_variants_from_matches(actants_dict):
    variants = {}
    for actants in actants_dict.values():
        for a in actants:
            if not a['is_in_environment']:
                continue
            if a['is_ambiguous']:
                variants[a['lemma']] = a['matches']
            else:
                variants[a['lemma']] = "no variations"
    return variants

def group_variants(parsed_blocks):
    grouped = defaultdict(list)
    for entry in parsed_blocks:
        np_text = entry.get("np_text") or entry.get("actant")  
        if np_text:
            grouped[np_text].append(entry)
    return dict(grouped)

def get_variants_from_llm(actants_dict, environment_list, manual_synonyms=None, use_llm=True):
    if manual_synonyms is None:
        manual_synonyms = {}

    prompt = build_prompt_from_actants(actants_dict, environment_list)

    if not use_llm:
        return build_variants_from_matches(actants_dict)

    print("LLM call from amb_detector")
    try:
        response_text = llm_response(
            system_prompt="You are an expert in semantic disambiguation. Generate variants of the given object based on context.",
            user_prompt=prompt,
            max_tokens=300
        ).strip()

        parsed_blocks = parse_llm_json_or_blocks(response_text)
        return group_variants(parsed_blocks)


    except Exception as e:
        raise ValueError(f"!!!LLM failed {e}")
        return build_variants_from_matches(actants_dict)

def get_combined_variants_from_actants(actants_dict, environment_list, use_llm=True, manual_synonyms={}):

    # 1. Rule-based
    rule_variants = build_variants_from_matches(actants_dict)
    combined_variants = dict(rule_variants)  # копия
    match_sources = {k: ['rule'] if v != "no variations" else [] for k, v in rule_variants.items()}

    # 2. LLM-based
    if use_llm:
        llm_variants = get_variants_from_llm(
            actants_dict,
            environment_list,
            manual_synonyms=manual_synonyms,
            use_llm=True
        )
        for k, v in llm_variants.items():
            if v == "no variations":
                continue
            if k not in combined_variants or combined_variants[k] == "no variations":
                combined_variants[k] = v
                match_sources[k] = ['llm']
            else:
                merged = list(set(combined_variants[k]) | set(v))
                combined_variants[k] = merged
                match_sources[k].append('llm')

    for k, v in combined_variants.items():
        if v == []:
            combined_variants[k] = "no variations"

    return combined_variants, match_sources

def update_ambiguity_flags(actants_dict):
    for pred, actants in actants_dict.items():
        for actant in actants:
            matches = actant.get("matches", [])
            actant["is_ambiguous"] = len(matches) > 1

def build_frame_prompt(lemma, variants, environment, template_path="amb_detector/prompts/frame_prompt.txt"):
    if not variants:
        return "No variants provided."

    variants_str = ", ".join(variants)
    environment_str = ", ".join(environment)

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    prompt = (
        template
        .replace("{{lemma}}", lemma)
        .replace("{{variants}}", variants_str)
        .replace("{{environment}}", environment_str)
    )

    return prompt

def activate_frames_llm(variants_dict, environment_list):
    frames_per_np = {}

    for np_text, variants in variants_dict.items():
        if variants == "no variations" or not variants:
            frames_per_np[np_text] = []
            continue

        if isinstance(variants[0], dict):
            variant_names = [v.get("variant") or v.get("text") or str(v) for v in variants]
        else:
            variant_names = variants

        relevant_env = [obj for obj in environment_list if obj in variant_names]
        prompt = build_frame_prompt(np_text, variant_names, relevant_env)

        try:
            response_text = llm_response(
                system_prompt="You are a cognitive linguistics assistant.",
                user_prompt=prompt,
                max_tokens=300
            ).strip()

            frames_per_np[np_text] = parse_llm_json_or_blocks(response_text)

        except Exception as e:
            raise ValueError(f"!!!Error in activationg frames for {np_text}: {e}")
            frames_per_np[np_text] = []

    return frames_per_np


def detect_conflict_or_variability(frames_per_predicate):
    predicates_info = {
        "conflict": {},     
        "variability": {},  
    }

    for pred, frames in frames_per_predicate.items():
        if not frames or not isinstance(frames, list):
            continue

        frame_names = set()
        valency_sets = set()
        variants = set()

        for frame_info in frames:
            frame_name = frame_info.get("frame")
            valency = tuple(sorted(frame_info.get("valency", [])))
            variant = frame_info.get("variant")

            if frame_name:
                frame_names.add(frame_name)
            if valency:
                valency_sets.add(valency)
            if variant:
                variants.add(variant)

        if len(frame_names) > 1:
            predicates_info["conflict"][pred] = [
                {"variant": f.get("variant"), "frame": f.get("frame")}
                for f in frames
            ]
        elif len(frame_names) == 1 and len(variants) > 1:
            predicates_info["variability"][pred] = list(sorted(variants))

    return predicates_info

def is_intent_aligned_cosine(intents, variants, embedding_model=None, threshold=0.6):
    if not intents or not variants:
        return False

    intent_embeddings = [get_embedding(t) for t in intents]
    variant_embeddings = [get_embedding(t) for t in list(variants)]

    sims = cosine_similarity(intent_embeddings, variant_embeddings)
    max_sim = sims.max() if sims.size > 0 else 0
    return max_sim >= threshold

def extract_strings(value):
    if isinstance(value, list):
        return [v.lower() for v in value if isinstance(v, str)]
    elif isinstance(value, dict):
        result = []
        for subval in value.values():
            result.extend(extract_strings(subval))
        return result
    else:
        return []

def run_disambiguation_pipeline(instruction_text, environment_list, use_llm=True, debug=False):
    print(f"[amb_detector] Starting disambiguation pipeline for: '{instruction_text}'")
    print(f"[amb_detector] Environment ({len(environment_list)} objects): {environment_list}")
    # 1
    doc = preprocess_text(instruction_text)
    if debug:
        print("Doc tokens:", [token.text for token in doc])
    
    # 2
    predicates = identify_predicates_from_doc(doc, debug=debug)
    print(f"[amb_detector] Predicates found: {[p['text'] for p in predicates]}")
    if debug:
        print("Predicates found:", predicates)
    
    # 3
    actants_dict = extract_actants_with_embeddings(instruction_text, predicates, environment_list, doc=doc, debug=debug)
    for pred, actants in actants_dict.items():
        print(f"[amb_detector] Actants for '{pred}': {[a['text'] + ' -> ' + str(a['matches']) for a in actants]}")
    if debug:
        for pred, actants in actants_dict.items():
            print(f"Actants for predicate '{pred}':")
            for a in actants:
                print(" ", a)
    
    # 4
    update_ambiguity_flags(actants_dict)
    ambiguous_actants = [a['text'] for actants in actants_dict.values() for a in actants if a.get('is_ambiguous')]
    print(f"[amb_detector] Ambiguous actants: {ambiguous_actants if ambiguous_actants else 'none'}")
    
    # 5
    has_ambiguity = any(
        a.get("is_ambiguous", False)
        for actants in actants_dict.values()
        for a in actants
    )

    if use_llm and has_ambiguity:
        variants = get_variants_from_llm(actants_dict, environment_list, manual_synonyms=manual_synonyms, use_llm=True)
    else:
        variants = build_variants_from_matches(actants_dict)

    variants, match_sources = get_combined_variants_from_actants(
        actants_dict,
        environment_list,
        use_llm=use_llm
    )
    print(f"[amb_detector] Combined variants: {variants}")
    print(f"[amb_detector] Match sources: {match_sources}")
  
    if debug:
        print("Variants from pipeline:")
        print(variants)
    
    # 6
    print(f"[amb_detector] Activating frames via LLM...")
    frames_per_predicate = activate_frames_llm(variants, environment_list)
    print(f"[amb_detector] Frames per predicate: {frames_per_predicate}")

    if debug:
        print("Frames per predicate:")
        for pred, frames in frames_per_predicate.items():
            print(f" {pred}: {frames}")
    
    # 7
    predicates_info = detect_conflict_or_variability(frames_per_predicate)
    print(f"[amb_detector] Conflicts: {predicates_info['conflict'] if predicates_info['conflict'] else 'none'}")
    print(f"[amb_detector] Variability: {predicates_info['variability'] if predicates_info['variability'] else 'none'}")
    if debug:
        print("Predicates info (conflict / variability):")
        print(predicates_info)
    
    # Сохраняем кеш эмбеддингов после обработки
    save_embedding_cache()
    
    return {
        "predicates": predicates,
        "actants": actants_dict,
        "variants": variants,
        "match_sources": match_sources,
        "frames": frames_per_predicate,
        "predicates_info": predicates_info
    }

def process_dataset(df):
    results = []
    debug_logs = []
    print("Process dataset")
    for _, row in tqdm(df.iterrows(), total=len(df)): 
        print("Process dataset")
        instruction = row['task']
        environment = [x.strip() for x in row['environment_short'].split(',')]
        print(environment)
        detection_result = run_disambiguation_pipeline(instruction, environment)

        # HR (Help Rate)
        pred_info = detection_result.get('predicates_info', {})
        conflict = pred_info.get('conflict') or {}
        variability = pred_info.get('variability') or {}
        ambiguity = {**conflict, **variability}
        hr = int(bool(conflict or variability))

        # CHR (Correct Help Rate)
        if row['ambiguity_type'] in ['unambiguous_task', 'singlelabel_task', 'creative_singlelabel_task', 'unsafe_task' ]:
            chr = 1 if hr == 0 else 0
        else:
            chr = 1 if hr == 1 else 0

        # IA (Intent Alignment)
        user_intent_raw = row['user_intent'] if pd.notna(row['user_intent']) else ''
        user_intents = re.split(r'[\|,\-\/&]', user_intent_raw.lower())
        user_intents = [x.strip() for x in user_intents if x.strip()]

        found_variants = set()
        for variants in ambiguity.values():
            found_variants.update(extract_strings(variants))

        if found_variants and user_intents:
            intent_aligned = is_intent_aligned_cosine(user_intents, found_variants)
            ia_val = 1 if intent_aligned else 0
        else:
            ia_val = np.nan 

        results.append({
            'ambiguity_type': row['ambiguity_type'],
            'HR': hr,
            'CHR': chr,
            'IA': ia_val
        })
        
    return pd.DataFrame(results), _

if __name__ == '__main__':
    introplan_dataset = pd.read_csv('amb_detector/data/introplan_mobile.csv')

    results_df, _ = process_dataset(introplan_dataset)

    final_report = results_df.groupby('ambiguity_type').agg(
        HR=('HR', 'mean'),
        CHR=('CHR', 'mean'),
        IA=('IA', lambda x: x[~x.isna()].mean())
    ).reset_index()

    print(final_report)
    
    # Финальное сохранение кеша после выполнения
    save_embedding_cache()


# ambik_dataset = pd.read_csv('data/ambik.csv')
# ambik_dataset = ambik_dataset[['ambiguity_type', 'environment_short', 'unambiguous_direct', 'ambiguous_task', 'user_intent']]
# amb = ambik_dataset[['environment_short', 'ambiguity_type', 'ambiguous_task', 'user_intent']]
# ambik_dataset.ambiguity_type = ['unambiguous_direct']*len(ambik_dataset)
# ambik_dataset = pd.concat([ambik_dataset, amb])

# def process_dataset(df):
#     results = []

#     for _, row in tqdm(df.iterrows(), total=len(df)):
#         if row['ambiguity_type'] == 'unambiguous_direct':
#             instruction = row['unambiguous_direct']
#         else:
#             instruction = row['ambiguous_task']

#         environment = [x.strip() for x in row['environment_short'].split(',')]

#         detection_result = run_disambiguation_pipeline(instruction, environment)

#         # HR
#         pred_info = detection_result.get('predicates_info', {})
#         conflict = pred_info.get('conflict') or {}
#         variability = pred_info.get('variability') or {}
#         ambiguity = {**conflict, **variability}
#         hr = int(bool(conflict or variability))

#         # CHR
#         if row['ambiguity_type'] == 'preferences':
#             chr = 1 if hr == 1 else 0
#         else:
#             chr = 1 if hr != 1 else 0

#         # IA (Intent Alignment)
#         if hr == 1:
#             user_intent_raw = row['user_intent'] if pd.notna(row['user_intent']) else ''
#             user_intents = re.split(r'[\|,\-\/&]', user_intent_raw.lower())
#             user_intents = [x.strip() for x in user_intents if x.strip()]

#             found_variants = set()
#             for variants in ambiguity.values():
#                 found_variants.update(extract_strings(variants))

#             intent_aligned = is_intent_aligned_cosine(user_intents, found_variants, model)
#             ia = 1 if intent_aligned else 0
#         else:
#             ia = np.nan  # Не вычисляем IA для hr == 0


#         results.append({
#             'ambiguity_type': row['ambiguity_type'],
#             'HR': hr,
#             'CHR': chr,
#             'IA': ia
#         })

#     return pd.DataFrame(results), _

# results_df, _ = process_dataset(ambik_dataset)

# final_report = results_df.groupby('ambiguity_type').agg(
#     HR=('HR', 'mean'),
#     CHR=('CHR', 'mean'),
#     IA=('IA', lambda x: x[~x.isna()].mean())  # Считается только по не-NaN, т.е. hr == 1
# ).reset_index()

# print(final_report)
