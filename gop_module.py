#!/usr/bin/env python3
"""
Portable GOP (Goodness of Pronunciation) module for Kannada speech assessment.
Modified to use relative paths and work in containerized environments.
"""

import os
import re
import subprocess
import time
import argparse
import shutil
import multiprocessing
from typing import List, Dict, Tuple, Optional

# -------- Helper to run shell commands --------
def run(cmd, log=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sourced_cmd = f"cd {base_dir} && source {base_dir}/path.sh && {cmd}"
    print(f"[RUN] {sourced_cmd}")
    result = subprocess.run(sourced_cmd, shell=True, executable='/bin/bash')
    if result.returncode != 0:
        print(f"Command failed: {sourced_cmd}")
        raise RuntimeError(f"Command failed: {sourced_cmd}")


_NON_WORD_PATTERN = re.compile(r"[^0-9a-z\u0C80-\u0CFF'\s]")


def _normalize_transcript(transcript: str) -> str:
    """Lowercase and strip unsupported symbols from the transcript."""
    if transcript is None:
        return ""
    cleaned = _NON_WORD_PATTERN.sub(" ", transcript.lower())
    return " ".join(cleaned.split())

class GOPProcessor:
    """Singleton to cache frequently used Kaldi paths."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
            self.MODEL_DIR = os.path.join(self.base_dir, "models", "vosk_kannada_model")
            self.LANG_DIR = os.path.join(self.MODEL_DIR, "graph")
            self.IVECTOR_DIR = os.path.join(self.MODEL_DIR, "ivector")
            self.LM_DIR = os.path.join(self.base_dir, "models", "LM_2gram_aiish")
            self.LEXICON_SOURCES = f"{self.LM_DIR}/lexicon.txt {self.LANG_DIR}/phones/align_lexicon.txt"
            self.WORKDIR = os.path.join(self.base_dir, "gop_work_tmp")
            os.makedirs(self.WORKDIR, exist_ok=True)
            GOPProcessor._initialized = True


def compute_gop_batch(audio_transcript_pairs: List[Tuple[str, str]], nj: Optional[int] = None) -> List[Dict]:
    """Run Kaldi GOP pipeline for multiple wav/transcript pairs in one batch."""

    if not audio_transcript_pairs:
        raise ValueError("audio_transcript_pairs must contain at least one item")

    cpu_count = multiprocessing.cpu_count() or 1
    if nj is None:
        nj = min(len(audio_transcript_pairs), cpu_count)
    nj = max(1, min(nj, cpu_count))

    processor = GOPProcessor()
    batch_id = f"batch_{int(time.time() * 1000)}"
    workdir = os.path.join(processor.WORKDIR, batch_id)
    data_dir = os.path.join(workdir, "data")
    ali_dir = os.path.join(workdir, "ali")
    probs_dir = os.path.join(workdir, "probs")
    gop_dir = os.path.join(workdir, "gop")

    for d in [data_dir, ali_dir, probs_dir, gop_dir]:
        os.makedirs(d, exist_ok=True)

    start_time = int(time.time() * 1000)
    utt_ids: List[str] = []

    try:
        # -------- Prepare combined Kaldi data dir --------
        with open(os.path.join(data_dir, "wav.scp"), "w") as wav_f, \
             open(os.path.join(data_dir, "text"), "w") as text_f, \
             open(os.path.join(data_dir, "utt2spk"), "w") as utt2spk_f:

            for wav_path, transcript in audio_transcript_pairs:
                utt_id = os.path.splitext(os.path.basename(wav_path))[0]
                utt_ids.append(utt_id)

                wav_f.write(f"{utt_id} {wav_path}\n")
                normalized_transcript = _normalize_transcript(transcript)
                if not normalized_transcript:
                    raise ValueError(f"Transcript is empty after normalization for {utt_id}")
                text_f.write(f"{utt_id} {normalized_transcript}\n")
                utt2spk_f.write(f"{utt_id} {utt_id}\n")

        run(f"utils/utt2spk_to_spk2utt.pl < {data_dir}/utt2spk > {data_dir}/spk2utt")

        # -------- Feature extraction --------
        run(f"steps/make_mfcc.sh --nj {nj} --mfcc-config conf/mfcc_hires.conf {data_dir}")
        run(f"steps/compute_cmvn_stats.sh {data_dir}")
        run(f"utils/fix_data_dir.sh {data_dir}")

        run(
            f"steps/online/nnet2/extract_ivectors.sh --nj {nj} "
            f"{data_dir} {processor.LM_DIR} {processor.IVECTOR_DIR} "
            f"{data_dir}/ivectors/ivectors_offline"
        )

        # -------- Network outputs --------
        run(
            f"steps/nnet3/compute_output.sh --nj {nj} "
            f"--online-ivector-dir {data_dir}/ivectors/ivectors_offline "
            f"{data_dir} {processor.MODEL_DIR}/am {probs_dir}"
        )

        # -------- Phone-level transcripts --------
        run(f"make_text_phone.pl {data_dir}/text {processor.LEXICON_SOURCES} > {data_dir}/text-phone")
        run(f"utils/split_data.sh {data_dir} {nj}")

        for i in range(1, nj + 1):
            run(
                f"utils/sym2int.pl --map-oov '#0' -f 2- {processor.LM_DIR}/words.txt "
                f"{data_dir}/split{nj}/{i}/text > {data_dir}/split{nj}/{i}/text.int"
            )

        run(
            f"utils/sym2int.pl -f 2- {processor.LM_DIR}/phones.txt "
            f"{data_dir}/text-phone > {data_dir}/text-phone.int"
        )

        # -------- Graph compilation --------
        for i in range(1, nj + 1):
            run(
                f"compile-train-graphs-without-lexicon "
                f"--read-disambig-syms={processor.LM_DIR}/phones/disambig.int "
                f"{processor.MODEL_DIR}/am/tree {processor.MODEL_DIR}/am/final.mdl "
                f"ark,t:{data_dir}/split{nj}/{i}/text.int "
                f"ark,t:{data_dir}/text-phone.int "
                f"ark:|gzip -c > {ali_dir}/fsts.{i}.gz"
            )

        with open(os.path.join(ali_dir, "num_jobs"), "w") as f:
            f.write(str(nj))

        # -------- Alignments --------
        run(
            f"steps/align_mapped.sh --nj {nj} --graphs {ali_dir} "
            f"{data_dir} {probs_dir} {processor.LANG_DIR} {processor.MODEL_DIR}/am {ali_dir}"
        )

        # -------- Pure phone map --------
        run(
            f"local/remove_phone_markers.pl {processor.LANG_DIR}/phones.txt "
            f"{processor.LM_DIR}/phones-pure.txt "
            f"{processor.LM_DIR}/phone-to-pure-phone.int"
        )

        # -------- Phone alignments --------
        for i in range(1, nj + 1):
            run(
                f"ali-to-phones --per-frame=true {processor.MODEL_DIR}/am/final.mdl "
                f"'ark,t:gunzip -c {ali_dir}/ali.{i}.gz|' "
                f"'ark,t:|gzip -c >{ali_dir}/ali-phone.{i}.gz'"
            )

        # -------- GOP computation --------
        for i in range(1, nj + 1):
            run(
                f"compute-gop --phone-map={processor.LM_DIR}/phone-to-pure-phone.int "
                f"{processor.MODEL_DIR}/am/final.mdl "
                f"'ark,t:gunzip -c {ali_dir}/ali.{i}.gz|' "
                f"'ark,t:gunzip -c {ali_dir}/ali-phone.{i}.gz|' "
                f"'ark:{probs_dir}/output.{i}.ark' "
                f"'ark,t:{gop_dir}/gop.{i}.txt' "
                f"'ark,t:{gop_dir}/feat.{i}.txt'"
            )

        # -------- Parse GOP results --------
        parsed_results = {}
        for i in range(1, nj + 1):
            gop_file = os.path.join(gop_dir, f"gop.{i}.txt")
            if not os.path.exists(gop_file):
                continue

            with open(gop_file) as f:
                current_utt = None
                for line in f:
                    tokens = line.strip().split()
                    if not tokens:
                        continue

                    if tokens[0] in utt_ids:
                        current_utt = tokens[0]
                        parsed_results.setdefault(current_utt, {"scores": [], "perphone": []})
                        start_idx = 1
                    else:
                        if current_utt is None:
                            continue
                        start_idx = 0

                    for j in range(start_idx, len(tokens), 4):
                        if j + 2 >= len(tokens):
                            continue
                        try:
                            phone_id = tokens[j + 1]
                            gop_val = float(tokens[j + 2])
                            parsed_results[current_utt]["scores"].append(gop_val)
                            parsed_results[current_utt]["perphone"].append((phone_id, gop_val))
                        except (ValueError, IndexError):
                            continue

        end_time = int(time.time() * 1000)
        latency_ms = end_time - start_time

        formatted_results = []
        for utt_id in utt_ids:
            utt_data = parsed_results.get(utt_id)
            if utt_data and utt_data["scores"]:
                avg_score = sum(utt_data["scores"]) / len(utt_data["scores"])
                formatted_results.append(
                    {
                        "utt_id": utt_id,
                        "sentence_gop": avg_score,
                        "perphone_gop": utt_data["perphone"],
                        "latency_ms": latency_ms // len(utt_ids),
                    }
                )
            else:
                formatted_results.append(
                    {
                        "utt_id": utt_id,
                        "sentence_gop": float("nan"),
                        "perphone_gop": [],
                        "latency_ms": latency_ms // len(utt_ids),
                        "error": "No GOP scores found",
                    }
                )

        return formatted_results

    except Exception as exc:
        end_time = int(time.time() * 1000)
        latency_ms = end_time - start_time
        return [
            {
                "utt_id": os.path.splitext(os.path.basename(pair[0]))[0],
                "sentence_gop": float("nan"),
                "perphone_gop": [],
                "latency_ms": latency_ms // len(audio_transcript_pairs),
                "error": str(exc),
            }
            for pair in audio_transcript_pairs
        ]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def compute_gop(wav_path, transcript):
    """Backward-compatible single-utterance GOP helper."""

    results = compute_gop_batch([(wav_path, transcript)])
    return results[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", help="Path to wav file")
    parser.add_argument("text", help="Transcript")
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    result = compute_gop(args.wav, args.text)

    if args.detailed:
        print(f"Sentence GOP: {result['utt_id']} {result['sentence_gop']}")
        print("Per-phone GOPs:")
        for phone_id, gop_val in result['perphone_gop']:
            print(f"{phone_id} {gop_val}")
        print(f"Latency: {result['latency_ms']} ms")
    else:
        print(f"{result['utt_id']} {result['sentence_gop']}")
        print(f"Latency: {result['latency_ms']} ms")

if __name__ == "__main__":
    main()
