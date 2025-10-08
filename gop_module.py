#!/usr/bin/env python3
"""
Portable GOP (Goodness of Pronunciation) module for Kannada speech assessment.
Modified to use relative paths and work in containerized environments.
"""

import os
import sys
import subprocess
import time
import argparse
import tempfile

# -------- Helper to run shell commands --------
def run(cmd, log=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sourced_cmd = f"cd {base_dir} && source {base_dir}/path.sh && {cmd}"
    print(f"[RUN] {sourced_cmd}")
    result = subprocess.run(sourced_cmd, shell=True, executable='/bin/bash')
    if result.returncode != 0:
        print(f"Command failed: {sourced_cmd}")
        raise RuntimeError(f"Command failed: {sourced_cmd}")

def compute_gop(wav_path, transcript):
    """
    Compute Goodness of Pronunciation for a given WAV file and transcript.

    Args:
        wav_path (str): Path to the WAV audio file
        transcript (str): The expected transcript text

    Returns:
        dict: GOP results containing sentence score, per-phone scores, and latency
    """
    # Get the base directory (where this script is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # -------- Config with relative paths --------
    MODEL_DIR = os.path.join(base_dir, "models", "vosk_kannada_model")
    LANG_DIR = os.path.join(MODEL_DIR, "graph")
    IVECTOR_DIR = os.path.join(MODEL_DIR, "ivector")
    LM_DIR = os.path.join(base_dir, "models", "LM_2gram_aiish")

    WORKDIR = os.path.join(base_dir, "gop_work_tmp")
    nj = 1

    os.makedirs(WORKDIR, exist_ok=True)

    # -------- IDs and paths --------
    UTT_ID = os.path.splitext(os.path.basename(wav_path))[0]
    DATA = os.path.join(WORKDIR, f"data_{UTT_ID}")
    ALI = os.path.join(WORKDIR, f"ali_{UTT_ID}")
    PROBS = os.path.join(WORKDIR, f"probs_{UTT_ID}")
    GOP = os.path.join(WORKDIR, f"gop_{UTT_ID}")

    for d in [DATA, ALI, PROBS, GOP]:
        os.makedirs(d, exist_ok=True)

    # -------- Start timer --------
    start_time = int(time.time() * 1000)

    try:
        # -------- Prepare Kaldi data dir --------
        with open(os.path.join(DATA, "wav.scp"), "w") as f:
            f.write(f"{UTT_ID} {wav_path}\n")
        with open(os.path.join(DATA, "text"), "w") as f:
            f.write(f"{UTT_ID} {transcript}\n")
        with open(os.path.join(DATA, "utt2spk"), "w") as f:
            f.write(f"{UTT_ID} {UTT_ID}\n")

        run(f"utils/utt2spk_to_spk2utt.pl < {DATA}/utt2spk > {DATA}/spk2utt")

        # -------- Features --------
        run(f"steps/make_mfcc.sh --nj 1 --mfcc-config conf/mfcc_hires.conf {DATA}")
        run(f"steps/compute_cmvn_stats.sh {DATA}")
        run(f"utils/fix_data_dir.sh {DATA}")

        run(f"steps/online/nnet2/extract_ivectors.sh --nj 1 "
            f"{DATA} {LM_DIR} {IVECTOR_DIR} "
            f"{DATA}/ivectors/ivectors_offline")

        # -------- Log-likelihoods --------
        run(f"steps/nnet3/compute_output.sh --nj {nj} "
            f"--online-ivector-dir {DATA}/ivectors/ivectors_offline "
            f"{DATA} {MODEL_DIR}/am {PROBS}")

        # -------- Prepare phone-level transcript --------
        run(f"make_text_phone.pl {DATA}/text {LM_DIR}/lexicon.txt > {DATA}/text-phone")
        run(f"utils/split_data.sh {DATA} {nj}")
        run(f"utils/sym2int.pl -f 2- {LM_DIR}/words.txt "
            f"{DATA}/split1/1/text > {DATA}/split1/1/text.int")
        run(f"utils/sym2int.pl -f 2- {LM_DIR}/phones.txt "
            f"{DATA}/text-phone > {DATA}/text-phone.int")

        # -------- Graph compilation --------
        run(f"compile-train-graphs-without-lexicon "
            f"--read-disambig-syms={LM_DIR}/phones/disambig.int "
            f"{MODEL_DIR}/am/tree {MODEL_DIR}/am/final.mdl "
            f"ark,t:{DATA}/split{nj}/1/text.int "
            f"ark,t:{DATA}/text-phone.int "
            f"ark:|gzip -c > {ALI}/fsts.1.gz")
        with open(os.path.join(ALI, "num_jobs"), "w") as f:
            f.write(str(nj))

        # -------- Alignment --------
        run(f"steps/align_mapped.sh --nj {nj} --graphs {ALI} "
            f"{DATA} {PROBS} {LANG_DIR} {MODEL_DIR}/am {ALI}")

        # -------- Pure phone map --------
        run(f"local/remove_phone_markers.pl {LANG_DIR}/phones.txt "
            f"{LM_DIR}/phones-pure.txt "
            f"{LM_DIR}/phone-to-pure-phone.int")

        # -------- Convert to phone alignments --------
        run(f"ali-to-phones --per-frame=true {MODEL_DIR}/am/final.mdl "
            f"'ark,t:gunzip -c {ALI}/ali.1.gz|' "
            f"'ark,t:|gzip -c >{ALI}/ali-phone.1.gz'")

        # -------- Compute GOP --------
        # compute-gop expects: <model> <transition-alignments> <phoneme-alignments> <prob-matrix> <gop-wspecifier> <phone-feature-wspecifier>
        run(f"compute-gop --phone-map={LM_DIR}/phone-to-pure-phone.int "
            f"{MODEL_DIR}/am/final.mdl "
            f"'ark,t:gunzip -c {ALI}/ali.1.gz|' "
            f"'ark,t:gunzip -c {ALI}/ali-phone.1.gz|' "
            f"'ark:{PROBS}/output.1.ark' "
            f"'ark,t:{GOP}/gop.1.txt' "
            f"'ark,t:{GOP}/feat.1.txt'")

        # -------- Parse GOP results --------
        avg = None
        perphone = []
        with open(os.path.join(GOP, "gop.1.txt")) as f:
            scores = []
            for line in f:
                tokens = line.strip().split()
                for j in range(1, len(tokens), 4):  # every "[ phone gop ]"
                    if j+2 < len(tokens):
                        try:
                            phone_id = tokens[j+1]
                            gop_val = float(tokens[j+2])
                            scores.append(gop_val)
                            perphone.append((phone_id, gop_val))
                        except ValueError:
                            continue
            if scores:
                avg = sum(scores) / len(scores)
            else:
                avg = float("nan")

        # -------- End timer --------
        end_time = int(time.time() * 1000)
        latency_ms = end_time - start_time

        return {
            "utt_id": UTT_ID,
            "sentence_gop": avg,
            "perphone_gop": perphone,
            "latency_ms": latency_ms
        }

    except Exception as e:
        # -------- End timer --------
        end_time = int(time.time() * 1000)
        latency_ms = end_time - start_time

        return {
            "utt_id": UTT_ID,
            "sentence_gop": float("nan"),
            "perphone_gop": [],
            "latency_ms": latency_ms,
            "error": str(e)
        }
    finally:
        # Clean up temporary directories
        import shutil
        try:
            shutil.rmtree(WORKDIR, ignore_errors=True)
        except:
            pass

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
