# Container-friendly paths for Kaldi setup
export KALDI_ROOT=/opt/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin:$KALDI_ROOT/src/gmmbin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/featbin:$KALDI_ROOT/src/online2bin:$KALDI_ROOT/src/ivectorbin:$PWD:$PATH
export LC_ALL=C

# We use this both in the (optional) LM training and the G2P-related scripts
PYTHON='python3'

### Below are the paths used by the optional parts of the recipe

# We only need the Festival stuff below for the optional text normalization(for LM-training) step
FEST_ROOT=tools/festival
NSW_PATH=${FEST_ROOT}/festival/bin:${FEST_ROOT}/nsw/bin
export PATH=$PATH:$NSW_PATH

# SRILM is needed for LM model building
SRILM_ROOT=$KALDI_ROOT/tools/srilm
SRILM_PATH=$SRILM_ROOT/bin:$SRILM_ROOT/bin/i686-m64
export PATH=$PATH:$SRILM_PATH

# Sequitur G2P executable
sequitur=$KALDI_ROOT/tools/sequitur-g2p/g2p.py
sequitur_path="$(dirname $sequitur)/lib/$PYTHON/site-packages"

# Directory under which the LM training corpus should be extracted
LM_CORPUS_ROOT=./lm-corpus
export PYTHONUNBUFFERED=1

# Ensure we can find our local scripts
export PATH=$PWD/local:$PATH
