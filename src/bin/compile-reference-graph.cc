// bin/compile-reference-graph.cc

// This binary is a part of the implementation of lattice-based SST training approach
// introduced in ﻿http://arxiv.org/abs/1905.13150 by Joachim Fainberg, et al.

// Copyright 2009-2012  Microsoft Corporation
//           2012-2015  Johns Hopkins University (Author: Daniel Povey)
//           2019       Brno University of Technology (author: Martin Kocour)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/training-graph-compiler.h"
#include "lat/kaldi-lattice.h"

using namespace kaldi;
typedef kaldi::int32 int32;
using fst::SymbolTable;
using fst::VectorFst;
using fst::StdArc;
using fst::MutableFst;
using fst::MakeLinearAcceptor;
using std::vector;
using std::string;

template<class Arc>
void MakeEditTransducer(const vector<int32> &trans, const vector<int32> &words, MutableFst<Arc> *ofst) {
    typedef typename Arc::StateId StateId;
    typedef typename Arc::Weight Weight;

    // We do not create large Edit transducer with N*N arcs, where N=len(words)

    vector<int32> uniq_labels(trans);
    SortAndUniq(&uniq_labels);

    ofst->DeleteStates();
    StateId cur_state = ofst->AddState();
    ofst->SetStart(cur_state);
    for (size_t i = 0; i < uniq_labels.size(); i++) {
        for (size_t j = 0; j < words.size(); j++) {
            Arc arc;
            if (words[j] == uniq_labels[i]) {
                arc = Arc(uniq_labels[i], words[j], Weight(-1.0), cur_state);
            } else {
                arc = Arc(uniq_labels[i], words[j], Weight::One(), cur_state);
            }
            ofst->AddArc(cur_state, arc);
        }
    }
    int32 eps = 0;
    for (size_t i = 0; i < words.size(); i++) {
        if(words[i] == eps) {
            continue;
        }
        Arc arc1(eps, words[i], Weight::One(), cur_state);
        Arc arc2(words[i], eps, Weight::One(), cur_state);
        ofst->AddArc(cur_state, arc1);
        ofst->AddArc(cur_state, arc2);
    }
    ofst->SetFinal(cur_state, Weight::One());
}

template<class Arc, class I>
void MakeReferenceTransducer(const vector<I> &labels, MutableFst<Arc> *ofst) {
    MakeLinearAcceptor(labels, ofst);
}

template<class Arc>
void MakeHypothesisTransducer(CompactLattice &clat, vector<vector<double>> scale, MutableFst<Arc> *ofst) {
    ScaleLattice(scale, &clat); // typically scales to zero.
    fst::RemoveAlignmentsFromCompactLattice(&clat);
    kaldi::Lattice lat;
    ConvertLattice(clat, &lat); // convert to non-compact form.. won't introduce
    // extra states because already removed alignments.
    ConvertLattice(lat, ofst); // this adds up the (lm,acoustic) costs to get
    // the normal (tropical) costs.
    fst::Project(ofst, fst::PROJECT_OUTPUT); // project on words.
}

int main(int argc, char *argv[]) {
    try {

        BaseFloat acoustic_scale = 0.0;
        BaseFloat lm_scale = 0.0;

        const char *usage =
            "Creates FST graphs from transcripts. (Graphs are used in the lattice based SST training)\n"
            "\n"
            "Usage: compile-reference-graph [options] <word-rxfilename> <transcripts-rspecifier> <lattice-rspecifier> <fst-wspecifier>\n"
            "e.g. compile-reference-graph 'words.int' 'ark:sym2int.pl --map-oov 1 -f 2- words.txt text|' ark:1.lats ark:reference.fsts\n"
            "\n";
        ParseOptions po(usage);
        po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
        po.Register("lm-scale", &lm_scale, "Scaling factor for graph/lm costs");

        po.Read(argc, argv);

        if (po.NumArgs() != 4) {
            po.PrintUsage();
            exit(1);
        }

        string word_rspecifier = po.GetArg(1);
        string transcript_rspecifier = po.GetArg(2);
        string lats_rspecifier = po.GetArg(3);
        string fsts_wspecifier = po.GetArg(4);

        vector<int32> word_syms;
        if (!ReadIntegerVectorSimple(word_rspecifier, &word_syms))
            KALDI_ERR << "Could not read word symbols from "
                      << word_rspecifier;

        vector<vector<double> > scale = fst::LatticeScale(lm_scale, acoustic_scale);

        SequentialInt32VectorReader transcript_reader(transcript_rspecifier);
        RandomAccessCompactLatticeReader clat_reader(lats_rspecifier);
        TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);

        int num_succeed = 0, num_fail = 0;

        for (; !transcript_reader.Done(); transcript_reader.Next()) {
            std::string key = transcript_reader.Key();
            const std::vector<int32> &transcript = transcript_reader.Value();

            VectorFst<StdArc> edit_fst;
            MakeEditTransducer(transcript, word_syms, &edit_fst);

            if (edit_fst.Start() == fst::kNoStateId) {
                KALDI_WARN << "Empty edit FST for utterance "
                           << key;
                num_fail++;
                continue;
            }

            VectorFst<StdArc> reference_fst;
            MakeReferenceTransducer(transcript, &reference_fst);

            if (reference_fst.Start() == fst::kNoStateId) {
                KALDI_WARN << "Empty transcript FST for utterance "
                           << key;
                num_fail++;
                continue;
            }

            CompactLattice clat = clat_reader.Value(key);
            fst::VectorFst<StdArc> hypothesis_fst;
            MakeHypothesisTransducer(clat, scale, &hypothesis_fst);

            if (hypothesis_fst.Start() == fst::kNoStateId) {
                KALDI_WARN << "Empty lattice for utterance "
                           << key;
                num_fail++;
                continue;
            }

            VectorFst<StdArc> ref_edit_fst;
            fst::TableCompose(reference_fst, edit_fst, &ref_edit_fst); // TODO add cache
            fst::ArcSort(&ref_edit_fst, fst::OLabelCompare<StdArc>());

            if (ref_edit_fst.Start() == fst::kNoStateId) {
                KALDI_WARN << "Empty composition of transcripts with edit FST for utterance "
                           << key;
                num_fail++;
                continue;
            }

            VectorFst<StdArc> ref_edit_hyp_fst;
            fst::TableCompose(ref_edit_fst, hypothesis_fst, &ref_edit_hyp_fst); // TODO add cache
            fst::ArcSort(&ref_edit_hyp_fst, fst::OLabelCompare<StdArc>());

            if (ref_edit_hyp_fst.Start() == fst::kNoStateId) {
                KALDI_WARN << "Empty composition of transcripts with edit FST and hypothesis for utterance "
                           << key;
                num_fail++;
                continue;
            }

            StdArc::Weight threshold = StdArc::Weight().One();
            fst::Prune(&ref_edit_hyp_fst, threshold);
            fst::Project(&ref_edit_hyp_fst, fst::PROJECT_OUTPUT);
            fst::RmEpsilon(&ref_edit_hyp_fst);
            VectorFst<StdArc> ref_edit_hyp_fst_determinized;
            fst::DeterminizeStar(ref_edit_hyp_fst, &ref_edit_hyp_fst_determinized);
            fst::Minimize(&ref_edit_hyp_fst_determinized);

            if (ref_edit_hyp_fst_determinized.Start() != fst::kNoStateId) {
                num_succeed++;
                fst_writer.Write(key, ref_edit_hyp_fst_determinized);
            } else {
                KALDI_WARN << "Empty final graph for utterance "
                           << key;
                num_fail++;
                continue;
            }
        }

        KALDI_LOG << "compile-train-graphs: succeeded for " << num_succeed
                  << " graphs, failed for " << num_fail;
        return (num_succeed != 0 ? 0 : 1);
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
