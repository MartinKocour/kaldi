// bin/compile-reference-graph.cc

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

template<class Arc>
void MakeEditTransducer(const std::vector<int32> &trans, const std::vector<int32> &words, fst::MutableFst<Arc> *ofst) {
    typedef typename Arc::StateId StateId;
    typedef typename Arc::Weight Weight;

    std::vector<int32> uniq_labels(trans);
    kaldi::SortAndUniq(&uniq_labels);

    ofst->DeleteStates();
    StateId cur_state = ofst->AddState();
    ofst->SetStart(cur_state);
    for (size_t i = 0; i < uniq_labels.size(); i++) {
        for (size_t j = 0; j < words.size(); j++) {
            Arc arc;
            if (words[j] == uniq_labels[i]) {
                arc = Arc(uniq_labels[i], words[j], Weight::One(), cur_state);
            } else {
                arc = Arc(uniq_labels[i], words[j], Weight::Zero(), cur_state);
            }
            ofst->AddArc(cur_state, arc);
        }
    }
    int32 eps = 0;
    for (size_t i = 0; i < words.size(); i++) {
        if(words[i] == eps) {
            continue;
        }
        Arc arc1(eps, words[j], Weight::Zero(), cur_state);
        Arc arc2(words[j], eps, Weight::Zero(), cur_state);
        ofst->AddArc(cur_state, arc);
    }
    ofst->SetFinal(cur_state, Weight::One());
}

template<class Arc, class I>
void MakeReferenceTransducer(const std::vector<I> &labels, fst::MutableFst<Arc> *ofst) {
    fst::MakeLinearAcceptor(labels, ofst);
}

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        typedef kaldi::int32 int32;
        using fst::SymbolTable;
        using fst::VectorFst;
        using fst::StdArc;

        const char *usage =
                "Creates FST graphs from transcripts. (Graphs are used in the lattice based SST training)\n"
                "\n"
                "Usage: compile-reference-graph [options] <word-rxfilename> <transcripts-rspecifier> <fst-graphs-wspecifier>\n"
                "e.g. compile-reference-graph 'words.int' 'ark:sym2int.pl --map-oov 1 -f 2- words.txt text|' ark:reference.fsts\n"
                "\n";
        ParseOptions po(usage);
        po.Read(argc, argv);

        if (po.NumArgs() != 3) {
            po.PrintUsage();
            exit(1);
        }

        std::string word_rspecifier = po.GetArg(1);
        std::string transcript_rspecifier = po.GetArg(2);
        std::string fsts_wspecifier = po.GetArg(3);

        std::vector<int32> word_syms;
        if (!ReadIntegerVectorSimple(word_rspecifier, &word_syms))
            KALDI_ERR << "Could not read word symbols from "
                      << word_rspecifier;

        SequentialInt32VectorReader transcript_reader(transcript_rspecifier);
        TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);

        int num_succeed = 0, num_fail = 0;

        for (; !transcript_reader.Done(); transcript_reader.Next()) {
            std::string key = transcript_reader.Key();
            const std::vector<int32> &transcript = transcript_reader.Value();

            VectorFst<StdArc> edit_fst;
            MakeEditTransducer(transcript, word_syms, &edit_fst);

            VectorFst<StdArc> reference_fst;
            MakeReferenceTransducer(transcript, &reference_fst);

            //TODO Add checks

            VectorFst<StdArc> ref_edit_fst;
            fst::TableCompose(reference_fst, edit_fst, &ref_edit_fst); // TODO add cache

            if (reference_fst.Start() != fst::kNoStateId) {
                num_succeed++;
                fst_writer.Write(key, reference_fst);
            } else {
                KALDI_WARN << "Empty decoding graph for utterance "
                           << key;
                num_fail++;
            }

            if (edit_fst.Start() != fst::kNoStateId) {
                num_succeed++;
                fst_writer.Write(key, edit_fst);
            } else {
                KALDI_WARN << "Empty decoding graph for utterance "
                           << key;
                num_fail++;
            }

            if (ref_edit_fst.Start() != fst::kNoStateId) {
                num_succeed++;
                fst_writer.Write(key, ref_edit_fst);
            } else {
                KALDI_WARN << "Empty decoding graph for utterance "
                           << key;
                num_fail++;
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
