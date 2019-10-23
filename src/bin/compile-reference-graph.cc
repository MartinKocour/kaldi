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
                "Usage: compile-reference-graphs-fst [options] <reference-transcriptions-rspecifier> <fst-graphs-wspecifier>\n"
                "e.g. compile-reference-graphs-fst 'ark:sym2int.pl -f 2- words.txt text|' ark:reference.fsts"
                "see also "; //TODO add other programs
        ParseOptions po(usage);
        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
            po.PrintUsage();
            exit(1);
        }

        std::string transcript_rspecifier = po.GetArg(1);
        std::string fsts_wspecifier = po.GetArg(2);

        SequentialInt32VectorReader transcript_reader(transcript_rspecifier);
        TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);

        int num_succeed = 0, num_fail = 0;

        for (; !transcript_reader.Done(); transcript_reader.Next()) {
            std::string key = transcript_reader.Key();
            const std::vector<int32> &transcript = transcript_reader.Value();
            VectorFst<StdArc> reference_fst;

            MakeLinearAcceptor(transcript, &reference_fst);
            if (reference_fst.Start() == fst::kNoStateId) {
                reference_fst.DeleteStates();  // Just make it empty.
            }
            if (reference_fst.Start() != fst::kNoStateId) {
                num_succeed++;
                fst_writer.Write(key, reference_fst);
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
