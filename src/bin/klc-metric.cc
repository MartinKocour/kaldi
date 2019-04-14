// bin/copy-matrix.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "matrix/kaldi-matrix.h"
#include "transform/transform-common.h"
#include "hmm/transition-model.h"

namespace kaldi {

    void ApplySoftMaxPerRow(MatrixBase<BaseFloat> *mat) {
        for (int32 i = 0; i < mat->NumRows(); i++) {
            mat->Row(i).ApplySoftMax();
        }
    }

}  // namespace kaldi

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;

        const char *usage =
                "Kocour-Luque-Cernocky metric for data selection.\n"
                        "\n"
                        "Usage: klc-metric [options] <base-matrix-rspecifier> <matrix-rspecifier> <metric-wxfilename>\n"
                        "       klc-metric ark:base_mat.ark ark,t:- \n";

        ParseOptions po(usage);

        po.Read(argc, argv);

        if (po.NumArgs() != 3) {
            po.PrintUsage();
            exit(1);
        }

        std::string matrix_base_fn = po.GetArg(1),
                matrix_fn = po.GetArg(2),
                metric_fn = po.GetArg(3);

        RandomAccessBaseFloatMatrixReader base_mat_reader(matrix_base_fn);
        RandomAccessBaseFloatMatrixReader mat_reader(matrix_fn);

        Matrix<BaseFloat> base_mat(base_mat_reader.Value("ACC"));
        Matrix<BaseFloat> mat(mat_reader.Value("ACC"));

        /*
         * value = sqrt(sum_over_elmnts(square_over_elmnts(mat-base_mat)))
         */
        Matrix<BaseFloat> result(mat);
        mat.AddMat(-1.0, base_mat, kNoTrans);
        mat.ApplyPow(2);
        float value = sqrt(mat.Sum());

        Output ko(metric_fn, false);
        ko.Stream() << value << "\n";
        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}