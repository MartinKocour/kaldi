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

    void SetValue(Matrix<BaseFloat> *mat, std::vector<int32> cols_rows, BaseFloat n) {
        MatrixIndexT rows = mat->NumRows();
        MatrixIndexT cols = mat->NumCols();
        for (int i = 0; i < cols_rows.size(); i++) {
            MatrixIndexT pdf = cols_rows[i];
            mat->Range(pdf, 1, 0, cols).Set(1.0);
            mat->Range(0, rows, pdf, 1).Set(1.0);
        }
    }

}  // namespace kaldi

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;

        const char *usage =
                "Kocour-Luque-Cernocky metric for data selection.\n"
                "\n"
                "Usage: klc-metric [options] <nnet_rxfilename> <base-matrix-rspecifier> <matrix-rspecifier> <metric-wxfilename>\n"
                "       klc-metric final.mdl ark:base_mat.ark ark,t:- \n";

        std::string filter_phones_str = "";

        ParseOptions po(usage);
        po.Register("filter-phones", &filter_phones_str, "Colon separated list of phones. "
                "Following phones will be excluded from computation. e.g. 1:2:3:4:5");

        po.Read(argc, argv);

        if (po.NumArgs() != 4) {
            po.PrintUsage();
            exit(1);
        }

        std::vector<int32> filter_phones_;
        if (!SplitStringToIntegers(filter_phones_str, ":", true,
                                   &filter_phones_)) {
            KALDI_ERR << "Bad value for --filter-phones option: "
                      << filter_phones_str;
        }

        std::string nnet_rxfilename = po.GetArg(1),
                matrix_base_fn = po.GetArg(2),
                matrix_fn = po.GetArg(3),
                metric_fn = po.GetArg(4);

        TransitionModel trans_model;
        {
            bool binary_read;
            Input ki(nnet_rxfilename, &binary_read);
            trans_model.Read(ki.Stream(), binary_read);
        }

        std::vector<int32> filter_pdfs_;
        for (int trans_id = 1; trans_id <= trans_model.NumTransitionIds(); trans_id++) {
            int32 pdf = trans_model.TransitionIdToPdf(trans_id);
            int32 phone = trans_model.TransitionIdToPhone(trans_id);
            if (std::find(filter_phones_.begin(), filter_phones_.end(), phone) != filter_phones_.end()
                && std::find(filter_pdfs_.begin(), filter_pdfs_.end(), pdf) == filter_pdfs_.end()) {
                filter_pdfs_.push_back(pdf);
            }
        }

        RandomAccessBaseFloatMatrixReader base_mat_reader(matrix_base_fn);
        RandomAccessBaseFloatMatrixReader mat_reader(matrix_fn);

        Matrix<BaseFloat> base_mat(base_mat_reader.Value("ACC"));
        Matrix<BaseFloat> mat(mat_reader.Value("ACC"));

        SetValue(&mat, filter_pdfs_, 1.0); // Set 1.0 instead of silence activation
        SetValue(&base_mat, filter_pdfs_, 1.0); // Set 1.0 instead of silence activation

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
