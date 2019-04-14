// nnet2bin/nnet-am-compute.cc

// Copyright 2019 Martin Kocour

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
#include "hmm/transition-model.h"
#include "nnet2/train-nnet.h"
#include "nnet2/am-nnet.h"
#include "hmm/hmm-topology.h"

namespace kaldi {
    template<typename Real>
    void matrix_delete(Matrix<Real> & M, std::vector<MatrixIndexT> indices, int32 axis) {
        if (axis == 0) { // delete rows
            M.Transpose();
            matrix_delete(M, indices, 1);
            M.Transpose();
        } else { // delete cols
            std::sort(indices.begin(), indices.end());
            std::vector<MatrixIndexT> copy_indices(M.NumCols());
            MatrixIndexT last = -1;
            for (int i = 0; i < M.NumCols(); i++) {
                bool find = false;
                for (int j = i; j < M.NumCols(); j++) {
                    if(j <= last) continue; // j is already in the list
                    if(std::find(indices.begin(), indices.end(), j) == indices.end()) {
                        // indices does not contain j, move j-th column to i-th column
                        copy_indices[i] = j;
                        last = j;
                        find = true;
                        break;
                    }
                }
                if (!find) {
                    copy_indices[i] = -1; // Zero the i-th column
                }
            }
            Matrix<Real> tmp(M);
            M.CopyCols(tmp, copy_indices.data());
            M.Resize(M.NumRows(), M.NumCols() - indices.size(), kCopyData);
        }
    }
}

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        using namespace kaldi::nnet2;
        typedef kaldi::int32 int32;
        typedef kaldi::int64 int64;

        const char *usage =
                "Does the neural net computation for each file of input features, and\n"
                        "outputs as a matrix the result toghether with \"Covariance matrix\".\n"
                        "Note: if you want it to apply a log (e.g. for log-likelihoods), use\n"
                        "--apply-log=true\n"
                        "\n"
                        "Usage:  nnet-am-compute [options] <model-in> <feature-rspecifier> "
                        "<feature-or-loglikes-wspecifier> <fisher-wspecifier>\n"
                        "See also: nnet-compute, nnet-logprob, nnet-am-compute\n";

        bool divide_by_priors = false;
        bool apply_log = false;
        bool pad_input = true;
        std::string filter_phones_str = "";
        std::string use_gpu = "no";
        int32 chunk_size = 0;
        ParseOptions po(usage);
        po.Register("divide-by-priors", &divide_by_priors, "If true, "
                "divide by the priors stored in the model and re-normalize, apply-log may follow");
        po.Register("apply-log", &apply_log, "Apply a log to the result of the computation "
                "before outputting.");
        po.Register("pad-input", &pad_input, "If true, duplicate the first and last frames "
                "of input features as required for temporal context, to prevent #frames "
                "of output being less than those of input.");
        po.Register("use-gpu", &use_gpu,
                    "yes|no|optional|wait, only has effect if compiled with CUDA");
        po.Register("chunk-size", &chunk_size, "Process the feature matrix in chunks.  "
                "This is useful when processing large feature files in the GPU.  "
                "If chunk-size > 0, pad-input must be true.");
        po.Register("filter-phones", &filter_phones_str, "Colon separated list of phones. "
                "Following phones will be excluded from computation. e.g. 1:2:3:4:5");

        po.Read(argc, argv);

        if (po.NumArgs() != 4) {
            po.PrintUsage();
            exit(1);
        }
        // If chunk_size is greater than 0, pad_input needs to be true.
        KALDI_ASSERT(chunk_size < 0 || pad_input);

#if HAVE_CUDA==1
        CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
        std::vector<int32> filter_phones_;
        if (!SplitStringToIntegers(filter_phones_str, ":", true,
                                   &filter_phones_)) {
            KALDI_ERR << "Bad value for --filter-phones option: "
                      << filter_phones_str;
        }

        std::string nnet_rxfilename = po.GetArg(1),
                features_rspecifier = po.GetArg(2),
                features_or_loglikes_wspecifier = po.GetArg(3),
                fisher_wspecifier = po.GetArg(4);

        TransitionModel trans_model;
        AmNnet am_nnet;
        {
            bool binary_read;
            Input ki(nnet_rxfilename, &binary_read);
            trans_model.Read(ki.Stream(), binary_read);
            am_nnet.Read(ki.Stream(), binary_read);
        }

        Nnet &nnet = am_nnet.GetNnet();

        std::vector<int32> filter_pdfs_;
        for (int trans_id = 1; trans_id <= trans_model.NumTransitionIds(); trans_id++) {
            int32 pdf = trans_model.TransitionIdToPdf(trans_id);
            int32 phone = trans_model.TransitionIdToPhone(trans_id);
            if (std::find(filter_phones_.begin(), filter_phones_.end(), phone) != filter_phones_.end()
                && std::find(filter_pdfs_.begin(), filter_pdfs_.end(), pdf) == filter_pdfs_.end()) {
                filter_pdfs_.push_back(pdf);
            }
        }

        int64 num_done = 0, num_frames = 0;

        CuVector<BaseFloat> inv_priors(am_nnet.Priors());
        KALDI_ASSERT(!divide_by_priors || inv_priors.Dim() == am_nnet.NumPdfs() &&
                                          "Priors in neural network not set up.");
        inv_priors.ApplyPow(-1.0);

        SequentialBaseFloatMatrixReader feature_reader(features_rspecifier);
        BaseFloatCuMatrixWriter writer(features_or_loglikes_wspecifier);
        BaseFloatCuMatrixWriter fisher_writer(fisher_wspecifier);

        MatrixIndexT filtered_dim = nnet.OutputDim() - filter_pdfs_.size();
        CuMatrix<BaseFloat> acc(filtered_dim, filtered_dim);
        CuVector<BaseFloat> mean(filtered_dim);

        for (; !feature_reader.Done();  feature_reader.Next()) {
            std::string utt = feature_reader.Key();
            const Matrix<BaseFloat> &feats  = feature_reader.Value();

            int32 output_frames = feats.NumRows(), output_dim = nnet.OutputDim();
            if (!pad_input)
                output_frames -= nnet.LeftContext() + nnet.RightContext();
            if (output_frames <= 0) {
                KALDI_WARN << "Skipping utterance " << utt << " because output "
                           << "would be empty.";
                continue;
            }

            Matrix<BaseFloat> output(output_frames, output_dim);
            CuMatrix<BaseFloat> cu_output(output);
            if (chunk_size > 0 && chunk_size < feats.NumRows()) {
                NnetComputationChunked(nnet, feats, chunk_size, &output);
                cu_output.CopyFromMat(output);
            } else {
                CuMatrix<BaseFloat> cu_feats(feats);
                NnetComputation(nnet, cu_feats, pad_input, &cu_output);
                output.CopyFromMat(cu_output);
            }

            if (divide_by_priors) {
                cu_output.MulColsVec(inv_priors); // scales each column by the corresponding element
                // of inv_priors.
                for (int32 i = 0; i < cu_output.NumRows(); i++) {
                    CuSubVector<BaseFloat> frame(cu_output, i);
                    BaseFloat p = frame.Sum();
                    if (!(p > 0.0)) {
                        KALDI_WARN << "Bad sum of probabilities " << p;
                    } else {
                        frame.Scale(1.0 / p); // re-normalize to sum to one.
                    }
                }
            }

            if (apply_log) {
                cu_output.ApplyFloor(1.0e-20);
                cu_output.ApplyLog();
            }

            if (filter_pdfs_.size() > 0) {
                Matrix<BaseFloat> filtered_output(cu_output);
                matrix_delete(filtered_output, filter_pdfs_, 1);
                cu_output.Resize(output_frames, filtered_output.NumCols());
                cu_output.CopyFromMat(filtered_output);
            }

            for (int32 i = 0; i < cu_output.NumRows(); i++) {
                CuSubVector<BaseFloat> frame(cu_output, i); // output of 1 frame
                acc.AddVecVec(1.0, frame, frame);
                mean.AddVec(1.0, frame);
            }

            writer.Write(utt, cu_output);
            num_frames += feats.NumRows();
            num_done++;

            KALDI_VLOG(3) << "Done: " << num_done;
        }

        mean.Scale(1 / (float) num_frames);
        acc.Scale(1 / (float) num_frames);
        acc.AddVecVec(-1.0, mean, mean);

        fisher_writer.Write("ACC", acc);

#if HAVE_CUDA==1
        CuDevice::Instantiate().PrintProfile();
#endif

        KALDI_LOG << "Processed " << num_done << " feature files, "
                  << num_frames << " frames of input were processed.";

        return (num_done == 0 ? 1 : 0);
    } catch(const std::exception &e) {
        std::cerr << e.what() << '\n';
        return -1;
    }
}
