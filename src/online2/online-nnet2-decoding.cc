// online2/online-nnet2-decoding.cc

// Copyright    2013-2014  Johns Hopkins University (author: Daniel Povey)

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

#include "online2/online-nnet2-decoding.h"
#include "lat/lattice-functions.h"
#include "lat/determinize-lattice-pruned.h"

namespace kaldi {

SingleUtteranceNnet2Decoder::SingleUtteranceNnet2Decoder(
    const OnlineNnet2DecodingConfig &config,
    const TransitionModel &tmodel,
    const nnet2::AmNnet &model,
    const fst::Fst<fst::StdArc> &fst,
    OnlineNnet2FeaturePipeline *feature_pipeline,
    const OnlineGmmAdaptationState &gmm_adaptation_state,
    const OnlineGmmDecodingModels &models):
    config_(config),
    feature_pipeline_(feature_pipeline),
    tmodel_(tmodel),
    decodable_(model, tmodel, config.decodable_opts, feature_pipeline),
    decoder_(fst, config.decoder_opts),
    models_(models),
    gmm_orig_adaptation_state_(gmm_adaptation_state),
    gmm_adaptation_state_(gmm_adaptation_state) {
  if (!SplitStringToIntegers(config_.silence_phones, ":", false,
                           &silence_phones_))
    KALDI_ERR << "Bad --silence-phones option '"
              << config_.silence_phones << "'";
  SortAndUniq(&silence_phones_);
  decoder_.InitDecoding();
}

void SingleUtteranceNnet2Decoder::AdvanceDecoding() {
  int32 old_frames = decoder_.NumFramesDecoded();
  decoder_.AdvanceDecoding(&decodable_);
  {  // possibly estimate fMLLR.
    int32 new_frames = decoder_.NumFramesDecoded();
    BaseFloat frame_shift = feature_pipeline_->FrameShiftInSeconds();
    // if the original adaptation state (at utterance-start) had no transform,
    // then this means it's the first utt of the speaker... even if not, if we
    // don't have a transform it probably makes sense to treat it as the 1st utt
    // of the speaker, i.e. to do fMLLR adaptation sooner.
    bool is_first_utterance_of_speaker =
            (gmm_orig_adaptation_state_.transform.NumRows() == 0);
    bool end_of_utterance = false;
    if (config_.adaptation_policy_opts.DoAdapt(old_frames * frame_shift,
                                               new_frames * frame_shift,
                                               is_first_utterance_of_speaker))
      this->EstimateFmllr(end_of_utterance);
  }
}

void SingleUtteranceNnet2Decoder::FinalizeDecoding() {
  decoder_.FinalizeDecoding();
}

int32 SingleUtteranceNnet2Decoder::NumFramesDecoded() const {
  return decoder_.NumFramesDecoded();
}

void SingleUtteranceNnet2Decoder::GetLattice(bool end_of_utterance,
                                             CompactLattice *clat) const {
  if (NumFramesDecoded() == 0)
    KALDI_ERR << "You cannot get a lattice if you decoded no frames.";
  Lattice raw_lat;
  decoder_.GetRawLattice(&raw_lat, end_of_utterance);

  if (!config_.decoder_opts.determinize_lattice)
    KALDI_ERR << "--determinize-lattice=false option is not supported at the moment";

  BaseFloat lat_beam = config_.decoder_opts.lattice_beam;
  DeterminizeLatticePhonePrunedWrapper(
      tmodel_, &raw_lat, lat_beam, clat, config_.decoder_opts.det_opts);
}

void SingleUtteranceNnet2Decoder::GetBestPath(bool end_of_utterance,
                                              Lattice *best_path) const {
  decoder_.GetBestPath(best_path, end_of_utterance);
}

bool SingleUtteranceNnet2Decoder::EndpointDetected(
    const OnlineEndpointConfig &config) {
  return kaldi::EndpointDetected(config, tmodel_,
                                 feature_pipeline_->FrameShiftInSeconds(),
                                 decoder_);  
}

// HERE STARTS GMM CODE
// FMLR ESTIMATION WITH GMM MODEL
void SingleUtteranceNnet2Decoder::EstimateFmllr(bool end_of_utterance) {
  if (decoder_.NumFramesDecoded() == 0) {
    KALDI_WARN << "You have decoded no data so cannot estimate fMLLR.";
  }

  if (GetVerboseLevel() >= 2) {
    Matrix<BaseFloat> feats;
    feature_pipeline_->GetAsMatrix(&feats);
    KALDI_VLOG(2) << "Features are " << feats;
  }


  GaussPost gpost;
  GetGaussianPosteriors(end_of_utterance, &gpost);

  OnlineGmmAdaptationState adaptation_state_ = gmm_adaptation_state_;
  FmllrDiagGmmAccs &spk_stats = adaptation_state_.spk_stats;

  if (spk_stats.beta_ !=
      gmm_orig_adaptation_state_.spk_stats.beta_) {
    // This could happen if the user called EstimateFmllr() twice on the
    // same utterance... we don't want to count any stats twice so we
    // have to reset the stats to what they were before this utterance
    // (possibly empty).
    spk_stats = gmm_orig_adaptation_state_.spk_stats;
  }

  int32 dim = feature_pipeline_->Dim();
  if (spk_stats.Dim() == 0)
    spk_stats.Init(dim);

  Matrix<BaseFloat> empty_transform;
  feature_pipeline_->SetTransform(empty_transform);
  Vector<BaseFloat> feat(dim);

  if (adaptation_state_.transform.NumRows() == 0) {
    // If this is the first time we're estimating fMLLR, freeze the CMVN to its
    // current value.  It doesn't matter too much what value this is, since we
    // have already computed the Gaussian-level alignments (it may have a small
    // effect if the basis is very small and doesn't include an offset as part
    // of the transform).
    feature_pipeline_->FreezeCmvn();
  }

  // GetModel() returns the model to be used for estimating
  // transforms.
  const AmDiagGmm &am_gmm = models_.GetModel();

  for (size_t i = 0; i < gpost.size(); i++) {
    feature_pipeline_->GetFrame(i, &feat);
    for (size_t j = 0; j < gpost[i].size(); j++) {
      int32 pdf_id = gpost[i][j].first; // caution: this gpost has pdf-id
      // instead of transition-id, which is
      // unusual.
      const Vector<BaseFloat> &posterior(gpost[i][j].second);
      spk_stats.AccumulateFromPosteriors(am_gmm.GetPdf(pdf_id),
                                         feat, posterior);
    }
  }

  const BasisFmllrEstimate &basis = models_.GetFmllrBasis();
  if (basis.Dim() == 0)
    KALDI_ERR << "In order to estimate fMLLR, you need to supply the "
              << "--fmllr-basis option.";
  Vector<BaseFloat> basis_coeffs;
  BaseFloat impr = basis.ComputeTransform(spk_stats,
                                          &adaptation_state_.transform,
                                          &basis_coeffs, config_.basis_opts);
  KALDI_VLOG(3) << "Objective function improvement from basis-fMLLR is "
                << (impr / spk_stats.beta_) << " per frame, over "
                << spk_stats.beta_ << " frames, #params estimated is "
                << basis_coeffs.Dim();
  feature_pipeline_->SetTransform(adaptation_state_.transform);
}

// gets Gaussian posteriors for purposes of fMLLR estimation.
// We exclude the silence phones from the Gaussian posteriors.
bool SingleUtteranceNnet2Decoder::GetGaussianPosteriors(bool end_of_utterance,
                                                                GaussPost *gpost) {
  // Gets the Gaussian-level posteriors for this utterance, using whatever
  // features and model we are currently decoding with.  We'll use these
  // to estimate basis-fMLLR with.
  if (decoder_.NumFramesDecoded() == 0) {
    KALDI_WARN << "You have decoded no data so cannot estimate fMLLR.";
    return false;
  }

  KALDI_ASSERT(config_.fmllr_lattice_beam > 0.0);

  // Note: we'll just use whatever acoustic scaling factor we were decoding
  // with.  This is in the lattice that we get from decoder_.GetRawLattice().
  Lattice raw_lat;
  decoder_.GetRawLatticePruned(&raw_lat, end_of_utterance,
                               config_.fmllr_lattice_beam);

  // At this point we could rescore the lattice if we wanted, and
  // this might improve the accuracy on long utterances that were
  // the first utterance of that speaker, if we had already
  // estimated the fMLLR by the time we reach this code (e.g. this
  // was the second call).  We don't do this right now.

  PruneLattice(config_.fmllr_lattice_beam, &raw_lat);

#if 1 // Do determinization.
  Lattice det_lat; // lattice-determinized lattice-- represent this as Lattice
  // not CompactLattice, as LatticeForwardBackward() does not
  // accept CompactLattice.


  fst::Invert(&raw_lat); // want to determinize on words.
  fst::ILabelCompare<kaldi::LatticeArc> ilabel_comp;
  fst::ArcSort(&raw_lat, ilabel_comp); // improves efficiency of determinization

  fst::DeterminizeLatticePruned(raw_lat,
                                double(config_.fmllr_lattice_beam),
                                &det_lat);

  fst::Invert(&det_lat); // invert back.

  if (det_lat.NumStates() == 0) {
    // Do nothing if the lattice is empty.  This should not happen.
    KALDI_WARN << "Got empty lattice.  Not estimating fMLLR.";
    return false;
  }
#else
  Lattice &det_lat = raw_lat; // Don't determinize.
#endif
  TopSortLatticeIfNeeded(&det_lat);

  // Note: the acoustic scale we use here is whatever we decoded with.
  Posterior post;
  BaseFloat tot_fb_like = LatticeForwardBackward(det_lat, &post);

  KALDI_VLOG(3) << "Lattice forward-backward likelihood was "
                << (tot_fb_like / post.size()) << " per frame over " << post.size()
                << " frames.";

  ConstIntegerSet<int32> silence_set(silence_phones_);  // faster lookup
  const TransitionModel &trans_model = models_.GetTransitionModel();
  WeightSilencePost(trans_model, silence_set,
                    config_.silence_weight, &post);

  const AmDiagGmm &am_gmm = (HaveTransform() ? models_.GetModel() :
                             models_.GetOnlineAlignmentModel());


  Posterior pdf_post;
  ConvertPosteriorToPdfs(trans_model, post, &pdf_post);

  Vector<BaseFloat> feat(feature_pipeline_->Dim());

  double tot_like = 0.0, tot_weight = 0.0;
  gpost->resize(pdf_post.size());
  for (size_t i = 0; i < pdf_post.size(); i++) {
    feature_pipeline_->GetFrame(i, &feat);
    for (size_t j = 0; j < pdf_post[i].size(); j++) {
      int32 pdf_id = pdf_post[i][j].first;
      BaseFloat weight = pdf_post[i][j].second;
      const DiagGmm &gmm = am_gmm.GetPdf(pdf_id);
      Vector<BaseFloat> this_post_vec;
      BaseFloat like = gmm.ComponentPosteriors(feat, &this_post_vec);
      this_post_vec.Scale(weight);
      tot_like += like * weight;
      tot_weight += weight;
      (*gpost)[i].push_back(std::make_pair(pdf_id, this_post_vec));
    }
  }
  KALDI_VLOG(3) << "Average likelihood weighted by posterior was "
                << (tot_like / tot_weight) << " over " << tot_weight
                << " frames (after downweighting silence).";
  return true;
}

bool SingleUtteranceNnet2Decoder::HaveTransform() {
  return (feature_pipeline_->HaveFmllrTransform());
}

void SingleUtteranceNnet2Decoder::GetGmmAdaptationState(
        OnlineGmmAdaptationState *adaptation_state) {
  *adaptation_state = gmm_adaptation_state_;
  feature_pipeline_->GetCmvnState(&adaptation_state->cmvn_state);
}


}  // namespace kaldi

