# Checkpoints

## Download

All checkpoint weights are hosted on HuggingFace, not in git (GitHub 100MB file limit):

**https://huggingface.co/IVPL/CaReDiff/tree/main/personalised**

```
personalised/
├── offline/
│   ├── backbone/                 frozen generic offline backbone (shared by all conditions)
│   │   ├── CausalTransformerDenoiser/checkpoint_120.pth
│   │   ├── DiffusionPriorNetwork/checkpoint_120.pth
│   │   └── EEGPredictionHead/checkpoint_120.pth
│   └── adapters/
│       ├── personality/ModifierNetwork/checkpoint_best.pth
│       ├── lhfb/ModifierNetwork/checkpoint_best.pth
│       └── both/ModifierNetwork/checkpoint_best.pth
└── online/
    ├── backbone/                 frozen generic online backbone (shared by all conditions)
    │   ├── CausalTransformerDenoiser/checkpoint_120.pth
    │   ├── DiffusionPriorNetwork/checkpoint_120.pth
    │   └── EEGPredictionHead/checkpoint_120.pth
    └── adapters/
        ├── personality/ModifierNetwork/checkpoint_best.pth
        ├── lhfb/ModifierNetwork/checkpoint_best.pth
        └── both/ModifierNetwork/checkpoint_best.pth
```

The backbone weights are identical to the generic submission of the same track.
Each adapter file also contains the fine-tuned EEG head, which overwrites the
backbone EEG head at load time.

## Placement

Download the `personalised/` folder to any location, keeping the directory
structure above, and set `PKG` to its absolute path. The evaluation commands
below reference all weights explicitly through `PKG`, so no further placement
is required.

## Evaluation

Run from `personalised/code/` in the CaReDiff repository
(see `offline/README.md` and `online/README.md` for full details).

```bash
PKG=<absolute path of the downloaded personalised folder>

# Personalised Offline (personality condition)
python main.py --config-name g2p_delta stage=test task=offline \
  data_dir=<MARS_ROOT> run_id=eval_offline_personality \
  trainer.batch_size=4 num_gts=10 \
  trainer.generic.eval_condition_mode=matched \
  trainer.generic.eval_eeg=false \
  trainer.main_model.args.personal_condition_mode=personality_only \
  resume_id=personality \
  trainer.ckpt_dir=$PKG/offline/adapters \
  trainer.pretrained.diffusion_decoder=$PKG/offline/backbone/CausalTransformerDenoiser/checkpoint_120.pth \
  trainer.pretrained.diffusion_prior=$PKG/offline/backbone/DiffusionPriorNetwork/checkpoint_120.pth \
  trainer.pretrained.eeg_head_checkpoint=$PKG/offline/backbone/EEGPredictionHead/checkpoint_120.pth

# Personalised Online (personality condition)
python main.py --config-name g2p_delta_online stage=test task=online \
  data_dir=<MARS_ROOT> run_id=eval_online_personality \
  trainer.batch_size=4 num_gts=10 \
  trainer.generic.eval_condition_mode=matched \
  trainer.generic.eval_eeg=false \
  trainer.main_model.args.personal_condition_mode=personality_only \
  trainer.model.diff_model.diffusion_decoder.args.past_l_emotion_drop_prob=0.2 \
  resume_id=personality \
  trainer.ckpt_dir=$PKG/online/adapters \
  trainer.pretrained.diffusion_decoder=$PKG/online/backbone/CausalTransformerDenoiser/checkpoint_120.pth \
  trainer.pretrained.diffusion_prior=$PKG/online/backbone/DiffusionPriorNetwork/checkpoint_120.pth \
  trainer.pretrained.eeg_head_checkpoint=$PKG/online/backbone/EEGPredictionHead/checkpoint_120.pth
```

For the `lhfb` / `both` conditions, change `resume_id` (`lhfb` or `both`) and
`personal_condition_mode` (`3dmm_only` or `3dmm_personality`) accordingly.
For the online track, keep `past_l_emotion_drop_prob=0.2`; this flag enables
the past-listener conditioning pathway and is required to reproduce the
reported numbers.

## SHA-256

### Offline

| File | SHA-256 |
|------|---------|
| `backbone/CausalTransformerDenoiser/checkpoint_120.pth` | `68faca9700415c949eecbe7bd3e381877a76b5e1b24bdab9c30e6fd5b628faa2` |
| `backbone/DiffusionPriorNetwork/checkpoint_120.pth` | `d1b66e87f51afd9bb93bdcef1b9e350e6366aa8f995920e400d7e7dd4e299357` |
| `backbone/EEGPredictionHead/checkpoint_120.pth` | `750c49999a180cda330b88d771f99d1dca0fd94a810470ea77a45561cfd58780` |
| `adapters/personality/ModifierNetwork/checkpoint_best.pth` | `8e0a501237c9b80b8c9e9524bd089fa5ca54ad747bdf9ed65dd97b8d883bf928` |
| `adapters/lhfb/ModifierNetwork/checkpoint_best.pth` | `0ddfde5284c580c3cc461006b2b7cd4df73d2715a8700d3838d8d5e5db8eb7f4` |
| `adapters/both/ModifierNetwork/checkpoint_best.pth` | `73434669c633bc6384acc9845e62c0c4302c9322be27f04e30005e55dda3ab92` |

### Online

| File | SHA-256 |
|------|---------|
| `backbone/CausalTransformerDenoiser/checkpoint_120.pth` | `f4fc53506fc94a65e86b52bfe1491669a73ca3429ad0b1ab51c62488854242f0` |
| `backbone/DiffusionPriorNetwork/checkpoint_120.pth` | `8b717d619cd37fc793f80f37a4af607bda5e9709c83b82f17916b2467a4380a6` |
| `backbone/EEGPredictionHead/checkpoint_120.pth` | `60c7a7ae4e6a233fdb59c0ee1e099daf1158931d876a5f46386a781fa2a52a52` |
| `adapters/personality/ModifierNetwork/checkpoint_best.pth` | `fad7691aea8c1895a11fcc7d40873d83ea44e0692e6dfb75f736d89cdd9e62d0` |
| `adapters/lhfb/ModifierNetwork/checkpoint_best.pth` | `c5d8c774774b8a1994be3149c3bb475384326e24a9af64edbf15cda0292da868` |
| `adapters/both/ModifierNetwork/checkpoint_best.pth` | `77a744f486c46484dc7a357c484bd5c3582345ea5f6947aea1aa625ba00e660a` |
