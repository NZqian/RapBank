
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from torch.nn import Module
# from torchaudio.pipelines._wav2vec2 import utils
#from . import utils
import utils




@dataclass
class Wav2Vec2Bundle:
    """Data class that bundles associated information to use pretrained :py:class:`~torchaudio.models.Wav2Vec2Model`.

    This class provides interfaces for instantiating the pretrained model along with
    the information necessary to retrieve pretrained weights and additional data
    to be used with the model.

    Torchaudio library instantiates objects of this class, each of which represents
    a different pretrained model. Client code should access pretrained models via these
    instances.

    Please see below for the usage and the available values.

    Example - Feature Extraction
        >>> import torchaudio
        >>>
        >>> bundle = torchaudio.pipelines.HUBERT_BASE
        >>>
        >>> # Build the model and load pretrained weight.
        >>> model = bundle.get_model()
        Downloading:
        100%|███████████████████████████████| 360M/360M [00:06<00:00, 60.6MB/s]
        >>>
        >>> # Resample audio to the expected sampling rate
        >>> waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        >>>
        >>> # Extract acoustic features
        >>> features, _ = model.extract_features(waveform)
    """  # noqa: E501

    _path: str
    _params: Dict[str, Any]
    _sample_rate: float
    _normalize_waveform: bool
    _model_type: str

    @property
    def sample_rate(self) -> float:
        """Sample rate of the audio that the model is trained on.

        :type: float
        """
        return self._sample_rate

    def _get_state_dict(self, dl_kwargs):
        # Note: This method is overridden in ASR bundle
        return utils._get_state_dict(self._path, dl_kwargs)

    def get_model(self, *, dl_kwargs=None) -> Module:
        """Construct the model and load the pretrained weight.

        The weight file is downloaded from the internet and cached with
        :func:`torch.hub.load_state_dict_from_url`

        Args:
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`.

        Returns:
            Variation of :py:class:`~torchaudio.models.Wav2Vec2Model`.

            For the models listed below, an additional layer normalization is performed on the input.

            For all other models, a :py:class:`~torchaudio.models.Wav2Vec2Model` instance is returned.

            - WAV2VEC2_LARGE_LV60K
            - WAV2VEC2_ASR_LARGE_LV60K_10M
            - WAV2VEC2_ASR_LARGE_LV60K_100H
            - WAV2VEC2_ASR_LARGE_LV60K_960H
            - WAV2VEC2_XLSR53
            - WAV2VEC2_XLSR_300M
            - WAV2VEC2_XLSR_1B
            - WAV2VEC2_XLSR_2B
            - HUBERT_LARGE
            - HUBERT_XLARGE
            - HUBERT_ASR_LARGE
            - HUBERT_ASR_XLARGE
            - WAVLM_LARGE
        """
        model = utils._get_model(self._model_type, self._params)
        state_dict = self._get_state_dict(dl_kwargs)
        model.load_state_dict(state_dict)
        if self._normalize_waveform:
            model = utils._extend_model(model, normalize_waveform=True)
        model.eval()
        return model




WAV2VEC2_XLSR_300M = Wav2Vec2Bundle(
    "wav2vec2_xlsr_300m.pth",
    {
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": True,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.0,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.0,
        "aux_num_out": None,
    },
    _model_type="Wav2Vec2",
    _sample_rate=16000,
    _normalize_waveform=True,
)
WAV2VEC2_XLSR_300M.__doc__ = """XLS-R model with 300 million parameters,
pre-trained on 436,000 hours of unlabeled audio from multiple datasets (
*Multilingual LibriSpeech* :cite:`Pratap_2020`,
*CommonVoice* :cite:`ardila2020common`,
*VoxLingua107* :cite:`valk2021voxlingua107`,
*BABEL* :cite:`Gales2014SpeechRA`, and
*VoxPopuli* :cite:`voxpopuli`) in 128 languages,
not fine-tuned.

Originally published by the authors of *XLS-R* :cite:`babu2021xls` under MIT License and
redistributed with the same license.
[`License <https://github.com/facebookresearch/fairseq/blob/30c912b73c0f88d41171879b2f03226a171004ef/LICENSE>`__,
`Source <https://github.com/facebookresearch/fairseq/tree/30c912b73c0f88d41171879b2f03226a171004ef/examples/wav2vec/xlsr#xls-r>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2Bundle` for usage details.
"""  # noqa: E501