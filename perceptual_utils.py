import copy
from typing import Any, Dict, List, Optional, Union

import torch
import torchaudio.transforms as T
from torch import Tensor
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CustomizedClapFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a CLAP feature extractor.

    This feature extractor inherits from `~feature_extraction_sequence_utils.SequenceFeatureExtractor` which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the *Short Time
    Fourier Transform* (STFT) which should match pytorch's `torch.stft` equivalent.

    Args:
        feature_size (`int`, *optional*, defaults to 64):
            The feature dimension of the extracted Mel spectrograms. This corresponds to the number of mel filters
            (`n_mels`).
        sampling_rate (`int`, *optional*, defaults to 48000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz). This only serves
            to warn users if the audio fed to the feature extractor does not have the same sampling rate.
        hop_length (`int`,*optional*, defaults to 480):
            Length of the overlaping windows for the STFT used to obtain the Mel Spectrogram. The audio will be split
            in smaller `frames` with a step of `hop_length` between each frame.
        max_length_s (`int`, *optional*, defaults to 10):
            The maximum input length of the model in seconds. This is used to pad the audio.
        fft_window_size (`int`, *optional*, defaults to 1024):
            Size of the window (in samples) on which the Fourier transform is applied. This controls the frequency
            resolution of the spectrogram. 400 means that the fourrier transform is computed on windows of 400 samples.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the attention masks coresponding to the input.
        frequency_min (`float`, *optional*, defaults to 0):
            The lowest frequency of interest. The STFT will not be computed for values below this.
        frequency_max (`float`, *optional*, defaults to 14000):
            The highest frequency of interest. The STFT will not be computed for values above this.
        top_db (`float`, *optional*):
            The highest decibel value used to convert the mel spectrogram to the log scale. For more details see the
            `audio_utils.power_to_db` function
        truncation (`str`, *optional*, defaults to `"fusion"`):
            Truncation pattern for long audio inputs. Two patterns are available:
                - `fusion` will use `_random_mel_fusion`, which stacks 3 random crops from the mel spectrogram and a
                  downsampled version of the entire mel spectrogram.
            If `config.fusion` is set to True, shorter audios also need to to return 4 mels, which will just be a copy
            of the original mel obtained from the padded audio.
                - `rand_trunc` will select a random crop of the mel spectrogram.
        padding (`str`, *optional*, defaults to `"repeatpad"`):
               Padding pattern for shorter audio inputs. Three patterns were originally implemented:
                - `repeatpad`: the audio is repeated, and then padded to fit the `max_length`.
                - `repeat`: the audio is repeated and then cut to fit the `max_length`
                - `pad`: the audio is padded.
    """

    model_input_names = ["input_features", "is_longer"]

    def __init__(
        self,
        feature_size=64,
        sampling_rate=48_000,
        hop_length=480,
        max_length_s=10,
        fft_window_size=1024,
        padding_value=0.0,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        frequency_min: float = 50,
        frequency_max: float = 14_000,
        top_db: int = None,
        truncation: str = "rand_trunc",
        padding: str = "repeatpad",
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.top_db = top_db
        self.truncation = truncation
        self.padding = padding
        self.fft_window_size = fft_window_size
        self.nb_frequency_bins = (fft_window_size >> 1) + 1
        self.hop_length = hop_length
        self.max_length_s = max_length_s
        self.nb_max_samples = max_length_s * sampling_rate
        self.sampling_rate = sampling_rate
        self.frequency_min = frequency_min
        self.frequency_max = frequency_max

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance, excpet for the
            mel filter banks, which do not need to be saved or printed as they are too long.
        """
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        return output

    def _torch_extract_fbank_features(self, waveform: Tensor) -> Tensor:
        """
        Compute the log-mel spectrogram of the provided `waveform` using the Hann window.
        """
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.fft_window_size,
            hop_length=self.hop_length,
            n_mels=self.feature_size,
            f_min=self.frequency_min,
            f_max=self.frequency_max,
            mel_scale="slaney",
            norm = "slaney"
        )
        transform = T.AmplitudeToDB()
        return transform(mel_spectrogram(waveform).mT)

    def _get_input_mel(self, waveform: Tensor, max_length: int, truncation: str, padding: str) -> Tensor:
        """
        Extracts the mel spectrogram and prepares it for the mode based on the `truncation` and `padding` arguments.
        """
        if waveform.size(0) > max_length:
            if truncation == "fusion":
                raise NotImplementedError("Truncation pattern 'fusion' is not supported with torch")
            elif truncation == "rand_trunc":
                # random crop to max_length
                idx = torch.randint(0, waveform.size(0) - max_length + 1, (1,))
                waveform = waveform[idx : idx + max_length]
            else:
                raise NotImplementedError(f"Truncation pattern '{truncation}' not implemented")

        elif waveform.size(0) < max_length:
            # pad the audio

            if padding == "repeat":
                n_repeat = (max_length // waveform.size(0)) + 1
                waveform = waveform.repeat(n_repeat)[:max_length]
            elif padding == "repeatpad":
                n_repeat = (max_length // waveform.size(0))
                waveform = waveform.repeat(n_repeat)
            padding_diff = max_length - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, padding_diff), mode="constant", value=self.padding_value)
        # extract mel spectrogram
        return self._torch_extract_fbank_features(waveform)

    def __call__(
        self,
        raw_speech: Union[Tensor, List[Tensor]],
        truncation: str = None,
        padding: Optional[str] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, Tensor]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).
        """
        truncation = truncation if truncation is not None else self.truncation
        padding = padding if padding else self.padding

        if isinstance(raw_speech, Tensor):
            raw_speech = [raw_speech]
        elif not isinstance(raw_speech, list):
            raise ValueError("raw_speech must be a Tensor or a list of Tensors.")

        # always return batch
        is_longer = []
        input_mel = []
        for waveform in raw_speech:
            max_len = max_length if max_length is not None else self.nb_max_samples
            mel = self._get_input_mel(waveform, max_len, truncation, padding)
            input_mel.append(mel.unsqueeze(0))
            is_longer.append(waveform.size(0) > max_len if max_len is not None else False)

        input_mel = torch.cat(input_mel, dim=0)
        is_longer = [[longer] for longer in is_longer]

        input_features = {"input_features": input_mel.expand(1,1,-1,-1)
, "is_longer": is_longer}
        input_features = BatchFeature(input_features)

        if return_tensors is not None:
            input_features = input_features.convert_to_tensors(return_tensors)

        return input_features