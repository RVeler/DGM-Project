import copy
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torchaudio.transforms as T
from torch import Tensor
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from transformers.utils import logging, PaddingStrategy, TensorType, is_tf_tensor, is_torch_tensor, to_numpy

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
    
class CustomizedSequenceFeatureExtractor(FeatureExtractionMixin):
    """
    This is a general feature extraction class for speech recognition.

    Args:
        feature_size (`int`):
            The feature dimension of the extracted features.
        sampling_rate (`int`):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`):
            The value that is used to fill the padding values / vectors.
    """

    def __init__(self, feature_size: int, sampling_rate: int, padding_value: float, **kwargs):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value

        self.padding_side = kwargs.pop("padding_side", "right")
        self.return_attention_mask = kwargs.pop("return_attention_mask", True)

        super().__init__(**kwargs)

    def pad(
        self,
        processed_features: Union[
            BatchFeature,
            List[BatchFeature],
            Dict[str, BatchFeature],
            Dict[str, List[BatchFeature]],
            List[Dict[str, BatchFeature]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """
        Pad input values / input vectors or a batch of input values / input vectors up to predefined length or to the
        max sequence length in the batch.

        Padding side (left/right) padding values are defined at the feature extractor level (with `self.padding_side`,
        `self.padding_value`)

        <Tip>

        If the `processed_features` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
        result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
        PyTorch tensors, you will lose the specific device of your tensors however.

        </Tip>

        Args:
            processed_features ([`BatchFeature`], list of [`BatchFeature`], `Dict[str, List[float]]`, `Dict[str, List[List[float]]` or `List[Dict[str, List[float]]]`):
                Processed inputs. Can represent one input ([`BatchFeature`] or `Dict[str, List[float]]`) or a batch of
                input values / vectors (list of [`BatchFeature`], *Dict[str, List[List[float]]]* or *List[Dict[str,
                List[float]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.

                Instead of `List[float]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors),
                see the note above for the return type.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(processed_features, (list, tuple)) and isinstance(processed_features[0], (dict, BatchFeature)):
            processed_features = {
                key: [example[key] for example in processed_features] for key in processed_features[0].keys()
            }

        # The model's main input name, usually `input_values`, has be passed for padding
        if self.model_input_names[0] not in processed_features:
            raise ValueError(
                "You should supply an instance of `transformers.BatchFeature` or list of `transformers.BatchFeature`"
                f" to this method that includes {self.model_input_names[0]}, but you provided"
                f" {list(processed_features.keys())}"
            )

        required_input = processed_features[self.model_input_names[0]]
        return_attention_mask = (
            return_attention_mask if return_attention_mask is not None else self.return_attention_mask
        )

        if len(required_input) == 0:
            if return_attention_mask:
                processed_features["attention_mask"] = []
            return processed_features

        # If we have PyTorch/TF tensors or lists as inputs, we cast them as Numpy arrays
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]

        if return_tensors is None:
            if is_tf_tensor(first_element):
                return_tensors = "tf"
            elif is_torch_tensor(first_element):
                return_tensors = "pt"
            elif isinstance(first_element, (int, float, list, tuple, np.ndarray)):
                return_tensors = "np"
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, pytorch or tensorflow object."
                )

        # for key, value in processed_features.items():
        #     if isinstance(value[0], (int, float)):
        #         processed_features[key] = to_numpy(value)
        #     else:
        #         processed_features[key] = [to_numpy(v) for v in value]

        # Convert padding_strategy in PaddingStrategy
        padding_strategy = self._get_padding_strategies(padding=padding, max_length=max_length)

        required_input = processed_features[self.model_input_names[0]]

        batch_size = len(required_input)
        if not all(len(v) == batch_size for v in processed_features.values()):
            raise ValueError("Some items in the output dictionary have a different batch size than others.")

        truncated_inputs = []
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in processed_features.items()}
            # truncation
            inputs_slice = self._truncate(
                inputs,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                truncation=truncation,
            )
            truncated_inputs.append(inputs_slice)

        if padding_strategy == PaddingStrategy.LONGEST:
            # make sure that `max_length` cannot be longer than the longest truncated length
            max_length = max(len(input_slice[self.model_input_names[0]]) for input_slice in truncated_inputs)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            # padding
            outputs = self._pad(
                truncated_inputs[i],
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                if value.dtype is np.dtype(np.float64):
                    value = value.astype(np.float32)
                # if value.dtype is np.dtype(torch.float64): #my adding
                #     value = value.astype(torch.float32)
                batch_outputs[key].append(value)

        batch_outputs["input_values"] = torch.cat(batch_outputs["input_values"], dim=0).unsqueeze(0)
        batch_outputs["attention_mask"] = torch.cat(batch_outputs["attention_mask"], dim=0).unsqueeze(0)
        return BatchFeature(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        processed_features: Union[Dict[str, np.ndarray], BatchFeature],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            processed_features (`Union[Dict[str, np.ndarray], BatchFeature]`):
                Dictionary of input values (`np.ndarray[float]`) / input vectors (`List[np.ndarray[float]]`) or batch
                of inputs values (`List[np.ndarray[int]]`) / input vectors (`List[np.ndarray[int]]`)
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see below)
            padding_strategy (`PaddingStrategy`, *optional*, default to `PaddingStrategy.DO_NOT_PAD`):
                PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The feature_extractor padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of (`int`, *optional*):
                Integer if set will pad the sequence to a multiple of the provided value. This is especially useful to
                enable the use of Tensor Core on NVIDIA hardware with compute capability `>= 7.5` (Volta), or on TPUs
                which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Set to False to avoid returning attention mask (default: set to model specifics)
        """
        required_input = processed_features[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) < max_length

        if return_attention_mask and "attention_mask" not in processed_features:
            processed_features["attention_mask"] = torch.ones(len(required_input), requires_grad=True)

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == "right":
                if return_attention_mask:
                    processed_features["attention_mask"] = torch.nn.functiona.pad(
                        processed_features["attention_mask"], (0, difference)
                    )
                padding_shape = ((0, difference), (0, 0)) if self.feature_size > 1 else (0, difference)
                processed_features[self.model_input_names[0]] = np.pad(
                    required_input, padding_shape, "constant", constant_values=self.padding_value
                )
            elif self.padding_side == "left":
                if return_attention_mask:
                    processed_features["attention_mask"] = torch.nn.functiona.pad(
                        processed_features["attention_mask"], (difference, 0)
                    )
                padding_shape = ((difference, 0), (0, 0)) if self.feature_size > 1 else (difference, 0)
                processed_features[self.model_input_names[0]] = torch.nn.functiona.pad(
                    required_input, padding_shape, "constant", constant_values=self.padding_value
                )
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return processed_features

    def _truncate(
        self,
        processed_features: Union[Dict[str, np.ndarray], BatchFeature],
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        truncation: Optional[bool] = None,
    ):
        """
        Truncate inputs to predefined length or max length in the batch

        Args:
            processed_features(`Union[Dict[str, np.ndarray], BatchFeature]`):
                Dictionary of input values (`np.ndarray[float]`) / input vectors (`List[np.ndarray[float]]`) or batch
                of inputs values (`List[np.ndarray[int]]`) / input vectors (`List[np.ndarray[int]]`)
            max_length (`int`, *optional*):
                maximum length of the returned list and optionally padding length (see below)
            pad_to_multiple_of (`int`, *optional*) :
                Integer if set will pad the sequence to a multiple of the provided value. This is especially useful to
                enable the use of Tensor Core on NVIDIA hardware with compute capability `>= 7.5` (Volta), or on TPUs
                which benefit from having sequence lengths be a multiple of 128.
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
        """
        if not truncation:
            return processed_features
        elif truncation and max_length is None:
            raise ValueError("When setting ``truncation=True``, make sure that ``max_length`` is defined.")

        required_input = processed_features[self.model_input_names[0]]

        # find `max_length` that fits `pad_to_multiple_of`
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_truncated = len(required_input) > max_length

        if needs_to_be_truncated:
            processed_features[self.model_input_names[0]] = processed_features[self.model_input_names[0]][:max_length]
            if "attention_mask" in processed_features:
                processed_features["attention_mask"] = processed_features["attention_mask"][:max_length]

        return processed_features

    def _get_padding_strategies(self, padding=False, max_length=None):
        """
        Find the correct padding strategy
        """

        # Get padding strategy
        if padding is not False:
            if padding is True:
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                raise ValueError(
                    f"When setting ``padding={PaddingStrategy.MAX_LENGTH}``, make sure that max_length is defined"
                )

        # Test if we have a padding value
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (self.padding_value is None):
            raise ValueError(
                "Asking to pad but the feature_extractor does not have a padding value. Please select a value to use"
                " as `padding_value`. For example: `feature_extractor.padding_value = 0.0`."
            )

        return padding_strategy
    
class CustomizedEncodecFeatureExtractor(CustomizedSequenceFeatureExtractor):
    """
    Constructs an EnCodec feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Instantiating a feature extractor with the defaults will yield a similar configuration to that of the
    [facebook/encodec_24khz](https://huggingface.co/facebook/encodec_24khz) architecture.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features. Use 1 for mono, 2 for stereo.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values.
        chunk_length_s (`float`, *optional*):
            If defined the audio is pre-processed into chunks of lengths `chunk_length_s` and then encoded.
        overlap (`float`, *optional*):
            Defines the overlap between each chunk. It is used to compute the `chunk_stride` using the following
            formulae : `int((1.0 - self.overlap) * self.chunk_length)`.
    """

    model_input_names = ["input_values", "padding_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 24000,
        padding_value: float = 0.0,
        chunk_length_s: float = None,
        overlap: float = None,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.chunk_length_s = chunk_length_s
        self.overlap = overlap

    # This is a property because you might want to change the chunk_length_s on the fly
    @property
    def chunk_length(self) -> Optional[int]:
        if self.chunk_length_s is None:
            return None
        else:
            return int(self.chunk_length_s * self.sampling_rate)

    # This is a property because you might want to change the chunk_length_s on the fly
    @property
    def chunk_stride(self) -> Optional[int]:
        if self.chunk_length_s is None or self.overlap is None:
            return None
        else:
            return max(1, int((1.0 - self.overlap) * self.chunk_length))

    def __call__(
        self,
        raw_audio: Union[Tensor, List[Tensor]],
        padding: Optional[Union[bool, str, PaddingStrategy]] = None,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_audio (`List[Tensor]`, `List[List[float]]`):
                The sequence or batch of sequences to be processed. Each sequence can be a tensor or a list of float
                values.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            truncation (`bool`, *optional*, defaults to `False`):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided audio input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if padding and truncation:
            raise ValueError("Both padding and truncation were set. Make sure you only set one.")
        elif padding is None:
            # by default let's pad the inputs
            padding = True

        is_batched = bool(
            isinstance(raw_audio, (list, tuple)) and (isinstance(raw_audio[0], (list, tuple, torch.Tensor)))
        )

        if is_batched:
            raw_audio = [torch.tensor(audio, dtype=torch.float32).T for audio in raw_audio]
        elif not is_batched and not isinstance(raw_audio, torch.Tensor):
            raw_audio = torch.tensor(raw_audio, dtype=torch.float32)
        elif isinstance(raw_audio, torch.Tensor) and raw_audio.dtype is not torch.float32:
            raw_audio = raw_audio.float()

        # always return batch
        if not is_batched:
            raw_audio = [raw_audio.T]

        # verify inputs are valid
        for idx, example in enumerate(raw_audio):
            if example.dim() > 2:
                raise ValueError(f"Expected input shape (channels, length) but got shape {example.shape}")
            if self.feature_size == 1 and example.dim() != 1:
                raise ValueError(f"Expected mono audio but example has {example.shape[-1]} channels")
            if self.feature_size == 2 and example.shape[-1] != 2:
                raise ValueError(f"Expected stereo audio but example has {example.shape[-1]} channels")

        padded_inputs = None
        input_values = BatchFeature({"input_values": raw_audio})
        if self.chunk_stride is not None and self.chunk_length is not None and max_length is None:
            if truncation:
                max_length = min(tensor.shape[0] for tensor in raw_audio)
                nb_step = int(torch.floor(torch.tensor(max_length, dtype=torch.float32) / self.chunk_stride))
                max_length = (nb_step - 1) * self.chunk_stride + self.chunk_length
            elif padding:
                max_length = max(tensor.shape[0] for tensor in raw_audio)
                nb_step = int(torch.ceil(torch.tensor(max_length, dtype=torch.float32) / self.chunk_stride))
                max_length = (nb_step - 1) * self.chunk_stride + self.chunk_length
                padding = "max_length"
            else:
                padded_inputs = input_values

        # normal padding on batch
        if padded_inputs is None:
            padded_inputs = self.pad(
                input_values,
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                return_attention_mask=padding,
            )
            if padding:
                padded_inputs["padding_mask"] = padded_inputs.pop("attention_mask")

        input_values = []
        for example in padded_inputs.pop("input_values"):
            if self.feature_size == 1:
                example = example[..., None]
            input_values.append(example.T)

        input_values = torch.cat(input_values, dim=0).unsqueeze(0)
        padded_inputs["input_values"] = input_values
        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs