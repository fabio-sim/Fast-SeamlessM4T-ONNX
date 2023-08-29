import torch
import torchaudio
from fairseq2.generation import Seq2SeqGenerator, SequenceGeneratorOptions
from fairseq2.nn.incremental_state import IncrementalStateBag
from seamless_communication.models.inference.translator import Translator

from fast_seamlessm4t.models import UnitYX2TEncoder

device = torch.device("cuda")

# Initialize a Translator object with a multitask model, vocoder on the GPU.
translator = Translator("seamlessM4T_medium", "vocoder_36langs", device)

input_text = "Hello, world."
tgt_lang = "eng"
src_lang = "eng"

translated_text, wav, sr = translator.predict(
    input_text, "t2st", tgt_lang=tgt_lang, src_lang=src_lang
)


# torchaudio.save(
#     "test.wav",
#     wav[0].cpu(),
#     sample_rate=sr,
# )

# Text Encoder
text_encoder = UnitYX2TEncoder(translator, input_modality="text")

seqs, seq_lens = text_encoder.tokenize(input_text, src_lang)

torch.onnx.export(
    text_encoder,
    (seqs,),
    "text_encoder.onnx",
    input_names=["seqs"],
    output_names=["encoder_output"],
    opset_version=16,
    dynamic_axes={
        "seqs": {1: "max_seq_len"},
        "encoder_output": {1: "max_seq_len"},
    },
)

# Speech Encoder
speech_encoder = UnitYX2TEncoder(translator, input_modality="speech")

seqs, seq_lens = speech_encoder.audio_from_file("test.wav")

torch.onnx.export(
    speech_encoder,
    (seqs,),
    "speech_encoder.onnx",
    input_names=["seqs"],
    output_names=["encoder_output"],
    opset_version=16,
    dynamic_axes={
        "seqs": {1: "max_seq_len"},
        "encoder_output": {1: "max_seq_len"},
    },
)

# Decoder
