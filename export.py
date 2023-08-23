import torch
import torchaudio
from seamless_communication.models.inference.translator import Translator

from fast_seamlessm4t.models import OnnxUnitYX2TModel

device = torch.device("cuda")

# Initialize a Translator object with a multitask model, vocoder on the GPU.
translator = Translator("seamlessM4T_medium", "vocoder_36langs", device)

input_text = "Hello, world."
tgt_lang = "eng"
src_lang = "eng"


onnx_model = OnnxUnitYX2TModel(translator, input_modality="text")

seqs, seq_lens = onnx_model.tokenize(input_text, src_lang)

torch.onnx.export(
    onnx_model,
    (seqs,),
    "test.onnx",
    input_names=["seqs"],
    output_names=["encoder_output"],
    opset_version=16,
    dynamic_axes={
        "seqs": {1: "max_seq_len"},
        "encoder_output": {1: "max_seq_len"},
    },
)

# translated_text, wav, sr = translator.predict(
#     input_text, "t2st", tgt_lang=tgt_lang, src_lang=src_lang
# )

# torchaudio.save(
#     "test.mp3",
#     wav[0].cpu(),
#     sample_rate=sr,
# )
