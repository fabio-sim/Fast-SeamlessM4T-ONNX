from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from fairseq2.memory import MemoryBlock
from seamless_communication.models.inference.translator import Translator
from seamless_communication.models.unity.model import UnitYX2TModel


class UnitYX2TEncoder(nn.Module):
    def __init__(self, translator: Translator, input_modality: str = "text"):
        super().__init__()

        assert input_modality in {"speech", "text"}
        self.input_modality = input_modality

        self.text_tokenizer = translator.text_tokenizer
        self.collate = translator.collate
        self.decode_audio = translator.decode_audio
        self.convert_to_fbank = translator.convert_to_fbank

        self.device = translator.device

        model = translator.model
        self.model = UnitYX2TModel(
            encoder_frontend=model.speech_encoder_frontend
            if input_modality == "speech"
            else model.text_encoder_frontend,
            encoder=model.speech_encoder
            if input_modality == "speech"
            else model.text_encoder,
            decoder_frontend=model.text_decoder_frontend,
            decoder=model.text_decoder,
            final_proj=model.final_proj,
            pad_idx=model.pad_idx,
        )

    def forward(
        self, seqs: torch.Tensor, seq_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        encoder_output, encoder_padding_mask = self.model.encode(seqs, seq_lens)
        return encoder_output

    # Text preprocessing
    def tokenize(self, text: str, lang: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.text_tokenizer.create_encoder(
            lang=lang, mode="source", device=self.device
        )

        ids = tokenizer(text)
        src = self.collate(ids)
        return src["seqs"], src["seq_lens"]

    # Speech preprocessing
    def audio_from_file(self, filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
        with Path(filename).open("rb") as fb:
            block = MemoryBlock(fb.read())

        decoded_audio = self.decode_audio(block)
        src = self.collate(self.convert_to_fbank(decoded_audio))["fbank"]
        return src["seqs"], src["seq_lens"]
