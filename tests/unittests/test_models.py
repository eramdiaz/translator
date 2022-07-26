import pytest
import torch
from translator.models import *


D_MODEL = 64
D_K = 8
D_V = 8
H = 8
SEQ_LEN = 16
D_FF = 128
N = 2


@pytest.fixture
def input_1():
    return torch.load('tests/material/enc_4_16_64.pt')


@pytest.fixture
def input_2():
    return torch.load('tests/material/dec_4_16_64.pt')


@pytest.fixture
def sent_1():
    return torch.load('tests/material/int_4_16_1.pt')


@pytest.fixture
def sent_2():
    return torch.load('tests/material/int_4_16_2.pt')


@pytest.fixture
def enc_mask():
    return torch.load('tests/material/encmask_4_16_16.pt')


@pytest.fixture
def encdec_mask():
    return torch.load('tests/material/encdecmask_4_16_16.pt')


@pytest.fixture
def dec_mask():
    return torch.load('tests/material/decmask_4_16_16.pt')


class TestEncoderCell:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.encoder_cell = EncoderCell(D_MODEL, D_K, D_V, H, SEQ_LEN, D_FF)

    def test_forward(self, input_1):
        assert self.encoder_cell(input_1, input_1, input_1).shape \
               == torch.Size((4, 16, 64))

    def test_mask(self, input_1, enc_mask):
        self.encoder_cell.eval()
        outputs = self.encoder_cell(input_1, input_1, input_1, enc_mask['mask'])

        eq_outputs = []
        for inp, mask_len in zip(input_1, enc_mask['enc_lengths']):
            masked_inp = inp[:mask_len.item()]
            eq_outputs.append(self.encoder_cell(masked_inp, masked_inp, masked_inp))

        for output, mask_len, eq_output in zip(outputs, enc_mask['enc_lengths'], eq_outputs):
            closeness = torch.isclose(output[:mask_len.item(), :], eq_output,
                                      rtol=1e-7, atol=1e-6)
            assert (closeness == False).sum().item() == 0


class TestDecoderCell:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.decoder_cell = DecoderCell(D_MODEL, D_K, D_V, H, SEQ_LEN, D_FF)

    def test_forward(self, input_1, input_2):
        assert self.decoder_cell(input_2, input_2, input_2, input_1, input_1).shape \
               == torch.Size((4, 16, 64))

    def test_mask(self, input_1, input_2, encdec_mask, dec_mask):
        self.decoder_cell.eval()
        outputs = self.decoder_cell(input_2, input_2, input_2, input_1, input_1,
                                    encdec_mask['mask'], dec_mask['mask'])

        eq_outputs = []
        for inp, out, emask_len, dmask_len in zip(input_1, input_2, encdec_mask['enc_lengths'],
                                                  dec_mask['dec_lengths']):
            masked_inp = inp[:emask_len.item(), :]
            masked_out = out[:dmask_len.item(), :]
            eq_outputs.append(self.decoder_cell(masked_out, masked_out, masked_out,
                                                        masked_inp, masked_inp))
        for output, eq_output, mask in zip(outputs, eq_outputs, dec_mask['dec_lengths']):
            closeness = torch.isclose(output[:mask.item(), :], eq_output, rtol=1e-7, atol=1e-6)
            assert (closeness == False).sum().item() == 0


class TestTransformer:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transformer = Transformer(N, 'tests/material/mock_tokenizer.model', SEQ_LEN,
                                              D_MODEL, D_K, D_V, H, D_FF)

    def test_forward(self, sent_1, sent_2):
        assert self.transformer(sent_1, sent_2).shape == \
               torch.Size((4, 16, self.transformer.tokenizer.vocab_size()))

    def test_mask(self, sent_1, sent_2, enc_mask, encdec_mask, dec_mask):
        self.transformer.eval()
        outputs = self.transformer(sent_1, sent_2, enc_mask['mask'],
                                   encdec_mask['mask'], dec_mask['mask'])

        eq_outputs = []
        for inp, out, e_mask, g_mask in zip(sent_1, sent_2, enc_mask['enc_lengths'],
                                            dec_mask['dec_lengths']):
            masked_inp = inp[:e_mask.item()]
            masked_out = out[:g_mask.item()]
            eq_outputs.append(self.transformer(masked_inp, masked_out))

        for output, eq_output, mask in zip(outputs, eq_outputs, dec_mask['dec_lengths']):
            closeness = torch.isclose(output[:mask.item(), :], eq_output, rtol=1e-7, atol=1e-5)
            assert (closeness == False).sum().item() == 0
