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
VOCAB_SIZE = 128


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


class TestEncoderCell:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.encoder_cell = EncoderCell(D_MODEL, D_K, D_V, H, SEQ_LEN, D_FF)

    def test_forward(self, input_1):
        assert self.encoder_cell(input_1, input_1, input_1).shape \
               == torch.Size((4, 16, 64))

    def test_mask(self, input_1):
        self.encoder_cell.eval()
        masks = torch.LongTensor([8, 4, 16, 12])
        outputs = self.encoder_cell(input_1, input_1, input_1, masks)

        eq_outputs = []
        for inp, mask in zip(input_1, masks):
            masked_inp = inp[:mask.item()]
            eq_outputs.append(self.encoder_cell(masked_inp, masked_inp, masked_inp))

        for output, mask, eq_output in zip(outputs, masks, eq_outputs):
            closeness = torch.isclose(output[:mask, :], eq_output, rtol=1e-7, atol=1e-6)
            assert (closeness == False).sum().item() == 0


class TestDecoderCell:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.decoder_cell = DecoderCell(D_MODEL, D_K, D_V, H, SEQ_LEN, D_FF)

    def test_forward(self, input_1, input_2):
        assert self.decoder_cell(input_2, input_2, input_2, input_1, input_1).shape \
               == torch.Size((4, 16, 64))

    def test_mask(self, input_1, input_2):
        self.decoder_cell.eval()
        e_masks = torch.LongTensor([8, 4, 16, 12])
        g_masks = torch.LongTensor([9, 7, 10, 5])
        outputs = self.decoder_cell(input_2, input_2, input_2, input_1,
                                            input_1, e_masks, g_masks)

        eq_outputs = []
        for inp, out, e_mask, g_mask in zip(input_1, input_2, e_masks, g_masks):
            masked_inp = inp[:e_mask, :]
            masked_out = out[:g_mask, :]
            eq_outputs.append(self.decoder_cell(masked_out, masked_out, masked_out,
                                                        masked_inp, masked_inp))
        for output, eq_output, mask in zip(outputs, eq_outputs, g_masks):
            closeness = torch.isclose(output[:mask, :], eq_output, rtol=1e-7, atol=1e-6)
            assert (closeness == False).sum().item() == 0


class TestTransformer:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transformer = Transformer(N, VOCAB_SIZE, SEQ_LEN, D_MODEL, D_K, D_V, H, D_FF)

    def test_forward(self, sent_1, sent_2):
        assert self.transformer(sent_1, sent_2).shape == torch.Size((4, 16, 128))

    def test_mask(self, sent_1, sent_2):
        self.transformer.eval()
        e_masks = torch.LongTensor([8, 4, 16, 12])
        g_masks = torch.LongTensor([9, 7, 10, 5])
        outputs = self.transformer(sent_1, sent_2, e_masks, g_masks)

        eq_outputs = []
        for inp, out, e_mask, g_mask in zip(sent_1, sent_2, e_masks, g_masks):
            masked_inp = inp[:e_mask]
            masked_out = out[:g_mask]
            eq_outputs.append(self.transformer(masked_inp, masked_out))

        for output, eq_output, mask in zip(outputs, eq_outputs, g_masks):
            closeness = torch.isclose(output[:mask, :], eq_output, rtol=1e-7, atol=1e-6)
            assert (closeness == False).sum().item() == 0