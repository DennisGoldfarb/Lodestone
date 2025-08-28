from lodestone.data import tokenize_sequence, TOKEN_TO_IDX


def test_ac_tokenization():
    seq = 'ac-AC'
    indices = tokenize_sequence(seq)
    expected = [TOKEN_TO_IDX['ac-'], TOKEN_TO_IDX['A'], TOKEN_TO_IDX['C']]
    assert indices == expected
