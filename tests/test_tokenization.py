from lodestone.data import tokenize_sequence, TOKEN_TO_IDX


def test_ac_tokenization():
    seq = 'ac-AC'
    indices = tokenize_sequence(seq)
    expected = [TOKEN_TO_IDX['ac-'], TOKEN_TO_IDX['A'], TOKEN_TO_IDX['C']]
    assert indices == expected


def test_camC_tokenization():
    seq = 'ACcamCG'
    indices = tokenize_sequence(seq)
    expected = [
        TOKEN_TO_IDX['A'],
        TOKEN_TO_IDX['C'],
        TOKEN_TO_IDX['camC'],
        TOKEN_TO_IDX['G'],
    ]
    assert indices == expected


def test_oxM_tokenization():
    seq = 'ACoxMG'
    indices = tokenize_sequence(seq)
    expected = [
        TOKEN_TO_IDX['A'],
        TOKEN_TO_IDX['C'],
        TOKEN_TO_IDX['oxM'],
        TOKEN_TO_IDX['G'],
    ]
    assert indices == expected
