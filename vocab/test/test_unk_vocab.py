from vocab import UnkVocab, OutOfVocabularyException
from vocab.test.test_vocab import TestVocab
import unittest


class TestUnkVocab(TestVocab):

    V = UnkVocab

    def test_repr(self):
        self.assertEqual("UnkVocab({})".format(len(self.v)), repr(self.v))

    def test_word2index(self):
        self.assertEqual(2, self.v.word2index('bb'))
        self.assertListEqual(
            [1, 3, 2],
            self.v.word2index(['a', 'ccc', 'bb'])
        )
        self.assertEqual(0, self.v.word2index('dddd'))
        self.assertListEqual(
            [1, 0, 2],
            self.v.word2index(['a', 'dddd', 'bb'])
        )

    def test_index2word(self):
        self.assertEqual('bb', self.v.index2word(2))
        self.assertListEqual(
            ['a', 'ccc', 'bb'],
            self.v.index2word([1, 3, 2])
        )
        with self.assertRaises(OutOfVocabularyException):
            self.v.index2word(-1)
            self.v.index2word(3)
            self.v.index2word([0, 3])


if __name__ == '__main__':
    unittest.main()
