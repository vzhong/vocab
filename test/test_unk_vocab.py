from vocab import UnkVocab, OutOfVocabularyException
from test.test_vocab import TestVocab
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

    def test_pad(self):
        sentences = [
            ['a', 'b', 'c'],
            ['d'],
        ]
        pad = '<pad>'

        # pad OOV
        with self.assertRaises(OutOfVocabularyException):
            self.V(['a', 'b', 'c', 'd']).word2padded_index(sentences, pad=pad, train=False, enforce_end_pad=False)
        # input OOV
        padded, lens = self.V(['a', '<pad>']).word2padded_index(sentences, pad=pad, train=False, enforce_end_pad=False)
        self.assertEqual([[1, 0, 0], [0, 2, 2]], padded)
        self.assertEqual([3, 1], lens)

        # train adds the input
        padded, lens = self.V(['<pad>']).word2padded_index(sentences, pad=pad, train=True, enforce_end_pad=False)
        self.assertEqual([[2, 3, 4], [5, 1, 1]], padded)
        self.assertEqual([3, 1], lens)

        # # train adds the pad
        padded, lens = self.V(['b', 'a', 'c', 'd']).word2padded_index(sentences, pad=pad, train=True, enforce_end_pad=False)
        self.assertEqual([[2, 1, 3], [4, 5, 5]], padded)
        self.assertEqual([3, 1], lens)

        # # add pad to end
        padded, lens = self.V().word2padded_index(sentences, pad=pad, train=True, enforce_end_pad=True)
        self.assertEqual([[1, 2, 3, 4], [5, 4, 4, 4]], padded)
        self.assertEqual([4, 2], lens)

    def test_depad(self):
        indices = [
            [0, 1, 2, 1],
            [3, 2, 0, 0],
        ]
        pad = '<pad>'
        with self.assertRaises(OutOfVocabularyException):  # pad OOV
            self.V(['a', 'b', 'c']).padded_index2word(indices, pad=pad)
        with self.assertRaises(OutOfVocabularyException):  # pad OOV
            self.V(['a', 'b', 'c', 'd']).padded_index2word(indices, pad=pad)
        depadded = self.V(['a', '<pad>', 'b']).padded_index2word(indices, pad=pad)
        self.assertEqual([['<unk>', 'a'], ['b']], depadded)


if __name__ == '__main__':
    unittest.main()
