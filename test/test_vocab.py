from vocab import Vocab, OutOfVocabularyException
import unittest
from copy import deepcopy


class TestVocab(unittest.TestCase):

    V = Vocab

    def setUp(self):
        self.v = self.V(['a', 'bb', 'ccc'])

    def test_repr(self):
        self.assertEqual("Vocab({})".format(len(self.v)), repr(self.v))

    def test_eq(self):
        a = deepcopy(self.v)
        self.assertEqual(self.v, a)
        a.counts['a'] += 1
        self.assertNotEqual(self.v, a)

    def test_contains_same_content(self):
        self.assertTrue(self.v.contains_same_content(self.V(['a', 'ccc', 'bb'])))
        self.assertFalse(self.v.contains_same_content(self.V(['a', 'bb'])))
        self.assertFalse(self.v.contains_same_content(self.V(['a', 'ccc'])))
        self.assertFalse(self.v.contains_same_content(self.V(['a', 'bb', 'ccc', 'dddd'])))

    def test_copy(self):
        a = deepcopy(self.v)
        b = self.v.copy(keep_words=True)
        self.assertEqual(self.v, a)

        c = self.v.copy(keep_words=False)
        
        a.counts['a'] += 1
        self.assertNotEqual(self.v, a)

    def test_to_dict(self):
        self.assertDictEqual(
            {
                'index2word': ['a', 'bb', 'ccc'],
                'counts': {'a': 1, 'bb': 1, 'ccc': 1},
            },
            self.v.to_dict(),
        )

    def test_from_dict(self):
        self.assertEqual(
            self.v,
            self.V.from_dict({
                'index2word': ['a', 'bb', 'ccc'],
                'counts': {'a': 1, 'bb': 1, 'ccc': 1},
            })
        )

    def test_prune_by_count(self):
        v = self.V(['a'] * 1 + ['b'] * 2 + ['c'] * 3 + ['d'] * 4)
        self.assertTrue(self.V(['c'] * 3 + ['d'] * 4).contains_same_content(v.prune_by_count(3)))

    def test_prune_by_total(self):
        v = self.V(['a'] * 1 + ['b'] * 2 + ['c'] * 3 + ['d'] * 4)
        self.assertTrue(self.V(['b'] * 2 + ['c'] * 3 + ['d'] * 4).contains_same_content(v.prune_by_total(3)))

    def test_word2index(self):
        self.assertEqual(1, self.v.word2index('bb'))
        self.assertListEqual(
            [0, 2, 1],
            self.v.word2index(['a', 'ccc', 'bb'])
        )
        with self.assertRaises(OutOfVocabularyException):
            self.v.word2index('dddd')
            self.v.word2index(['a', 'dddd'])

    def test_index2word(self):
        self.assertEqual('bb', self.v.index2word(1))
        self.assertListEqual(
            ['a', 'ccc', 'bb'],
            self.v.index2word([0, 2, 1])
        )
        with self.assertRaises(OutOfVocabularyException):
            self.v.index2word(-1)
            self.v.index2word(3)
            self.v.index2word([0, 3])


if __name__ == '__main__':
    unittest.main()
