from collections import Counter
from copy import deepcopy


class OutOfVocabularyException(Exception):
    pass


class Vocab:

    _reserved = set()  # words in here will not be discarded during pruning

    def __init__(self, words=()):
        """
        Arguments:
            words (:obj:`list` of :obj:`str`, optional): words to build vocab from.
        """
        self._index2word = []
        self._word2index = {}
        self.counts = Counter()
        for w in self._reserved:
            self.word2index(w, train=True)
            self.counts[w] = 0
        if words:
            self.word2index(words, train=True)

    def __len__(self):
        return len(self._index2word)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))

    def __eq__(self, another):
        for i, w in enumerate(self._index2word):
            if w not in self._reserved:
                if w != another._index2word[i] or self.counts[w] != another.counts[w]:
                    return False
        return True

    def __ne__(self, another):
        return not self.__eq__(another)

    def contains_same_content(self, another, same_counts=True):
        """
        Arguments:
            another (Vocab): another vocab to compare against.
            same_counts (:obj:`bool`, optional): whether to also check the counts.

        Returns:
            bool: whether this vocab and `another` contains the same content.
        """
        words = set(list(self._word2index.keys())).union(set(list(another._word2index.keys())))
        for w in words:
            if (w in another._word2index) != (w in self._word2index):
                return False
            if same_counts and self.counts[w] != another.counts[w]:
                return False
        return True

    def to_dict(self):
        """
        Returns:
            dict: dictionary of the voca object.
        """
        return {
            'index2word': [w for w in self._index2word if w not in self._reserved],
            'counts': {k: v for k, v in self.counts.items() if k not in self._reserved},
        }

    @classmethod
    def from_dict(cls, d):
        """
        Arguments:
            d (dict): dictionary of the vocab object.

        Returns:
            Vocab: vocab object from the given dictionary.
        """
        v = cls()
        for i, w in enumerate(d['index2word']):
            v.word2index(w, train=True)
            v.counts[w] = d['counts'][w]
        return v

    def copy(self, keep_words=True):
        """
        Arguments:
            keep_words (bool): whether to copy words in the vocab. Defaults to `True`.

        Returns:
            Vocab: a copy of this vocab.
        """
        return deepcopy(self) if keep_words else self.__class__()

    def prune_by_count(self, cutoff):
        """
        Arguments:
            cutoff (int): words occurring less than this number of times are removed from the new vocab.

        Returns:
            Vocab: a copy of this vocab object with words occurring less than `cutoff` times removed.
        """
        another = self.copy(keep_words=False)
        for w, c in self.counts.items():
            if c >= cutoff:
                another.word2index(w, train=True)
                another.counts[w] = c
        return another

    def prune_by_total(self, total):
        """
        Arguments:
            total (int): maximum vocab size
        Returns:
            Vocab: a copy of this vocab with only the top `total` words kept.
        """
        another = self.copy(keep_words=False)
        keep = [k for k, c in self.counts.most_common(total) if k not in self._reserved]
        for w in keep[:total]:
            another.word2index(w, train=True)
            another.counts[w] = self.counts[w]
        return another

    def word2index(self, word, train=False):
        """
        Args:
            word (str): word to look up index for.
            train (:obj:`bool`, optional): if `True`, then this word will be added to the voculary. Defaults to `False`.

        Returns:
            int: index corresponding to `word`.

            if `word` is a :obj:`list` of :obj:`str` then this function will be applied for each word and the corresponding list of indices is returned.

        Raises:
            OutOfVocabularyException: if `train` is `False` and `word` is not in the vocabulary
        """
        if isinstance(word, (list, tuple)):
            return [self.word2index(w, train=train) for w in word]
        self.counts[word] += train
        if word in self._word2index:
            return self._word2index[word]
        else:
            if train:
                self._index2word += [word]
                self._word2index[word] = len(self._word2index)
            else:
                return self._handle_oov_word(word)
        return self._word2index[word]

    def _handle_oov_word(self, word):
        """
        What to do when the word is out of vocabulary and not in training mode.
        You should not use this function explicitly.

        Arguments:
            word (str): word that trigged the OOV exception.
        """
        raise OutOfVocabularyException("Word '{}' is not in the vocabulary".format(word))

    def index2word(self, index):
        """
        Args:
            index (int): index to look up word for.

        Returns:
            str: word corresponding to `index`.

            if `index` is a :obj:`list` of :obj:`int` then this function will be applied for each index and the corresponding list of words is returned.

        Raises:
            OutOfVocabularyException: if `index` is not a valid index to the vocabulary.
        """
        if isinstance(index, list):
            return [self.index2word(i) for i in index]
        if index < 0:
            raise OutOfVocabularyException('Index {} is negative and is not a valid word index'.format(index))
        if index >= len(self):
            raise OutOfVocabularyException('Index {} exceeds vocab size {} and is not a valid word index'.format(index, len(self)))
        return self._index2word[index]
