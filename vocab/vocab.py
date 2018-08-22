import sys
from collections import Counter
from copy import deepcopy


class OutOfVocabularyException(Exception):
    pass


class Vocab:
    """
    A vocabulary object for converting between words and numerical indices.

    Attributes:
        _index2word (list): an ordered list of words in the vocabulary.
        _word2index (dict): maps words to their respective indices.
        counts (dict): the number of times each word has been added to the vocabulary.
        
    """

    _reserved = set()  # words in here will not be discarded during pruning

    def __init__(self, words=()):
        """
        Args:
            words (:obj:`list` of :obj:`str`, optional): words to build vocab from.

        Example:
            >>> Vocab(['initial', 'words', 'for', 'the', 'vocabulary'])
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
        """
        Returns:
            int: number of words in the vocabulary.
        """
        return len(self._index2word)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))

    def __eq__(self, another):
        if self.__class__ != another.__class__:
            return False
        if len(self) != len(another):
            return False
        for i, w in enumerate(self._index2word):
            if w not in self._reserved:
                if w != another._index2word[i] or self.counts[w] != another.counts[w]:
                    return False
        return True

    def __ne__(self, another):
        return not self.__eq__(another)

    def contains_same_content(self, another, same_counts=True):
        """
        Args:
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
        Args:
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
        Args:
            keep_words (bool): whether to copy words in the vocab. Defaults to `True`.

        Returns:
            Vocab: a copy of this vocab.
        """
        return deepcopy(self) if keep_words else self.__class__()

    def prune_by_count(self, cutoff):
        """
        Args:
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
        Args:
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

        Args:
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
        return sys.intern(self._index2word[index])

    def word2padded_index(self, lists_of_words, pad='<pad>', train=False, enforce_end_pad=True):
        """
        Args:
            lists_of_words (list): list of lists of words to pad
            pad (:obj:`str`, optional): word to use for padding. Defaults to `'<pad>'`.
            train (:obj:`bool`, optional): whether to add unknown words to the vocabulary. Defaults to `False`.
            enforce_end_pad (:obj:`bool`, optional): whether to always append a pad word to the end of each sentence.

        Returns:
            list: list of lists of word indices that are padded to be a matrix
            list: list of lengths for each valid sequence. Note that if `enforce_end_pad=True`, then the valid sequence includes the additional pad at the end.

        Raises:
            OutOfVocabularyException: if `lists_of_words` contains words not in the vocabulary and `train=False`.
        """
        if pad not in self._word2index and not train:
            raise OutOfVocabularyException("Pad word '{}' is not in the vocabulary".format(pad))
        seqs = [s + [pad] for s in lists_of_words] if enforce_end_pad else lists_of_words
        lens = [len(s) for s in seqs]
        max_len = max(lens)
        indices = [self.word2index(s, train=train) for s in seqs]
        pad_index = self.word2index(pad, train=train)
        padded_indices = [s + [pad_index] * (max_len - l) for s, l in zip(indices, lens)]
        return padded_indices, lens

    def padded_index2word(self, padded_indices, pad='<pad>'):
        """
        Args:
            padded_indices (list): list of lists of word indices to depad
            pad (:obj:`str`, optional): word to use for padding. Defaults to `'<pad>'`.

        Returns:
            list: list of lists of words that correspond to the depadded `padded_indices`.
            list: list of lengths for each valid sequence. Note that if `enforce_end_pad=True`, then the valid sequence includes the additional pad at the end.

        Raises:
            OutOfVocabularyException: if `padded_indices` contains indices not in the vocabulary or if `pad` is a word not in the vocabulary.
        """
        if pad not in self._word2index:
            raise OutOfVocabularyException("Pad word '{}' is not in the vocabulary".format(pad))
        pad_index = self.word2index(pad)
        depadded = []
        for indices in padded_indices:
            try:
                end = indices.index(pad_index)
            except ValueError as e:
                end = len(indices)
            finally:
                depadded.append(self.index2word(indices[:end]))
        return depadded
