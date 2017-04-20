from vocab.vocab import Vocab


class UnkVocab(Vocab):

    _reserved = {'<unk>'}

    def _handle_oov_word(self, word):
        return self.word2index('<unk>')
