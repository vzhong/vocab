Vocab
=====

.. image:: https://readthedocs.org/projects/vocab/badge/?version=latest
    :target: http://vocab.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://travis-ci.org/vzhong/vocab.svg?branch=master
    :target: https://travis-ci.org/vzhong/vocab

Vocab is a python package that provides vocabulary objects for natural language processing.


Installation
------------


.. code-block:: sh

    pip install vocab
    pip install git+https://github.com/vzhong/vocab.git


Usage
-----

.. code-block:: python

    >>> from vocab import Vocab, UnkVocab
    >>> v = Vocab()
    >>> v.word2index('hello', train=True)
    0
    >>> v.word2index(['hello', 'world'], train=True)
    [0, 1]
    >>> v.index2word([1, 0])
    ['world', 'hello']
    >>> v.index2word(1)
    'world'
    >>> small = v.prune_by_count(2)
    >>> small.to_dict()
    {'counts': {'hello': 2}, 'index2word': ['hello']}
    >>> u = UnkVocab()
    >>> u.word2index(['hello', 'world'], train=True)
    [1, 2]
    >>> u.word2index('hello friend !'.split())
    [1, 0, 0]
    >>> u.index2word(0)
    '<unk>'
