from nltk.corpus import wordnet as wn

mapping_ids_synset = lambda x: wn.synset_from_pos_and_offset('n', int(x[1:]))

tree_distance = lambda x, y, z: getattr(x, z)(y)

mapping_vocidx_to_synsets = lambda anchor, vocab: [mapping_ids_synset(vocab.mapping_global_idx_ids[t]) for t in vocab.mapping_idx_global_idx[anchor]]