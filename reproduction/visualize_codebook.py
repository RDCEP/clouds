"""visualize_codebook.py: Finds the best patches to represent a sparse dictionary.

"Best" means: cosine similarity, highest coefficient (when sparsely coded), or largest
projection along the codeword direction.
"""
# from pipeline.load import load_data
from tensorflow.python_io import tf_record_iterator
from sporco.admm import bpdn
import numpy as np
import heapq
import glob
import os


def gen_encodings(encoder, glob):
    """Yields encoded patch, the file it came from, and its pixel coordinate.
    """
    for f in glob.glob(FLAGS.data_glob):
        for p in tf_record_iterator(f):
            fname = p["filename"]
            coord = p["coord"]
            patch = np.fromstring(p["patch"], dtype=np.float32).reshape(p["shape"])
            patch = encoder.predict(np.expand_dims(patch)).squeeze()
            yield patch, fname, coord


class TopK:
    """For each codeword track the top and bottom `k` patches according to `ranker`.
    `ranker` should return a list of the sample's "representation score" for each code
    word.
    """

    def __init__(self, ranker, k):
        """
        Args:
            ranker: Fn(example, sdict, code) -> [ranks] which will rank the example with
                respect to every codeword.
            k: The number of top / bottom examples to keep for each codeword
        """
        self.ranker = ranker
        self.top = []

    def update(self, info, *args):
        scores = self.ranker(*args)
        raise NotImplementedError()
        # TODO Update representative of every codeword

    def save_examples(path):
        pass


# TODO
def cosine(example, sdict, code):
    pass


def coef(example, sdict, code):
    pass


def proj(example, sdict, code):
    pass


def get_flags():
    p = ArgumentParser(description=__doc__)
    p.add_argument("encoder")
    p.add_argument("encoder_step", type=int)
    p.add_argument("sparse_dict")
    p.add_argument("out_dir")
    p.add_argument("data_glob", nargs="+")
    p.add_argument("--cosine_similarity", action="store_true")
    p.add_argument("--coefficient", action="store_true")
    p.add_argument("--projection", action="store_true")
    p.add_argument(
        "--num_top", type=int, default=10, help="Number of examples to keep."
    )

    args = p.parse_args()

    if not any([args.cosine_similarity, args.coefficient, args.projection]):
        raise ValueError("Need to select at least one top thing to track.")

    return args


if __name__ == "__main__":

    # TODO
    FLAGS = get_flags()
    encoder = None
    sparse_dict = None

    # Initialize top k tracking objects
    top_ks = {}
    if FLAGS.cosine_similarity:
        top_ks["cosine_similar"] = TopK(cosine, FLAGS.num_top)
    if FLAGS.coefficient:
        top_ks["coefficient"] = TopK(coef, FLAGS.num_top)
    if FLAGS.projections:
        top_ks["projections"] = TopK(proj, FLAGS.num_top)

    # Iterate through every encoding, tracking top k representatives of every codeword
    for enc, fname, coord in gen_encodings(encoder, FLAGS.data_glob):
        for t in top_ks:
            sdict = code = None  # TODO
            top_ks[t].update((fname, coord), enc, sdict, code)

    for t in top_ks:
        top_ks.save_examples(os.path.join(FLAGS.out_dir, t))
