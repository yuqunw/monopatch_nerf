import sys
sys.path.append('.')
sys.path.append('..')

from pathlib import Path
from scripts.fusion_no_normals import fuse_reconstruction


def main(args):
    # first perform fusion
    output_path = Path(args.output_path)
    (output_path / 'results').mkdir(exist_ok=True, parents=True)
    pc_file = output_path / 'results' / 'fused.ply'
    sparse_path = Path(args.sparse_path)

    fuse_reconstruction(str(output_path), pc_file, args.threshold, args.min_views, args.device, sparse_path=sparse_path)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--sparse_path', type=str)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--min_views', default=5, type=int)
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()
    main(args)
