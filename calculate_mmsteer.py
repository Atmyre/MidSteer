import argparse
import pickle
from compute_steering_vectors import calculate_mmster

from core.pickle import unpickle

def run(
    pos_means: dict,
    neg_means: dict,
    pos_covs: dict,
    neg_covs: dict,
    forward_output_path: str,
    inverse_output_path: str,
):
    
    mmsteer_transforms_forward = calculate_mmster(
        pos_means=pos_means,
        pos_covariances=pos_covs,
        neg_means=neg_means,
        neg_covariances=neg_covs,
    )
    with open(forward_output_path, 'wb') as fout:
        pickle.dump(mmsteer_transforms_forward, fout)

    mmsteer_transforms_inverse = calculate_mmster(
        pos_means=neg_means,
        pos_covariances=neg_covs,
        neg_means=pos_means,
        neg_covariances=pos_covs,
    )
    with open(inverse_output_path, 'wb') as fout:
        pickle.dump(mmsteer_transforms_inverse, fout)

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_means', type=str, required=True, help='Path to pos_means vectors')
    parser.add_argument('--neg_means', type=str, required=True, help='Path to pos_means vectors')
    parser.add_argument('--pos_covs', type=str, required=True, help='Path to pos_means vectors')
    parser.add_argument('--neg_covs', type=str, required=True, help='Path to pos_means vectors')
    parser.add_argument('--forward_output_path', type=str, required=True, help='Output path for forward matrix')
    parser.add_argument('--inverse_output_path', type=str, required=True, help='Output path for inverse matrix')
    args = parser.parse_args()
   
    
    run(
        pos_means=unpickle(args.pos_means),
        neg_means=unpickle(args.neg_means),
        pos_covs=unpickle(args.pos_covs),
        neg_covs=unpickle(args.neg_covs),
        forward_output_path=args.forward_output_path,
        inverse_output_path=args.inverse_output_path,
    )
    


if __name__ == "__main__":
    main()