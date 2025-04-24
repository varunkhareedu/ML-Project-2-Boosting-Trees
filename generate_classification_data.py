import argparse
import csv
import numpy as np

def classification_data_generator(weights, bias, rnge, N, seed):
    rng = np.random.default_rng(seed=seed)
    X = rng.uniform(low=rnge[0], high=rnge[1], size=(N, len(weights)))
    logits = np.dot(X, np.reshape(weights, (-1, 1))) + bias
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs).astype(float)  # Convert to binary class labels
    return X, y

def write_data(filename, X, y):
    with open(filename, "w", newline="") as file:
        header = [f"x_{i}" for i in range(X.shape[1])] + ["y"]
        writer = csv.writer(file)
        writer.writerow(header)
        for row in np.hstack((X, y)):
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, help="Number of samples.")
    parser.add_argument("-m", nargs='*', type=float, help="Weight coefficients.")
    parser.add_argument("-b", type=float, help="Bias.")
    parser.add_argument("-rnge", nargs=2, type=float, help="Range of feature values.")
    parser.add_argument("-seed", type=int, help="Random seed.")
    parser.add_argument("-output_file", type=str, help="Output CSV file.")
    args = parser.parse_args()

    weights = np.array(args.m)
    X, y = classification_data_generator(weights, args.b, args.rnge, args.N, args.seed)
    write_data(args.output_file, X, y)

if __name__ == "__main__":
    main()