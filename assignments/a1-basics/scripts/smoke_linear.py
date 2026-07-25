import torch

from cs336_basics.model import Linear


def main() -> None:
    d_in = 3
    d_out = 2

    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    weight = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
        ]
    )

    linear = Linear(
        in_features=d_in,
        out_features=d_out,
    )

    with torch.no_grad():
        linear.weight.copy_(weight)

    actual = linear(x)
    expected = x @ weight.T

    print("x.shape:", x.shape)
    print("weight.shape:", weight.shape)
    print("actual.shape:", actual.shape)
    print("actual:")
    print(actual)
    print("expected:")
    print(expected)

    torch.testing.assert_close(actual, expected)
    print("Linear smoke test passed.")


if __name__ == "__main__":
    main()