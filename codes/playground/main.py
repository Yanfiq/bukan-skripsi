def main():
    import torch
    x = torch.zeros(1, 2, 3, 4)
    print(x.shape)              # Output: torch.Size([2, 1, 3, 1])
    print(x)
    print(x.squeeze().shape)      # Output: torch.Size([2, 3])
    print(x.squeeze(dim=1).shape) # Output: torch.Size([2, 3, 1])
    print(x.squeeze())


if __name__ == "__main__":
    main()
