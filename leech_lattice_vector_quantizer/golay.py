class GolayCode:
    """Extended binary Golay code (24, 12, 8), represented as Torch tensors."""

    def __init__(self, device: str | torch.device | None = None):
        self.device = _default_device(device)
        self.G = self._generator_matrix(self.device)
        self.codewords = self._generate_codewords()

    @staticmethod
    def _generator_matrix(device: torch.device) -> torch.Tensor:
        eye = torch.eye(12, dtype=torch.float32, device=device)
        parity = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0],
                [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        return torch.cat([eye, parity], dim=1)

    def encode(self, msg: torch.Tensor) -> torch.Tensor:
        msg = msg.to(device=self.device, dtype=torch.float32)
        return ((msg @ self.G).remainder(2)).to(torch.int16)

    def _generate_codewords(self) -> torch.Tensor:
        ids = torch.arange(1 << 12, device=self.device, dtype=torch.int64)
        shifts = torch.arange(12, device=self.device, dtype=torch.int64)
        messages = ((ids[:, None] >> shifts[None, :]) & 1).to(torch.float32)
        return self.encode(messages)