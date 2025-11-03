import chess
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import tqdm

def fen_to_tensor(fen: str) -> torch.Tensor:
    board = chess.Board(fen)
    tensor = torch.zeros(12, 8, 8, dtype=torch.float32)
    piece_to_index = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_offset = 0 if piece.color == chess.WHITE else 6
            piece_index = piece_to_index[piece.piece_type] + color_offset
            row = chess.square_rank(square)
            col = chess.square_file(square)
            tensor[piece_index, row, col] = 1.0

    return tensor


class NNUEEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

# Dataset in data/chessData.csv
# Each line: FEN string, evaluation score
# Careful: large file and mate scores start with #!
# scores in centipawns

class ChessDataset(Dataset):
    def __init__(self, csv_file: str, n_positions: int = None):
        self.data = []
        
        print("Loading and converting dataset...")
        with open(csv_file, "r") as f:
            for i, line in tqdm.tqdm(enumerate(f)):
                if n_positions is not None and len(self) >= n_positions:
                    break
                
                parts = line.strip().split(",")
                if len(parts) == 2:
                    fen, score = parts[0], parts[1]
                    if score.startswith("#"):
                        continue
                    try:
                        score_val = float(score) / 100.0
                        # Convert to tensor immediately during loading
                        tensor = fen_to_tensor(fen)
                        self.data.append((tensor, score_val))
                    except ValueError:
                        continue
        
        print(f"Loaded {len(self.data)} positions")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor, score = self.data[idx]
        return tensor, torch.tensor(score, dtype=torch.float32)


def create_dataloaders(n_positions: int = None) -> tuple[DataLoader, DataLoader]:
    # Load dataset
    dataset = ChessDataset("data/chessData.csv", n_positions)

    # Split into train/test (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


def train_loop(num_epochs: int, n_positions: int = None):
    # Training loop
    evaluator = NNUEEvaluator()
    optimizer = torch.optim.Adam(evaluator.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_loader, test_loader = create_dataloaders(n_positions)

    for epoch in range(num_epochs):
        evaluator.train()
        running_loss = 0.0
        for inputs, targets in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = evaluator(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}")

        # Evaluation loop
        evaluator.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = evaluator(inputs)
                loss = criterion(outputs.squeeze(), targets)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}")
        
        # Save the trained model (every epoch in case it crashes)
        torch.save(evaluator.state_dict(), "nnue_evaluator.pth")
        

train_loop(num_epochs=5, n_positions=10000)