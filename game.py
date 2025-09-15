import pygame
from dataclasses import dataclass
import chess


board = chess.Board()
piece_to_image = {
    "P": pygame.image.load("images/pawnW.png"),
    "N": pygame.image.load("images/knightW.png"),
    "B": pygame.image.load("images/bishopW.png"),
    "R": pygame.image.load("images/rookW.png"),
    "Q": pygame.image.load("images/queenW.png"),
    "K": pygame.image.load("images/kingW.png"),
    "p": pygame.image.load("images/pawnB.png"),
    "n": pygame.image.load("images/knightB.png"),
    "b": pygame.image.load("images/bishopB.png"),
    "r": pygame.image.load("images/rookB.png"),
    "q": pygame.image.load("images/queenB.png"),
    "k": pygame.image.load("images/kingB.png"),
}


# Modify the image
for key, image in piece_to_image.items():
    piece_to_image[key] = pygame.transform.scale(image, (100, 100))


class SelectedPiece:
    row: int = -1
    col: int = -1
    is_selected: bool = False


@dataclass
class Tile:
    row: int
    col: int
    piece: chess.Piece = None
    user_highlight: bool = False
    selected: bool = False
    legal_move: bool = False

    def draw(self):
        rect = pygame.Rect(self.col * 100, self.row * 100, 100, 100)
        light_square = (self.row + self.col) % 2 == 0
        if self.user_highlight:
            color = COLOR_LIGHT_HIGHLIGHT if light_square else COLOR_DARK_HIGHLIGHT
            font_color = COLOR_DARK_HIGHLIGHT if light_square else COLOR_LIGHT_HIGHLIGHT
        else:
            color = COLOR_LIGHT if light_square else COLOR_DARK
            font_color = COLOR_DARK if light_square else COLOR_LIGHT

        pygame.draw.rect(screen, color, rect)

        # Transparent overlay for selected or legal_move
        if self.selected or self.legal_move:
            overlay = pygame.Surface((100, 100), pygame.SRCALPHA)
            if self.selected:
                overlay_color = COLOR_SELECTED
            else:
                overlay_color = COLOR_LEGAL_MOVE
            pygame.draw.rect(overlay, overlay_color, (0, 0, 100, 100), 8, 15)
            screen.blit(overlay, (self.col * 100, self.row * 100))

        # Add row and column labels
        if self.col == 0:
            label = FONT.render(str(8 - self.row), True, font_color)
            screen.blit(label, (5, self.row * 100 + 5))
        if self.row == 7:
            label = FONT.render(chr(97 + self.col), True, color)
            screen.blit(label, (self.col * 100 + 85, 5))

    def draw_piece(self):
        self.piece = board.piece_at((7 - self.row) * 8 + self.col)
        if self.piece:
            # Draw the piece (replace with your piece drawing logic)
            image = piece_to_image[self.piece.symbol()]
            screen.blit(image, (self.col * 100, self.row * 100))


tiles_flat: list[Tile] = [Tile(row=i // 8, col=i % 8) for i in range(64)]
tiles: list[list[Tile]] = [[tiles_flat[i * 8 + j] for j in range(8)] for i in range(8)]

# Initialize Pygame
pygame.init()

# Set up the game window
screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("GUI SigmaZero V2 Chess Engine")

COLOR_LIGHT = (238, 238, 210)
COLOR_DARK = (118, 150, 86)
COLOR_LIGHT_HIGHLIGHT = (222, 132, 111)
COLOR_DARK_HIGHLIGHT = (199, 115, 87)
COLOR_LEGAL_MOVE = (175, 175, 175, 150)
COLOR_SELECTED = (255, 255, 0, 100)
FONT = pygame.font.Font(None, 24)

for tile in tiles_flat:
    tile.draw()

# Game loop
running = True
FPS = 10
clock = pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONUP:
            x, y = event.pos
            row, col = y // 100, x // 100
            if event.button == 3:  # Right click
                tile: Tile = tiles[row][col]
                tile.user_highlight = not tile.user_highlight

            elif event.button == 1:  # Left click
                tile: Tile = tiles[row][col]

                if tile.piece and tile.piece.color == board.turn:  # Select piece
                    for t in tiles_flat:
                        t.selected = False
                        t.legal_move = False
                        t.user_highlight = False

                    tile.selected = True
                    SelectedPiece.is_selected = True
                    SelectedPiece.row = row
                    SelectedPiece.col = col
                    legal_moves = list(board.legal_moves)
                    for move in legal_moves:
                        if move.from_square == (7 - row) * 8 + col:
                            to_row = 7 - (move.to_square // 8)
                            to_col = move.to_square % 8
                            tiles[to_row][to_col].legal_move = True

                elif SelectedPiece.is_selected:
                    if not tile.legal_move:
                        continue

                    for tile in tiles_flat:
                        tile.selected = False
                        tile.legal_move = False
                        tile.user_highlight = False

    # --- DRAWING SECTION ---
    screen.fill((0, 0, 0))  # Clear screen

    # Draw tiles
    for tile in tiles_flat:
        tile.draw()

    # Draw pieces
    for tile in tiles_flat:
        tile.draw_piece()

    pygame.display.flip()
    clock.tick(FPS)

# Quit Pygame
pygame.quit()
