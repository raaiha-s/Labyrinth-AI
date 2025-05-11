import pygame
import random
import heapq
import math
import numpy as np
from copy import deepcopy

pygame.init()

WIDTH, HEIGHT = 800, 600
ROWS, COLS = 7, 7
TILE_SIZE = WIDTH // COLS

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (70, 130, 180)
GREEN = (34, 139, 34)
RED = (220, 20, 60)
ORANGE = (255, 140, 0)
GRAY = (169, 169, 169)
YELLOW = (255, 255, 0)

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Labyrinth AI - Human vs AI")

# Game state
grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
last_shifted = None 

# Players
human_pos = [0, 0]
ai_pos = [6, 6]
goals = []
traps = []
shift_phase = True  

turn_counter = 0
goals_collected = {'human': 0, 'ai': 0}
current_turn = "HUMAN"
trap_cooldown = {'human': 0, 'ai': 0}

font = pygame.font.Font(None, 36)

def generate_goals():
    global goals
    goals = []
    for _ in range(3):
        while True:
            pos = [random.randint(0, ROWS-1), random.randint(0, COLS-1)]
            if pos not in [human_pos, ai_pos] and pos not in goals:
                goals.append(pos)
                break

def draw_grid():
    win.fill(GRAY)
    for i in range(ROWS):
        for j in range(COLS):
            x, y = j * TILE_SIZE, i * TILE_SIZE
            rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)

            # Draw traps
            if [i, j] in traps:
                pygame.draw.circle(win, RED, (x + TILE_SIZE//2, y + TILE_SIZE//2), TILE_SIZE//4)
            
            # Draw goals
            if [i, j] in goals:
                pygame.draw.rect(win, ORANGE, rect, 5)
            
            # Draw players
            if [i, j] == human_pos:
                pygame.draw.circle(win, BLUE, (x + TILE_SIZE//2, y + TILE_SIZE//2), TILE_SIZE//3)
            elif [i, j] == ai_pos:
                pygame.draw.circle(win, GREEN, (x + TILE_SIZE//2, y + TILE_SIZE//2), TILE_SIZE//3)
            
            # Draw grid lines
            pygame.draw.rect(win, BLACK, rect, 1)
    
    # Draw UI elements
    goals_text = font.render(f"Human: {goals_collected['human']} | AI: {goals_collected['ai']}", True, BLACK)
    win.blit(goals_text, (10, HEIGHT - 50))
    
    turn_text = font.render(f"Current Turn: {current_turn}", True, BLACK)
    win.blit(turn_text, (10, 10))
    
    if shift_phase:
        shift_text = font.render("SHIFT PHASE: Click row/column to shift", True, YELLOW)
        win.blit(shift_text, (WIDTH//2 - 200, HEIGHT - 50))

def shift_row(row_idx, direction):
    global last_shifted
    row = grid[row_idx]
    if direction == 'left':
        row = [row[-1]] + row[:-1]
    else:
        row = row[1:] + [row[0]]
    grid[row_idx] = row
    last_shifted = ('row', row_idx)

def shift_col(col_idx, direction):
    global last_shifted
    col = [grid[row][col_idx] for row in range(ROWS)]
    if direction == 'up':
        col = [col[-1]] + col[:-1]
    else:
        col = col[1:] + [col[0]]
    for row in range(ROWS):
        grid[row][col_idx] = col[row]
    last_shifted = ('col', col_idx)

def is_shift_valid(shift_type, index):
    if last_shifted is None:
        return True
    last_type, last_idx = last_shifted
    return not (shift_type == last_type and index == last_idx)

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star(start, goal, avoid_traps=True):
    heap = []
    heapq.heappush(heap, (0 + heuristic(start, goal), 0, start, []))
    visited = set()

    while heap:
        est_total, cost, current, path = heapq.heappop(heap)
        if tuple(current) in visited:
            continue
        visited.add(tuple(current))
        new_path = path + [current]
        if current == goal:
            return new_path[1:]
        for nb in get_neighbors(current, avoid_traps):
            heapq.heappush(heap, (cost + 1 + heuristic(nb, goal), cost + 1, nb, new_path))
    return []

def get_neighbors(pos, avoid_traps=True):
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    neighbors = []
    for d in dirs:
        ni, nj = pos[0] + d[0], pos[1] + d[1]
        if 0 <= ni < ROWS and 0 <= nj < COLS:
            if avoid_traps and [ni, nj] in traps:
                continue
            neighbors.append([ni, nj])
    return neighbors

def update_traps():
    global traps
    if not traps:  # Initialize traps if empty
        traps = []
        for _ in range(3):
            while True:
                pos = [random.randint(0, ROWS-1), random.randint(0, COLS-1)]
                if pos != human_pos and pos != ai_pos and pos not in goals:
                    traps.append(pos)
                    break
    else:  # Move existing traps
        new_traps = []
        for trap in traps:
            players = [human_pos, ai_pos]
            closest = min(players, key=lambda p: heuristic(trap, p))
            direction = [np.sign(closest[0] - trap[0]), np.sign(closest[1] - trap[1])]
            new_pos = [trap[0] + direction[0], trap[1] + direction[1]]
            if 0 <= new_pos[0] < ROWS and 0 <= new_pos[1] < COLS:
                new_traps.append(new_pos)
            else:
                new_traps.append(trap)
        traps = new_traps

def generate_shift_moves(state):
    moves = []
    for row in range(ROWS):
        for direction in ['left', 'right']:
            if is_shift_valid_in_state(state, 'row', row):
                new_state = deepcopy(state)
                new_grid, new_last_shifted = shift_row_in_state(new_state['grid'], row, direction)
                new_state['grid'] = new_grid
                new_state['last_shifted'] = new_last_shifted
                new_state['shift'] = ('row', row, direction)
                moves.append(new_state)
    
    for col in range(COLS):
        for direction in ['up', 'down']:
            if is_shift_valid_in_state(state, 'col', col):
                new_state = deepcopy(state)
                new_grid, new_last_shifted = shift_col_in_state(new_state['grid'], col, direction)
                new_state['grid'] = new_grid
                new_state['last_shifted'] = new_last_shifted
                new_state['shift'] = ('col', col, direction)
                moves.append(new_state)
    return moves

def is_shift_valid_in_state(state, shift_type, index):
    if state['last_shifted'] is None:
        return True
    last_type, last_idx = state['last_shifted']
    return not (shift_type == last_type and index == last_idx)

def shift_row_in_state(grid, row_idx, direction):
    new_grid = [row.copy() for row in grid]
    row = new_grid[row_idx]
    if direction == 'left':
        new_row = [row[-1]] + row[:-1]
    else:
        new_row = row[1:] + [row[0]]
    new_grid[row_idx] = new_row
    return new_grid, ('row', row_idx)

def shift_col_in_state(grid, col_idx, direction):
    new_grid = [row.copy() for row in grid]
    col = [new_grid[row][col_idx] for row in range(ROWS)]
    if direction == 'up':
        new_col = [col[-1]] + col[:-1]
    else:
        new_col = col[1:] + [col[0]]
    for row in range(ROWS):
        new_grid[row][col_idx] = new_col[row]
    return new_grid, ('col', col_idx)

def minimax(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or state['winner'] is not None:
        return evaluate_state(state)
    
    if maximizing_player:
        max_eval = -math.inf
        for child in generate_shift_moves(state):
            eval = minimax(child, depth-1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for child in generate_shift_moves(state):
            eval = minimax(child, depth-1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def evaluate_state(state):
    ai_score = 0
    human_score = 0
    
    ai_goal = min(state['goals'], key=lambda g: heuristic(state['ai_pos'], g)) if state['goals'] else None
    human_goal = min(state['goals'], key=lambda g: heuristic(state['human_pos'], g)) if state['goals'] else None
    
    if ai_goal:
        ai_path = a_star(state['ai_pos'], ai_goal, avoid_traps=False)
        ai_score += (10 - len(ai_path)) * 100 if ai_path else -1000
        
    if human_goal:
        human_path = a_star(state['human_pos'], human_goal, avoid_traps=False)
        human_score += (10 - len(human_path)) * 100 if human_path else -1000
    
    for trap in state['traps']:
        ai_score -= 50 if heuristic(trap, state['ai_pos']) < 2 else 0
        human_score -= 50 if heuristic(trap, state['human_pos']) < 2 else 0
    
    return ai_score - human_score

def apply_shift(shift):
    shift_type, index, direction = shift
    if shift_type == 'row':
        shift_row(index, direction)
    else:
        shift_col(index, direction)

def check_goal_collection(pos, player):
    global goals_collected, goals
    for goal in goals[:]:
        if pos == goal:
            goals_collected[player] += 1
            goals.remove(goal)
            generate_goals()
            update_traps()
            break

def check_win_condition():
    if goals_collected['human'] >= 3:
        return "Human Wins!"
    if goals_collected['ai'] >= 3:
        return "AI Wins!"
    return None

def ai_turn():
    global shift_phase, ai_pos, grid, last_shifted
    
    best_score = -math.inf
    best_shift = None
    original_state = deepcopy({
        'ai_pos': ai_pos,
        'human_pos': human_pos,
        'goals': goals.copy(),
        'traps': traps.copy(),
        'grid': [row.copy() for row in grid],
        'last_shifted': last_shifted,
        'winner': None
    })
    
    for shift_state in generate_shift_moves(original_state):
        score = minimax(shift_state, 2, -math.inf, math.inf, False)
        if score > best_score:
            best_score = score
            best_shift = shift_state
    
    if best_shift:
        apply_shift(best_shift['shift'])
        grid = best_shift['grid']
        last_shifted = best_shift['last_shifted']
    
    # Move phase using A*
    if goals:
        closest_goal = min(goals, key=lambda g: heuristic(ai_pos, g))
        path = a_star(ai_pos, closest_goal)
        if path:
            ai_pos = path[0]
    
    check_goal_collection(ai_pos, 'ai')

generate_goals()
update_traps()

running = True
while running:
    clock = pygame.time.Clock()
    clock.tick(10)
    
    winner = check_win_condition()
    if winner:
        draw_grid()
        win_text = font.render(winner, True, YELLOW)
        win.blit(win_text, (WIDTH//2 - 100, HEIGHT//2 - 50))
        pygame.display.update()
        pygame.time.delay(3000)
        running = False
        continue
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if current_turn == "HUMAN" and shift_phase:
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                row = y // TILE_SIZE
                col = x // TILE_SIZE
                
                # Determine shift type
                if abs(x - col*TILE_SIZE) < TILE_SIZE/4:
                    if is_shift_valid('col', col):
                        shift_col(col, 'up' if y < HEIGHT/2 else 'down')
                        shift_phase = False
                elif abs(y - row*TILE_SIZE) < TILE_SIZE/4:
                    if is_shift_valid('row', row):
                        shift_row(row, 'left' if x < WIDTH/2 else 'right')
                        shift_phase = False
        
        elif current_turn == "HUMAN" and not shift_phase:
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                col = x // TILE_SIZE
                row = y // TILE_SIZE
                if [row, col] in get_neighbors(human_pos):
                    human_pos = [row, col]
                    check_goal_collection(human_pos, 'human')
                    current_turn = "AI"
                    shift_phase = True
                    turn_counter += 1
                    
                    if turn_counter % 3 == 0:
                        update_traps()
    
    if current_turn == "AI":
        ai_turn()
        current_turn = "HUMAN"
        shift_phase = True
        turn_counter += 1
        
        if turn_counter % 3 == 0:
            update_traps()
    
    draw_grid()
    pygame.display.update()

pygame.quit()