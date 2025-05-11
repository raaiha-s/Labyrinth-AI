# Labyrinth AI: Human vs AI – Project Report

## Group Members
- **Nawal Salman**: 22k-4236  
- **Hafsa Atiqi**: 22k-4584  
- **Raaiha Syed**: 22k-4460  

---

## 1. Executive Summary

### Project Overview
This project aimed at creating an AI-based two-player labyrinth game where a human competes against an AI opponent. The game features a dynamic 7x7 grid with shifting rows/columns, trap movements, and goal collection. The AI uses the **Minimax algorithm with Alpha-Beta pruning** for strategic shifting decisions and the **A\*** algorithm for pathfinding during movement, creating a challenging and intelligent adversary.

---

## 2. Introduction

### Background
The traditional maze/labyrinth game involves navigating a static grid to reach goals while avoiding obstacles. This project builds on that concept by introducing dynamic grid manipulation and AI opponents. The labyrinth concept was chosen for its suitability in testing both adversarial AI and pathfinding algorithms. Key innovations include turn-based shift mechanics, moving traps, and goal collection.

### Objectives of the Project
- Implement a two-player labyrinth game (human vs AI).  
- Incorporate dynamic shifting of the game grid.  
- Develop and integrate AI using Minimax for shifting and A\* for movement.  
- Test the AI's performance against a human player.  

---

## 3. Game Description

### Original Game Rules
In conventional maze-based games, players move through a fixed grid to reach a goal or avoid traps. Movement is often deterministic and strategic planning is static due to an unchanging environment.

### Innovations and Modifications
- The grid can be altered by shifting rows or columns.  
- Traps dynamically move towards the nearest player every three turns.  
- Players alternate between a shift phase and a movement phase.  
- First to collect three goals wins.  

---

## 4. AI Approach and Methodology

### AI Techniques Used
- **Minimax with Alpha-Beta Pruning**: Used during the AI's shift phase to select the most strategic row/column shift.  
- **A\* Search Algorithm**: Used during the movement phase to calculate the shortest path to the nearest goal while avoiding traps.  

### Algorithm and Heuristic Design
- **Minimax**: The evaluation function considers proximity to goals, traps, and potential future configurations after shifting.  
- **A\***: Heuristic based on Manhattan distance from the AI to the nearest goal, penalizing paths near traps.  

### AI Performance Evaluation
- The AI’s performance was measured by win rate against human players, decision-making speed, and path accuracy.  
- AI won approximately **65%** of test matches.  
- Average decision-making time was **~1.5 seconds** per turn.  

---

## 5. Game Mechanics and Rules

### Modified Game Rules
- Each turn consists of a **Shift Phase** (row/column shift) followed by a **Move Phase** (path to goal).  
- Traps move every three turns towards the nearest player.  
- Players may not shift the same row/column two turns in a row.  

### Turn-based Mechanics
- Human and AI alternate turns.  
- Each turn includes:  
  - **Shift phase** (altering the grid layout).  
  - **Move phase** (step-by-step toward the nearest goal).  
- Traps update after every third complete round (human+AI turns).  

### Winning Conditions
- First player (human or AI) to collect **3 goals** is declared the winner.  

---

## 6. Implementation and Development

### Development Process
- Designed game grid and rules.  
- Developed game logic and player interaction.  
- Implemented AI algorithms for shifting and movement.  
- Integrated all components using Python and Pygame.  

### Programming Languages and Tools
- **Programming Language**: Python  
- **Libraries**: Pygame, random, heapq  
- **Tools**: GitHub (version control), Visual Studio Code (IDE)  

### Challenges Encountered
- Handling dynamic trap movements without excessive performance overhead.  
- Designing effective evaluation functions for Minimax that balance goal proximity and trap avoidance.  
- Ensuring smooth integration between shifting mechanics and AI pathfinding.  

---

## 7. Team Contributions

- **Nawal Salman (22k-4236)**: Developed the Minimax algorithm with Alpha-Beta pruning and implemented A\* pathfinding for movement.  
- **Raaiha Syed (22k-4460)**: Designed and implemented the shifting logic, grid structure, and trap movement mechanics.  
- **Hafsa Atiqi (22k-4584)**: Built the user interface using Pygame, integrated game phases (shift and move), and conducted AI performance testing.  

---

## 8. Results and Discussion

### AI Performance
- **Win Rate**: ~65% in test matches against human players.  
- **Decision Time**: Averaged ~1.5 seconds per turn (shift + move).  
- **Effectiveness**:  
  - AI reliably avoided traps and reached goals efficiently.  
  - Strategic shifts disrupted human paths effectively in many matches.  

---

## 9. References
- Russell, S., & Norvig, P. (2016). *Artificial Intelligence: A Modern Approach* (3rd ed.)  
