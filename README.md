# 🧬 Genetic Algorithm for TSP (Python)

A **Genetic Algorithm (GA)** implementation to solve the **Traveling Salesman Problem (TSP)** in Python.  
This project demonstrates **population initialization, selection, crossover, mutation**, and **fitness evaluation** for optimization problems. 🚀

---

## ✨ Features
- 🧬 Multi-strategy GA (selection, crossover, mutation)
- 📊 Tracks fitness and distance history over generations
- 🗺 Plots optimal route on a 2D city map
- 🔧 Configurable population size, generations, and GA parameters

---

## 🛠 Requirements
- 🐍 Python 3.7+
- 📦 Libraries: `matplotlib`

Install required packages with:
```bash
pip install matplotlib
```

---

## ▶️ How to Run
1. Make sure your dataset (e.g., `dj38.tsp`) is in the `data/` folder.
2. Run the main program:
```bash
python main.py
```

Expected outputs:
- Terminal prints the best route, fitness, and distance.
- Plots showing fitness history, distance history, and the final route on a 2D map.

---

## ⚙️ Configuration
- Modify GA parameters in `main.py`:
  ```python
  num_generations=600
  mutation_rate=0.1
  crossover_rate=0.8
  tournament_size=3
  ```
- You can change **selection**, **crossover**, and **mutation strategies** in `GeneticAlgorithm` class (`genetic.py`).

---

## 📂 Project Structure
```
├── data/                  # TSP datasets (.tsp files)
├── genetic.py             # Genetic Algorithm implementation
├── utils.py               # Helper functions (distance, fitness, plotting)
└── main.py                # Program entry point
```

---

## 🚀 Possible Improvements
- 👤 Custom GA operators (e.g., PMX crossover, adaptive mutation)
- 🔒 Parallelize population evaluation for faster computation
- 🎨 GUI to visualize GA evolution in real time
- 📈 Automatic parameter tuning or hyperparameter sweep

---

## 📜 License
MIT © Free to use and modify

---
✨ Built with ❤️ using Python
