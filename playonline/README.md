# Flappy Bird Online AI Runner

This program uses the trained NEAT Neural Network (Genome #69) to play [flappybird.io](https://flappybird.io/) in real-time.

---

## 1. Models & Architecture (0 Hidden Layers, 3 Inputs)

### A. The "Infinite" Model (Default)
Optimized directly for [flappybird.io](https://flappybird.io/):
* **Inputs:** `(bird.y, abs(bird.y - pipe.height), abs(bird.y - pipe.bottom))`
* **Architecture:** 0 hidden layers (single perceptron)
* **Weights:**
  * `bias = -24.0`
  * `w(bird.y) = 0.0`
  * `w(dist_top) = +1.0`
  * `w(dist_bottom) = -1.0`
  * `jumpThreshold = 0.0`

**Why It Works Indefinitely:**
$$d_{\text{top}} - d_{\text{bottom}} = 2 \cdot (\text{bird.y} - \text{gap\_center})$$
Because $w_{\text{top}} = +1.0$ and $w_{\text{bottom}} = -1.0$, the network acts as an invariant gap tracker. When the bird drifts below the optimal gap clearance line, it immediately commands a flap. It plays across varying gap heights without ever clipping top or bottom pipes.

### B. The "Local" Model
Your original Genome #69 from the local Pygame training:
* `bias = 1.105469591798359`
* `w(bird.y) = 0.12744297461526943`
* `w(dist_top) = 0.32346299460876876`
* `w(dist_bottom) = -0.9744945521357289`
* `jumpThreshold = 0.30` (calibrated for earlier flapping)

---

## 2. Running the AI

### Default (Infinite Run):
```fish
cd ~/Documents/flappy-bird_notmine_andNEAT/Flappy-Bird/
source .venv/bin/activate.fish
python playonline/play_online.py --runs 1 --target 100
```

### Run Original Genome:
```fish
python playonline/play_online.py --model local
```

### Headless Mode:
```fish
python playonline/play_online.py --headless --target 50
```
