#!/usr/bin/env python3
"""
Evolutionary Trainer for Flappy Bird Online (https://flappybird.io/).
Evolves a 0-hidden-layer 3-input Perceptron directly against the online game physics
until it finds weights that can navigate the pipes indefinitely.
"""

import os
import sys
import random
import numpy as np
from playwright.sync_api import sync_playwright

BROWSER_PATHS = [
    "/usr/bin/brave",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]

def find_browser():
    for p in BROWSER_PATHS:
        if os.path.exists(p):
            return p
    return None

def train():
    exe = find_browser()
    if not exe:
        print("[!] No browser found.")
        return

    print("[*] Launching headless browser for in-engine training...")
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.goto("https://flappybird.io/", wait_until="domcontentloaded")
        page.wait_for_selector("canvas")

        # In-engine evolutionary loop
        print("[*] Starting evolutionary training of 3-input perceptron (0 hidden layers)...")
        results = page.evaluate("""() => {
            const scaleY = 730.0 / 575.0;
            
            // Evaluates a single genome in fast headless simulation
            function evaluate(weights, maxSteps = 10000) {
                const g = window.__game;
                g.restart();
                g.flap();

                let lastFlapStep = -20;
                let steps = 0;

                while (g.currentState === 'play' && steps < maxSteps) {
                    steps++;
                    g.step(1/60);

                    // Read snapshot
                    const snap = g.renderSnapshot(1.0);
                    const pipes = snap.pipes;

                    // Upcoming pipe
                    let nextPipe = null;
                    for (const p of pipes) {
                        if (p.x + 0.15 > 0) {
                            nextPipe = p;
                            break;
                        }
                    }

                    // Convert snapshot coordinates (-1 to 1.2) to screen pixels (0 to 720)
                    // At canvas height 720: screenY approx 360 - snap.birdY * 260
                    const screenBirdY = 360 - snap.birdY * 260;
                    let screenTop, screenBot;
                    if (nextPipe) {
                        const gapCenter = 360 - nextPipe.gapY * 260;
                        const halfGap = nextPipe.halfGap * 260;
                        screenTop = gapCenter - halfGap;
                        screenBot = gapCenter + halfGap;
                    } else {
                        screenTop = 200;
                        screenBot = 360;
                    }

                    // Convert to Pygame training scale (floor at 730)
                    const by = screenBirdY * scaleY;
                    const pTop = screenTop * scaleY;
                    const pBot = screenBot * scaleY;

                    const dTop = Math.abs(by - pTop);
                    const dBot = Math.abs(by - pBot);

                    // 3-input Perceptron
                    const z = weights.bias + (weights.wy * by) + (weights.wtop * dTop) + (weights.wbot * dBot);
                    const out = Math.tanh(z);

                    if (out > weights.threshold && (steps - lastFlapStep >= weights.minInterval)) {
                        g.flap();
                        lastFlapStep = steps;
                    }
                }

                return {
                    score: g.score,
                    steps: steps,
                    deathCause: g.deathCause
                };
            }

            // Genetic Algorithm
            const POP_SIZE = 30;
            const GENERATIONS = 30;
            let population = [];

            // Seed with variants of base gap-centering and user's genome
            for (let i = 0; i < POP_SIZE; i++) {
                population.push({
                    bias: (Math.random() - 0.5) * 4.0,
                    wy: (Math.random() - 0.5) * 0.5,
                    wtop: Math.random() * 0.8,
                    wbot: -Math.random() * 0.8,
                    threshold: 0.1 + Math.random() * 0.4,
                    minInterval: 8 + Math.floor(Math.random() * 4),
                    fitness: 0,
                    score: 0
                });
            }

            // Include user-like seed
            population[0] = {
                bias: 1.105,
                wy: 0.127,
                wtop: 0.323,
                wbot: -0.974,
                threshold: 0.3,
                minInterval: 9,
                fitness: 0,
                score: 0
            };

            let bestEver = null;

            for (let gen = 0; gen < GENERATIONS; gen++) {
                // Evaluate
                for (let ind of population) {
                    const res = evaluate(ind, 8000);
                    ind.score = res.score;
                    ind.fitness = res.steps + (res.score * 500);
                    if (!bestEver || ind.fitness > bestEver.fitness) {
                        bestEver = JSON.parse(JSON.stringify(ind));
                    }
                }

                // If champion survived full run (score > 30)
                if (bestEver.score >= 30) break;

                // Sort descending
                population.sort((a, b) => b.fitness - a.fitness);

                // Elitism (keep top 4)
                const nextPop = population.slice(0, 4).map(p => JSON.parse(JSON.stringify(p)));

                // Mutate & crossover
                while (nextPop.length < POP_SIZE) {
                    const parent = population[Math.floor(Math.random() * 8)];
                    const child = JSON.parse(JSON.stringify(parent));
                    if (Math.random() < 0.8) child.bias += (Math.random() - 0.5) * 0.4;
                    if (Math.random() < 0.8) child.wy += (Math.random() - 0.5) * 0.05;
                    if (Math.random() < 0.8) child.wtop += (Math.random() - 0.5) * 0.1;
                    if (Math.random() < 0.8) child.wbot += (Math.random() - 0.5) * 0.1;
                    if (Math.random() < 0.5) child.threshold += (Math.random() - 0.5) * 0.1;
                    nextPop.push(child);
                }
                population = nextPop;
            }

            return bestEver;
        }""")

        print("\n" + "=" * 50)
        print("[+] Training Complete! Best Genome Found:")
        print(f"    Bias:        {results.get('bias'):.6f}")
        print(f"    w(bird.y):   {results.get('wy'):.6f}")
        print(f"    w(dist_top): {results.get('wtop'):.6f}")
        print(f"    w(dist_bot): {results.get('wbot'):.6f}")
        print(f"    Threshold:   {results.get('threshold'):.3f}")
        print(f"    Score:       {results.get('score')} pipes passed")
        print("=" * 50)
        browser.close()

if __name__ == "__main__":
    train()
