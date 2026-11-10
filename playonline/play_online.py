#!/usr/bin/env python3
"""
Play Flappy Bird Online (https://flappybird.io/) with a trained NEAT Neural Network.

Converts on-screen bird and pipe positions into the 3 NEAT inputs:
    inputs = (bird.y, abs(bird.y - pipe.height), abs(bird.y - pipe.bottom))
and decides when to flap based on output > 0.5.
"""

import os
import sys
import time
import re
import argparse
from playwright.sync_api import sync_playwright

from neural_net import FlappyNeuralNet

BROWSER_PATHS = [
    "/usr/bin/brave",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]

# Patterns for blocking ad and tracker domains
AD_BLOCK_REGEX = re.compile(
    r"(doubleclick|googlesyndication|google-analytics|adservice|posthog|adnxs|pubmatic|criteo|amazon-adsystem|video-ad|adsystem|pagead)",
    re.I
)

def find_browser_executable():
    for path in BROWSER_PATHS:
        if os.path.exists(path):
            return path
    return None

def build_game_bridge_script():
    """
    JavaScript bridge that hooks the canvas rendering pipeline in real-time,
    blocks ad overlays, converts canvas sprite coordinates to screen pixels,
    and evaluates the neural network on each frame at 60 FPS.
    """
    return """
    // Automatically hide ad elements and video overlays once DOM is ready
    window.addEventListener('DOMContentLoaded', () => {
        const adStyle = document.createElement('style');
        adStyle.textContent = `
            .video-ad-overlay, #takeover, .island-promo, [class*="ad-"] {
                display: none !important;
                opacity: 0 !important;
                pointer-events: none !important;
            }
        `;
        if (document.head) {
            document.head.appendChild(adStyle);
        }
    });

    const MODELS = {
        "infinite": {
            "bias": -24.0,
            "w_y": 0.0,
            "w_top": 1.0,
            "w_bottom": -1.0,
            "jumpThreshold": 0.0,
            "minFlapIntervalMs": 133
        },
        "local": {
            "bias": 1.105469591798359,
            "w_y": 0.12744297461526943,
            "w_top": 0.32346299460876876,
            "w_bottom": -0.9744945521357289,
            "jumpThreshold": 0.30,
            "minFlapIntervalMs": 140
        }
    };

    window.aiConfig = {
        enabled: true,
        scaleY: 730.0 / 575.0,
        weights: MODELS[window.aiModelChoice || "infinite"],
        jumpThreshold: (MODELS[window.aiModelChoice || "infinite"]).jumpThreshold,
        minFlapIntervalMs: (MODELS[window.aiModelChoice || "infinite"]).minFlapIntervalMs
    };

    window.aiStats = {
        score: 0,
        highScore: 0,
        runs: 0,
        steps: 0,
        alive: true,
        deathCause: null,
        lastInputs: [0, 0, 0],
        lastOutput: 0
    };

    window.aiRawPipes = [];
    window.aiBird = { x: 0, y: 0 };

    // Intercept 2D canvas drawing to extract exact on-screen coordinates
    const origDrawImage = CanvasRenderingContext2D.prototype.drawImage;
    CanvasRenderingContext2D.prototype.drawImage = function(...args) {
        const matrix = this.getTransform();
        const sx = args[1], sy = args[2], sw = args[3], sh = args[4];
        const dx = args[5], dy = args[6], dw = args[7], dh = args[8];
        const screenX = matrix.a * dx + matrix.c * dy + matrix.e;
        const screenY = matrix.b * dx + matrix.d * dy + matrix.f;
        const screenW = matrix.a * dw;

        // Bird sprite identification
        if ((sx === 3624 || sx === 3852 || sx === 3404) && sw === 220) {
            window.aiBird.x = screenX;
            window.aiBird.y = screenY;
        }

        // Pipe sprite identification
        if (sx === 1452 && sw === 206) {
            window.aiRawPipes.push({ x: screenX, y: screenY, w: screenW });
        }

        return origDrawImage.apply(this, args);
    };

    let lastFlapTime = 0;

    function aiLoop() {
        if (!window.aiConfig.enabled) return;
        const g = window.__game;

        if (g) {
            window.aiStats.score = g.score;
            if (g.score > window.aiStats.highScore) {
                window.aiStats.highScore = g.score;
            }

            if (g.currentState === 'getready') {
                g.flap();
            } else if (g.currentState === 'gameover') {
                if (window.aiStats.alive) {
                    window.aiStats.alive = false;
                    window.aiStats.deathCause = g.deathCause;
                }
            } else if (g.currentState === 'play') {
                window.aiStats.alive = true;
                window.aiStats.steps++;

                // Extract exact screen positions from game snapshot or canvas
                const snap = g.renderSnapshot ? g.renderSnapshot(1.0) : null;
                const scaleY = window.aiConfig.scaleY;
                let birdY = window.aiBird.y * scaleY;
                let pipeHeight = 250.0;
                let pipeBottom = 450.0;

                if (snap && snap.pipes) {
                    // In canvas pixels: screen center is ~360, canvas scale factor is ~260
                    birdY = (360 - snap.birdY * 260) * scaleY;
                    let nextP = null;
                    for (const p of snap.pipes) {
                        if (p.x > -0.25) {
                            nextP = p;
                            break;
                        }
                    }
                    if (nextP) {
                        // gapY is center, halfGap is half gap size in flappybird.io coordinates
                        const screenTop = 360 - (nextP.gapY + nextP.halfGap) * 260;
                        const screenBot = 360 - (nextP.gapY - nextP.halfGap) * 260;
                        pipeHeight = screenTop * scaleY;
                        pipeBottom = screenBot * scaleY;
                    }
                } else {
                    // Fallback to canvas draw tracking
                    const groups = {};
                    for (const p of window.aiRawPipes) {
                        const key = Math.round(p.x / 10) * 10;
                        if (!groups[key]) groups[key] = [];
                        groups[key].push(p);
                    }
                    window.aiRawPipes = [];

                    const pipes = [];
                    for (const k in groups) {
                        const pts = groups[k];
                        if (pts.length >= 2) {
                            const yVals = pts.map(p => p.y);
                            pipes.push({
                                x: pts[0].x,
                                w: pts[0].w,
                                top: Math.min(...yVals),
                                bottom: Math.max(...yVals)
                            });
                        }
                    }
                    pipes.sort((a, b) => a.x - b.x);

                    let nextPipe = null;
                    for (const p of pipes) {
                        if (p.x + p.w > window.aiBird.x - 35) {
                            nextPipe = p;
                            break;
                        }
                    }
                    if (nextPipe) {
                        pipeHeight = nextPipe.top * scaleY;
                        pipeBottom = nextPipe.bottom * scaleY;
                    }
                }

                const distTop = Math.abs(birdY - pipeHeight);
                const distBottom = Math.abs(birdY - pipeBottom);

                window.aiStats.lastInputs = [birdY, distTop, distBottom];

                // Evaluate the neural network
                const w = window.aiConfig.weights;
                const z = w.bias + (w.w_y * birdY) + (w.w_top * distTop) + (w.w_bottom * distBottom);
                const output = Math.tanh(z);
                window.aiStats.lastOutput = output;

                const now = performance.now();
                if (output > window.aiConfig.jumpThreshold && (now - lastFlapTime > window.aiConfig.minFlapIntervalMs)) {
                    g.flap();
                    lastFlapTime = now;
                }
            }
        }
        requestAnimationFrame(aiLoop);
    }

    window.startAI = function() {
        requestAnimationFrame(aiLoop);
    };

    window.restartGame = function() {
        const g = window.__game;
        if (g) {
            g.restart();
            window.aiStats.alive = true;
            window.aiStats.steps = 0;
            window.aiStats.runs++;
        }
    };
    """

def run_flappy_ai(headless=False, max_runs=1, target_score=100, model="online"):
    browser_exe = find_browser_executable()
    if not browser_exe:
        print("[!] Error: No compatible browser (Brave / Chrome / Chromium) found.")
        sys.exit(1)

    print(f"[*] Starting Flappy Bird AI on https://flappybird.io/")
    print(f"[*] Browser executable: {browser_exe}")
    print(f"[*] Mode: {'Headless' if headless else 'Headed (visible window)'}")
    print(f"[*] Neural Net Model: '{model}' (0 hidden layers, 3 inputs)")
    print(f"    Inputs = (bird.y, abs(bird.y - pipe.height), abs(bird.y - pipe.bottom))")
    print("-" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=browser_exe,
            headless=headless,
            args=[
                "--no-default-browser-check",
                "--disable-infobars",
                "--disable-background-networking"
            ]
        )
        context = browser.new_context(viewport={"width": 1280, "height": 720})
        page = context.new_page()

        # Set model choice in window
        page.add_init_script(f"window.aiModelChoice = '{model}';")

        # Block ads and video ad requests
        page.route(AD_BLOCK_REGEX, lambda route: route.abort())

        # Inject real-time screen coordinate extractor and AI loop
        page.add_init_script(build_game_bridge_script())

        print("[*] Navigating to https://flappybird.io/ ...")
        # Use domcontentloaded to avoid hanging on video ads / networkidle
        page.goto("https://flappybird.io/", wait_until="domcontentloaded")
        page.wait_for_selector("canvas", timeout=15000)
        page.wait_for_timeout(1000)

        # Start the AI loop
        page.evaluate("() => window.startAI()")
        print("[+] AI started! Flapping into the game...\n")

        total_runs = 0
        overall_best = 0

        while total_runs < max_runs:
            total_runs += 1
            print(f"=== RUN {total_runs}/{max_runs} ===")
            
            last_score = -1
            run_active = True

            while run_active:
                time.sleep(0.3)
                try:
                    stats = page.evaluate("() => window.aiStats")
                except Exception:
                    # User closed the window or connection dropped
                    print("  [!] Browser window closed by user.")
                    browser.close()
                    return
                
                score = stats["score"]
                if score != last_score:
                    last_score = score
                    inputs = stats.get("lastInputs", [0, 0, 0])
                    out = stats.get("lastOutput", 0.0)
                    print(f"  Score: {score:2d} | bird.y: {inputs[0]:.1f} | d_top: {inputs[1]:.1f} | d_bot: {inputs[2]:.1f} | NN out: {out:.3f}")
                
                if score > overall_best:
                    overall_best = score

                if not stats["alive"]:
                    print(f"  [-] Crash! Death cause: {stats.get('deathCause', 'collision')}. Run score: {score}")
                    run_active = False

                if score >= target_score:
                    print(f"\n[★] Target score {target_score} achieved!")
                    run_active = False

            if total_runs < max_runs:
                print("  [*] Restarting for next run...\n")
                page.wait_for_timeout(1000)
                try:
                    page.evaluate("() => window.restartGame()")
                except Exception:
                    break
                page.wait_for_timeout(500)

        print("\n" + "=" * 60)
        print(f"[*] Session Complete! Runs played: {total_runs} | High Score: {overall_best}")
        print("=" * 60)
        browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NEAT AI on https://flappybird.io/")
    parser.add_argument("--headless", action="store_true", help="Run without visible browser window")
    parser.add_argument("--runs", type=int, default=3, help="Number of games to play (default: 3)")
    parser.add_argument("--target", type=int, default=50, help="Target score before stopping (default: 50)")
    parser.add_argument("--model", choices=["infinite", "local"], default="infinite",
                        help="Choose model weights: 'infinite' (plays indefinitely) or 'local' (your original genome)")
    args = parser.parse_args()

    run_flappy_ai(headless=args.headless, max_runs=args.runs, target_score=args.target, model=args.model)

