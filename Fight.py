import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import pygame
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import re

# =========================
# Audio (pre-init for low latency) — MUST be before pygame.init()
# =========================
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=256)

# =========================
# Setup
# =========================
pygame.init()

# Screen / Panels
CAMERA_WIDTH  = 400
CAMERA_HEIGHT = 600
GAME_WIDTH    = 400
GAME_HEIGHT   = 600
TOTAL_WIDTH   = CAMERA_WIDTH + GAME_WIDTH
SCREEN_HEIGHT = 600

# Colors
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GREEN  = (0, 255, 0)
RED    = (255, 0, 0)
BLUE   = (0, 128, 255)
YELLOW = (255, 255, 0)
PURPLE = (160, 80, 220)
BROWN  = (139, 69, 19)
ORANGE = (255, 170, 0)
GOLD   = (255, 210, 60)
SHADOW = (0, 0, 0, 60)  # for reference

# Fighter props
FIGHTER_BASE_W = 60
FIGHTER_BASE_H = 80
FIGHTER_SPEED  = 3
PUNCH_RANGE    = 80

# Game props
FPS        = 60
FONT       = pygame.font.SysFont('Arial', 24)
SMALL_FONT = pygame.font.SysFont('Arial', 16)
INST_FONT  = pygame.font.SysFont('Arial', 12)

# Window
screen = pygame.display.set_mode((TOTAL_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Rehabilitation Fighter - Cartoon + Blinking City')
clock = pygame.time.Clock()

# =========================
# Audio: load & start
# =========================
pygame.mixer.init()
AUDIO_OK = True
try:
    # --- asset paths (change if needed)
    BGM_PATH          = "assets/bgm.wav"         # or .ogg/.wav
    SFX_PUNCH_PATH    = "assets/punch.ogg"
    SFX_COMBO_PATH    = "assets/combo.ogg"
    SFX_FIREBALL_PATH = "assets/fireball.ogg"
    SFX_HIT_PATH      = "assets/hit.ogg"

    # music
    pygame.mixer.music.load(BGM_PATH)
    pygame.mixer.music.set_volume(0.40)  # 0.0 ~ 1.0
    pygame.mixer.music.play(-1)          # loop

    # sfx
    sfx_punch    = pygame.mixer.Sound(SFX_PUNCH_PATH)
    sfx_combo    = pygame.mixer.Sound(SFX_COMBO_PATH)
    sfx_fireball = pygame.mixer.Sound(SFX_FIREBALL_PATH)
    sfx_hit      = pygame.mixer.Sound(SFX_HIT_PATH)

    sfx_punch.set_volume(0.7)
    sfx_combo.set_volume(0.8)
    sfx_fireball.set_volume(0.75)
    sfx_hit.set_volume(0.6)

except Exception as _e:
    print(f"[Audio] Failed to init/load: {_e}")
    AUDIO_OK = False
    class _NullSnd:
        def play(self): pass
        def set_volume(self, *_): pass
    sfx_punch = sfx_combo = sfx_fireball = sfx_hit = _NullSnd()

# =========================
# Universal Logger
# =========================
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# SIGNALS[name] = list of (t, value)
SIGNALS = defaultdict(list)

def _sanitize(name: str):
    return re.sub(r'[^a-zA-Z0-9_\-]+', '_', name)

def log_signal(name, value, t):
    """Log arbitrary signals; None values are ignored; bools are converted to 0/1."""
    if value is None:
        return
    if isinstance(value, bool):
        value = 1 if value else 0
    SIGNALS[name].append((float(t), float(value)))

EVENT_COUNTS = defaultdict(int)

def save_round_summary(round_idx: int):
    """
    Plot each signal recorded during this round (y-axis = value or 0/1, x-axis = time in seconds),
    and generate an event count bar chart and a full CSV.
    """
    out_dir = os.path.join(PLOTS_DIR, f"round{round_idx}")
    os.makedirs(out_dir, exist_ok=True)

    # --- Plot all signals ---
    for name, series in SIGNALS.items():
        if len(series) < 2:
            continue
        series = sorted(series, key=lambda x: x[0])
        ts = [p[0] for p in series]
        vs = [p[1] for p in series]

        plt.figure(figsize=(7,4))
        plt.plot(ts, vs)
        plt.title(f"{name} – Round {round_idx}")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.grid(True, linestyle="--", alpha=0.4)
        out_png = os.path.join(out_dir, f"{_sanitize(name)}.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

    # --- Event counts bar chart ---
    labels = ["move_forward","move_backward","punch_left","punch_right","combo_attack","fireball"]
    vals   = [EVENT_COUNTS.get(k,0) for k in labels]
    plt.figure(figsize=(8,4))
    plt.bar(labels, vals)
    plt.title(f"Event Counts – Round {round_idx}")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.1, str(v), ha="center", va="bottom")
    plt.tight_layout()
    out_bar = os.path.join(out_dir, f"event_counts.png")
    plt.savefig(out_bar, dpi=150); plt.close()

    # --- Full CSV (wide table: each column = one signal) ---
    # Aggregate all time points (deduplicate and sort)
    all_times = sorted({t for series in SIGNALS.values() for (t, _) in series})
    col_names = sorted(SIGNALS.keys())
    # Build name -> dict(time->value) mapping
    series_map = {name: {t: v for (t, v) in series} for name, series in SIGNALS.items()}
    csv_path = os.path.join(out_dir, f"signals.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_sec"] + col_names)
        for t in all_times:
            row = [t]
            for name in col_names:
                v = series_map.get(name, {}).get(t, "")
                row.append(v)
            w.writerow(row)

# =========================
# MediaPipe
# =========================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1
)

cap = cv2.VideoCapture(0)
cap.set(3, CAMERA_WIDTH)
cap.set(4, CAMERA_HEIGHT)

# =========================
# City Background with blinking buildings
# =========================
def lerp(a, b, t): return a + (b - a) * t

class CityBackground:
    def __init__(self, offset_x, width, height, seed=None):
        self.offset_x = offset_x
        self.w = width
        self.h = height
        self.ground_y = height // 2
        self.rng = random.Random(seed or 42)
        self.stars = self._gen_stars(40)
        self.buildings = self._gen_buildings()
        # Twinkle timing
        self.last_update = 0.0

    def _gen_stars(self, n):
        stars = []
        for _ in range(n):
            x = self.rng.randint(self.offset_x + 5, self.offset_x + self.w - 5)
            y = self.rng.randint(10, self.ground_y - 50)
            phase = self.rng.random() * 2 * math.pi
            stars.append({"x": x, "y": y, "phase": phase, "speed": self.rng.uniform(0.5, 1.2)})
        return stars

    def _gen_buildings(self):
        blds = []
        x = self.offset_x + 20
        while x < self.offset_x + self.w - 40:
            bw = self.rng.randint(50, 90)
            bh = self.rng.randint(100, 180)
            if x + bw > self.offset_x + self.w - 20:
                bw = self.offset_x + self.w - 20 - x
            color = (40 + self.rng.randint(0, 30), 40 + self.rng.randint(0, 30), 50 + self.rng.randint(0, 30))
            building = {
                "rect": pygame.Rect(x, self.ground_y - bh, bw, bh),
                "color": color,
                "windows": self._gen_windows(x, self.ground_y - bh, bw, bh),
            }
            blds.append(building)
            x += bw + self.rng.randint(12, 26)
        return blds

    def _gen_windows(self, bx, by, bw, bh):
        # Window grid
        windows = []
        pad_x, pad_y = 10, 18
        win_w, win_h = 12, 16
        gap_x, gap_y = 10, 12
        cols = max(1, (bw - 2 * pad_x + gap_x) // (win_w + gap_x))
        rows = max(2, (bh - 2 * pad_y + gap_y) // (win_h + gap_y))
        for r in range(rows):
            for c in range(cols):
                wx = bx + pad_x + c * (win_w + gap_x)
                wy = by + pad_y + r * (win_h + gap_y)
                initial_on = (random.random() < 0.65)
                next_toggle = time.time() + random.uniform(0.2, 2.5)  # independent blink timers
                windows.append({
                    "rect": pygame.Rect(wx, wy, win_w, win_h),
                    "on": initial_on,
                    "next": next_toggle
                })
        return windows

    def update(self, now):
        # toggle individual windows when their timers elapse
        for b in self.buildings:
            for w in b["windows"]:
                if now >= w["next"]:
                    # 40% chance to change state at each tick, then schedule next tick soon-ish
                    if random.random() < 0.40:
                        w["on"] = not w["on"]
                    w["next"] = now + random.uniform(0.25, 2.0)

    def draw(self, screen):
        # Sky gradient (night-ish)
        for y in range(self.ground_y):
            t = y / max(1, self.ground_y - 1)
            r = int(lerp(10, 25, t))
            g = int(lerp(10, 30, t))
            b = int(lerp(30, 70, t))
            pygame.draw.line(screen, (r, g, b), (self.offset_x, y), (self.offset_x + self.w, y))

        # Twinkling stars
        tnow = pygame.time.get_ticks() / 1000.0
        for s in self.stars:
            k = (math.sin(tnow * s["speed"] + s["phase"]) + 1.0) * 0.5  # 0..1
            col = (int(200 + 55 * k), int(200 + 55 * k), int(220 + 35 * k))
            pygame.draw.circle(screen, col, (s["x"], s["y"]), 1)

        # Ground
        pygame.draw.rect(screen, (30, 30, 30), pygame.Rect(self.offset_x, self.ground_y, self.w, self.h - self.ground_y))

        # Buildings + windows
        for b in self.buildings:
            # building body (slight vertical gradient via 2 rects)
            rect = b["rect"]
            body_top = (max(0, b["color"][0]-8), max(0, b["color"][1]-8), max(0, b["color"][2]-8))
            body_bot = (min(255, b["color"][0]+8), min(255, b["color"][1]+8), min(255, b["color"][2]+8))
            pygame.draw.rect(screen, body_top, rect)
            pygame.draw.rect(screen, body_bot, pygame.Rect(rect.x, rect.y + rect.height//2, rect.width, rect.height//2))
            pygame.draw.rect(screen, (20, 20, 20), rect, 2)

            # windows (blink on/off)
            for w in b["windows"]:
                if w["on"]:
                    # a warm window color with subtle variance
                    warm = (255, 220 + random.randint(-10, 10), 120 + random.randint(-10, 10))
                    pygame.draw.rect(screen, warm, w["rect"], border_radius=2)
                    pygame.draw.rect(screen, (40, 20, 0), w["rect"], 1, border_radius=2)
                else:
                    off = (35, 35, 40)
                    pygame.draw.rect(screen, off, w["rect"], border_radius=2)
                    pygame.draw.rect(screen, (20, 20, 25), w["rect"], 1, border_radius=2)

# Instance
city = CityBackground(CAMERA_WIDTH, GAME_WIDTH, GAME_HEIGHT)

# =========================
# Cartoon fighter drawing
# =========================
def draw_round_rect(surface, rect, color, radius=12, width=0, outline=None):
    pygame.draw.rect(surface, color, rect, width=width, border_radius=radius)
    if outline and width == 0:
        pygame.draw.rect(surface, outline, rect, width=2, border_radius=radius)

def draw_limb(surface, start, end, thickness, color, outline=(0,0,0)):
    # thick line with circle caps (cartoon limb)
    pygame.draw.line(surface, color, start, end, thickness)
    pygame.draw.circle(surface, color, start, thickness//2)
    pygame.draw.circle(surface, color, end, thickness//2)
    if outline:
        pygame.draw.line(surface, outline, start, end, 2)
        pygame.draw.circle(surface, outline, start, thickness//2, 2)
        pygame.draw.circle(surface, outline, end, thickness//2, 2)

def draw_cartoon_fighter(screen, x, y, facing_right=True, action="idle", main_color=(50,170,255), accent=(255,255,255)):
    """
    x,y = hip center; y is baseline hip
    """
    t = pygame.time.get_ticks() / 1000.0
    # gentle breathing / idle bob
    bob = 2.5 * math.sin(t * 2.5)
    scale = 1.0  # could scale if needed
    fw = int(FIGHTER_BASE_W * scale)
    fh = int(FIGHTER_BASE_H * scale)

    # orientation
    dir_sign = 1 if facing_right else -1

    # torso
    torso_w = int(fw * 0.75)
    torso_h = int(fh * 0.70)
    torso_rect = pygame.Rect(x - torso_w//2, int(y - torso_h + bob), torso_w, torso_h)
    draw_round_rect(screen, torso_rect, main_color, radius=16, outline=(0,0,0))

    # subtle chest highlight
    hil = pygame.Surface((torso_w, torso_h), pygame.SRCALPHA)
    pygame.draw.ellipse(hil, (255,255,255,40), pygame.Rect(torso_w//6, torso_h//10, torso_w//2, torso_h//2))
    screen.blit(hil, torso_rect.topleft)

    # belt
    belt_rect = pygame.Rect(torso_rect.x, torso_rect.y + int(torso_h*0.65), torso_w, int(torso_h*0.12))
    draw_round_rect(screen, belt_rect, (30,30,30), radius=6, outline=(0,0,0))

    # head
    head_r = int(fw * 0.28)
    head_center = (x, torso_rect.y - head_r - 6)
    pygame.draw.circle(screen, (255, 220, 180), head_center, head_r)
    pygame.draw.circle(screen, (0,0,0), head_center, head_r, 2)

    # hairband / headband
    hb_rect = pygame.Rect(head_center[0]-head_r, head_center[1]-4, head_r*2, 10)
    draw_round_rect(screen, hb_rect, (240,60,80), radius=4, outline=(0,0,0))

    # eyes + brows (look direction)
    eye_dx = 6 * dir_sign
    pygame.draw.circle(screen, (255,255,255), (head_center[0]-8+eye_dx, head_center[1]-2), 4)
    pygame.draw.circle(screen, (0,0,0), (head_center[0]-8+eye_dx, head_center[1]-2), 2)
    pygame.draw.circle(screen, (255,255,255), (head_center[0]+12+eye_dx, head_center[1]-2), 4)
    pygame.draw.circle(screen, (0,0,0), (head_center[0]+12+eye_dx, head_center[1]-2), 2)
    pygame.draw.line(screen, (0,0,0), (head_center[0]-16, head_center[1]-10), (head_center[0]-2, head_center[1]-12), 2)
    pygame.draw.line(screen, (0,0,0), (head_center[0]+4, head_center[1]-12), (head_center[0]+20, head_center[1]-10), 2)

    # mouth
    pygame.draw.arc(screen, (0,0,0),
                    pygame.Rect(head_center[0]-8, head_center[1]+6, 16, 10),
                    math.radians(10), math.radians(170), 2)

    # shoulders anchor
    shoulder_y = torso_rect.y + int(torso_h * 0.22)

    # arm targets by action
    left_hand = [x - int(fw*0.45), shoulder_y + int(fh*0.20)]
    right_hand = [x + int(fw*0.45), shoulder_y + int(fh*0.20)]
    if action == "punch_left":
        left_hand = [x + dir_sign * int(fw*0.65), shoulder_y - int(fh*0.05)]
    elif action == "punch_right":
        right_hand = [x + dir_sign * int(fw*0.65), shoulder_y + int(fh*0.03)]
    elif action == "combo":
        left_hand  = [x + dir_sign * int(fw*0.62), shoulder_y - int(fh*0.08)]
        right_hand = [x + dir_sign * int(fw*0.62), shoulder_y + int(fh*0.06)]

    # arms (upper from shoulder to elbow, lower elbow to hand)
    # We simplify as one limb segment cartoon-style
    limb_th = max(8, int(fw*0.17))
    glove_col = (255, 245, 245)
    # left arm
    draw_limb(screen, (x - int(fw*0.36), shoulder_y), left_hand, limb_th, (255, 220, 180))
    pygame.draw.circle(screen, glove_col, left_hand, limb_th//2 + 2)
    pygame.draw.circle(screen, (0,0,0), left_hand, limb_th//2 + 2, 2)
    # right arm
    draw_limb(screen, (x + int(fw*0.36), shoulder_y), right_hand, limb_th, (255, 220, 180))
    pygame.draw.circle(screen, glove_col, right_hand, limb_th//2 + 2)
    pygame.draw.circle(screen, (0,0,0), right_hand, limb_th//2 + 2, 2)

    # legs (simple stance)
    leg_th = max(10, int(fw*0.22))
    left_foot  = (x - int(fw*0.22), y + 32)
    right_foot = (x + int(fw*0.22), y + 32)
    draw_limb(screen, (x - int(fw*0.12), y), left_foot, leg_th, (50,50,55))
    draw_limb(screen, (x + int(fw*0.12), y), right_foot, leg_th, (50,50,55))
    # boots
    boot_w = int(fw*0.36); boot_h = 14
    left_boot  = pygame.Rect(left_foot[0]-boot_w//2, left_foot[1]-boot_h//2, boot_w, boot_h)
    right_boot = pygame.Rect(right_foot[0]-boot_w//2, right_foot[1]-boot_h//2, boot_w, boot_h)
    draw_round_rect(screen, left_boot, (240, 60, 80), radius=7, outline=(0,0,0))
    draw_round_rect(screen, right_boot,(240, 60, 80), radius=7, outline=(0,0,0))

# =========================
# Entities
# =========================
class Fighter:
    def __init__(self, x, y, is_player=True, main_color=(50,170,255)):
        self.x = x
        self.y = y
        self.max_health = 100
        self.health = self.max_health
        self.is_player = is_player
        self.main_color = main_color
        self.facing_right = True if is_player else False
        self.action = "idle"
        self.action_timer = 0
        self.punch_cooldown = 0
        self.combo_cooldown = 0
        self.hit_flash = 0
        self.move_cooldown = 0

    def update(self):
        self.action_timer = max(0, self.action_timer - 1)
        self.punch_cooldown = max(0, self.punch_cooldown - 1)
        self.combo_cooldown = max(0, self.combo_cooldown - 1)
        self.move_cooldown = max(0, self.move_cooldown - 1)
        self.hit_flash = max(0, self.hit_flash - 1)
        if self.action_timer <= 0:
            self.action = "idle"

    def move_forward(self):
        if self.move_cooldown <= 0:
            if self.facing_right:
                self.x = min(GAME_WIDTH - 50, self.x + FIGHTER_SPEED * 3)
            else:
                self.x = max(50, self.x - FIGHTER_SPEED * 3)
            self.move_cooldown = 10

    def move_backward(self):
        if self.move_cooldown <= 0:
            if self.facing_right:
                self.x = max(50, self.x - FIGHTER_SPEED * 3)
            else:
                self.x = min(GAME_WIDTH - 50, self.x + FIGHTER_SPEED * 3)
            self.move_cooldown = 10

    def punch_left(self):
        if self.punch_cooldown <= 0:
            self.action = "punch_left"; self.action_timer = 15; self.punch_cooldown = 15
            if AUDIO_OK: sfx_punch.play()
            return True
        return False

    def punch_right(self):
        if self.punch_cooldown <= 0:
            self.action = "punch_right"; self.action_timer = 15; self.punch_cooldown = 15
            if AUDIO_OK: sfx_punch.play()
            return True
        return False

    def combo_attack(self):
        if self.combo_cooldown <= 0:
            self.action = "combo"; self.action_timer = 25; self.combo_cooldown = 40
            if AUDIO_OK: sfx_combo.play()
            return True
        return False

    def take_damage(self, damage):
        self.health = max(0, self.health - damage)
        self.hit_flash = 15
        return self.health <= 0

    def draw(self, screen, offset_x):
        color = (255,255,255) if self.hit_flash > 10 else self.main_color
        draw_cartoon_fighter(screen, offset_x + int(self.x), int(self.y), self.facing_right, self.action, color)

class Fireball:
    def __init__(self, x, y, direction=1, speed=7, radius=9, color=ORANGE):
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = speed
        self.radius = radius
        self.color = color
        self.alive = True
        self.t0 = pygame.time.get_ticks() / 1000.0

    def update(self):
        self.x += self.speed * self.direction
        if self.x < 0 or self.x > GAME_WIDTH:
            self.alive = False

    def draw(self, screen, offset_x):
        cy = int(self.y) - FIGHTER_BASE_H // 2
        cx = offset_x + int(self.x)
        t = pygame.time.get_ticks() / 1000.0 - self.t0
        # core
        pygame.draw.circle(screen, (255, 210, 60), (cx, cy), self.radius)
        # aura rings (cartoony)
        pygame.draw.circle(screen, (255, 150, 40), (cx - 4*self.direction, cy), max(2, self.radius-3), 2)
        pygame.draw.circle(screen, (255, 240, 180), (cx + 3*self.direction, cy), max(1, self.radius-5), 1)
        # tiny sparks
        for i in range(3):
            sx = cx - self.direction * (self.radius + 4 + 3*i)
            sy = cy + int(2*math.sin(t*10 + i))
            pygame.draw.circle(screen, (255, 230, 120), (sx, sy), 1)

# =========================
# Pose helpers
# =========================
def vector(a, b): return np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float32)

def angle_between(a, b, c):
    ba = vector(b, a); bc = vector(b, c)
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba == 0 or nbc == 0: return None
    cosv = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

def get_pt(landmarks, lm): return np.array([landmarks[lm].x, landmarks[lm].y], dtype=np.float32)

def detect_elbow_angles(landmarks):
    try:
        LSH = get_pt(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        LEL = get_pt(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value)
        LWR = get_pt(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value)
        RSH = get_pt(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        REL = get_pt(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        RWR = get_pt(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value)
        la = angle_between(LSH, LEL, LWR)
        ra = angle_between(RSH, REL, RWR)
        if la is None or ra is None:
            return {"left_angle": None, "right_angle": None, "left_attack": False, "right_attack": False, "combo": False}
        left_attack = la < 90.0
        right_attack = ra < 90.0
        return {"left_angle": la, "right_angle": ra, "left_attack": left_attack, "right_attack": right_attack,
                "combo": left_attack and right_attack}
    except:
        return {"left_angle": None, "right_angle": None, "left_attack": False, "right_attack": False, "combo": False}

def detect_arm_raises(landmarks):
    try:
        LSH = get_pt(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        LEL = get_pt(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value)
        LWR = get_pt(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value)
        RSH = get_pt(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        REL = get_pt(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        RWR = get_pt(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value)
        WRIST_RAISE = 0.12
        ELBOW_RAISE = 0.07
        left_raised = (LWR[1] < LSH[1] - WRIST_RAISE) and (LEL[1] < LSH[1] - ELBOW_RAISE)
        right_raised = (RWR[1] < RSH[1] - WRIST_RAISE) and (REL[1] < RSH[1] - ELBOW_RAISE)
        return {"left_raised": left_raised, "right_raised": right_raised}
    except:
        return {"left_raised": False, "right_raised": False}

# =========================
# Hands helpers (3D + guards)
# =========================
def _np_xy(lms, idx):
    p = lms[idx]; return np.array([p.x, p.y], dtype=np.float32)

def _np_xyz(lms, idx):
    p = lms[idx]; return np.array([p.x, p.y, p.z], dtype=np.float32)

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def angle_between_vectors(u, v):
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0: return 180.0
    cosv = np.clip(np.dot(u, v) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

def joint_angle_3d(lm3, i, j, k):
    a = _np_xyz(lm3, i); b = _np_xyz(lm3, j); c = _np_xyz(lm3, k)
    v1 = _unit(a - b); v2 = _unit(c - b)
    return angle_between_vectors(v1, v2)

def finger_extended(lm2, lm3, mcp, pip, dip, tip,
                    EXT_PIP_MIN_DEG=150.0, EXT_DIP_MIN_DEG=145.0, EXT_LEN_RATIO_MIN=1.45):
    if lm3 is not None:
        pip_ang = joint_angle_3d(lm3, mcp, pip, dip)
        dip_ang = joint_angle_3d(lm3, pip, dip, tip)
    else:
        def joint_angle_2d(lm2, i, j, k):
            a = _np_xy(lm2, i); b = _np_xy(lm2, j); c = _np_xy(lm2, k)
            v1 = _unit(a - b); v2 = _unit(c - b)
            return angle_between_vectors(np.append(v1,0), np.append(v2,0))
        pip_ang = joint_angle_2d(lm2, mcp, pip, dip)
        dip_ang = joint_angle_2d(lm2, pip, dip, tip)
    if lm3 is not None:
        mcp_tip = np.linalg.norm(_np_xyz(lm3, tip) - _np_xyz(lm3, mcp))
        mcp_pip = np.linalg.norm(_np_xyz(lm3, pip) - _np_xyz(lm3, mcp)) or 1.0
    else:
        mcp_tip = np.linalg.norm(_np_xy(lm2, tip) - _np_xy(lm2, mcp))
        mcp_pip = np.linalg.norm(_np_xy(lm2, pip) - _np_xy(lm2, mcp)) or 1.0
    len_ratio_ok = (mcp_tip / mcp_pip) >= EXT_LEN_RATIO_MIN
    return (pip_ang >= EXT_PIP_MIN_DEG) and (dip_ang >= EXT_DIP_MIN_DEG) and len_ratio_ok

def per_hand_metrics(landmarks_2d, landmarks_3d):
    v_mid  = _unit(_np_xyz(landmarks_3d, 12) - _np_xyz(landmarks_3d, 9))
    v_ring = _unit(_np_xyz(landmarks_3d, 16) - _np_xyz(landmarks_3d, 13))
    angle3d = angle_between_vectors(v_mid, v_ring)
    palm_w = np.linalg.norm(_np_xy(landmarks_2d, 5) - _np_xy(landmarks_2d, 17)) or 1.0
    r_idx_mid  = np.linalg.norm(_np_xy(landmarks_2d, 8)  - _np_xy(landmarks_2d, 12)) / palm_w
    r_ring_pky = np.linalg.norm(_np_xy(landmarks_2d, 16) - _np_xy(landmarks_2d, 20)) / palm_w
    return angle3d, r_idx_mid, r_ring_pky

FINGER_IDX = {
    "Thumb":  [(1,2,3), (2,3,4)],   # MCP, IP
    "Index":  [(5,6,7), (6,7,8)],   # PIP, DIP
    "Middle": [(9,10,11),(10,11,12)],
    "Ring":   [(13,14,15),(14,15,16)],
    "Pinky":  [(17,18,19),(18,19,20)]
}

def finger_joint_angles(lm2, lm3):
    """
    Returns a dict mapping each finger to its joint angles, e.g.
    {'Index': {'PIP': deg, 'DIP': deg}, ...}. Thumb uses ('MCP','IP') keys.
    Prefers lm3 (3D landmarks) when available; falls back to a 2D approximation if not.
    """
    out = {}
    def ang(lm2, lm3, a,b,c):
        if lm3 is not None:
            return joint_angle_3d(lm3, a,b,c)
        else:
            # 2D approximation
            def _ang2d(i,j,k):
                p = _np_xy(lm2, i); q = _np_xy(lm2, j); r = _np_xy(lm2, k)
                v1 = _unit(np.append(p-q,0)); v2 = _unit(np.append(r-q,0))
                return angle_between_vectors(v1, v2)
            return _ang2d(a,b,c)

    for name, (pip_triplet, dip_triplet) in FINGER_IDX.items():
        a1 = ang(lm2, lm3, *pip_triplet)
        a2 = ang(lm2, lm3, *dip_triplet)
        if name == "Thumb":
            out[name] = {"MCP": a1, "IP": a2}
        else:
            out[name] = {"PIP": a1, "DIP": a2}
    return out

# =========================
# Combat helpers
# =========================
def resolve_hit(attacker, defender):
    distance = abs(attacker.x - defender.x)
    if distance > PUNCH_RANGE:
        return False
    if attacker.action == "punch_left" and attacker.action_timer > 10:
        defender.take_damage(5); attacker.action_timer = 10
        if AUDIO_OK: sfx_hit.play()
        return True
    if attacker.action == "punch_right" and attacker.action_timer > 10:
        defender.take_damage(5); attacker.action_timer = 10
        if AUDIO_OK: sfx_hit.play()
        return True
    if attacker.action == "combo" and attacker.action_timer > 20:
        defender.take_damage(15); attacker.action_timer = 20
        if AUDIO_OK: sfx_hit.play()
        return True
    return False

def reset_game():
    global player, enemy, game_start_time
    global prev_left_attack, prev_right_attack, prev_left_raise, prev_right_raise
    global last_attack_time, last_move_time
    global fireballs
    global v_ok_frames, v_active
    global ext_mid_ok, ext_ring_ok
    global last_both_fire_time, both_armed, both_inactive_frames

    player = Fighter(100, GAME_HEIGHT - 50, True, (50,170,255))
    enemy  = Fighter(300, GAME_HEIGHT - 50, False,(240, 90, 90))
    player.facing_right = True
    enemy.facing_right  = False
    game_start_time = time.time()

    prev_left_attack = prev_right_attack = False
    prev_left_raise  = prev_right_raise  = False
    last_attack_time = last_move_time    = 0.0

    fireballs = []

    v_ok_frames = {"Left": 0, "Right": 0}
    v_active    = {"Left": False, "Right": False}

    ext_mid_ok  = {"Left": False, "Right": False}
    ext_ring_ok = {"Left": False, "Right": False}

    last_both_fire_time = 0.0
    both_armed = True
    both_inactive_frames = 0

    global SIGNALS, EVENT_COUNTS
    SIGNALS = defaultdict(list)
    EVENT_COUNTS = defaultdict(int)

# =========================
# Gesture tuning (two hands, edge-trigger, anti-fist)
# =========================
# V-split window
ANGLE_MIN_IN   = 12.0
ANGLE_MAX_IN   = 60.0
ANGLE_MIN_OUT  = 14.0
ANGLE_MAX_OUT  = 63.0

# Debounce
VFRAMES_REQUIRED = 1

# Optional neighbor guard (kept off)
USE_NEIGHBOR_GAP_GUARD = False
NEIGHBOR_GAP_MAX_RATIO = 0.65

# Extension guard thresholds
EXT_PIP_MIN_DEG   = 150.0
EXT_DIP_MIN_DEG   = 145.0
EXT_LEN_RATIO_MIN = 1.45

# Two-hand latch
FIREBALL_COOLDOWN     = 0.30
FIREBALL_DAMAGE       = 4 #12
BOTH_RELEASE_FRAMES   = 6  # how many frames BOTH must be inactive to re-arm

# =========================
# Init game state
# =========================
player = Fighter(100, GAME_HEIGHT - 50, True, (50,170,255))
enemy  = Fighter(300, GAME_HEIGHT - 50, False,(240, 90, 90))
player.facing_right = True
enemy.facing_right  = False

game_start_time = time.time()
round_number = 1
player_wins = 0
enemy_wins = 0

# Pose triggers
last_attack_time = 0.0
ATTACK_COOLDOWN = 0.35
prev_left_attack = False
prev_right_attack = False

last_move_time = 0.0
MOVE_COOLDOWN = 0.35
prev_left_raise = False
prev_right_raise = False

# Hands / special
fireballs = []
v_ok_frames = {"Left": 0, "Right": 0}
v_active    = {"Left": False, "Right": False}
ext_mid_ok  = {"Left": False, "Right": False}
ext_ring_ok = {"Left": False, "Right": False}
last_min_angle = None

last_both_fire_time = 0.0
both_armed = True
both_inactive_frames = 0

round_over = False
round_end_time = None

# =========================
# Main loop
# =========================
running = True
frame_count = 0
try:
    while running:
        frame_count += 1
        now = time.time()

        # ---- Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    round_number += 1
                    reset_game()
                    round_over = False
                    round_end_time = None
                elif event.key == pygame.K_q:
                    player.move_backward()
                elif event.key == pygame.K_e:
                    player.move_forward()
                elif event.key == pygame.K_a:
                    player.punch_left()
                elif event.key == pygame.K_d:
                    player.punch_right()
                elif event.key == pygame.K_SPACE:
                    player.combo_attack()
                # ---- Audio hotkeys ----
                elif event.key == pygame.K_m:
                    # toggle mute/resume BGM
                    if AUDIO_OK:
                        cur = pygame.mixer.music.get_volume()
                        if cur > 0.01:
                            pygame.mixer.music.set_volume(0.0)
                        else:
                            pygame.mixer.music.set_volume(0.40)
                elif event.key == pygame.K_LEFTBRACKET:   # '['
                    if AUDIO_OK:
                        v = max(0.0, pygame.mixer.music.get_volume() - 0.05)
                        pygame.mixer.music.set_volume(v)
                elif event.key == pygame.K_RIGHTBRACKET:  # ']'
                    if AUDIO_OK:
                        v = min(1.0, pygame.mixer.music.get_volume() + 0.05)
                        pygame.mixer.music.set_volume(v)

        # ---- Camera + Inference
        ret, frame = cap.read()
        angles_info = {}
        raise_info = {}
        last_min_angle = None

        if ret:
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)
            hands_results = hands.process(image_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                lms = results.pose_landmarks.landmark
                angles_info = detect_elbow_angles(lms)
                raise_info  = detect_arm_raises(lms)
            else:
                angles_info = {"left_angle": None, "right_angle": None, "left_attack": False, "right_attack": False, "combo": False}
                raise_info  = {"left_raised": False, "right_raised": False}

            t_rel = now - game_start_time

            lh = angles_info.get("left_angle");  rh = angles_info.get("right_angle")
            left_up  = raise_info.get("left_raised", False)
            right_up = raise_info.get("right_raised", False)

            log_signal("pose.left_elbow_deg",  lh, t_rel)
            log_signal("pose.right_elbow_deg", rh, t_rel)
            log_signal("pose.left_shoulder_raise",  left_up,  t_rel)
            log_signal("pose.right_shoulder_raise", right_up, t_rel)

            # HANDS: V-split w/ extension + hysteresis (two-hands latch handled later)
            if hands_results and hands_results.multi_hand_landmarks:
                labels = []
                if hands_results.multi_handedness:
                    for hd in hands_results.multi_handedness:
                        labels.append(hd.classification[0].label)  # "Left" or "Right"
                else:
                    labels = ["Right"] * len(hands_results.multi_hand_landmarks)

                # temporary states this frame; we'll store only active/inactive
                cur_active = {"Left": v_active["Left"], "Right": v_active["Right"]}

                for i, (hlm2d, label) in enumerate(zip(hands_results.multi_hand_landmarks, labels)):
                    hlm3d = hands_results.multi_hand_world_landmarks[i] if hands_results.multi_hand_world_landmarks else None

                    mp_drawing.draw_landmarks(
                        frame, hlm2d, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(80,255,80), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(80,130,255), thickness=2)
                    )

                    if hlm3d is not None:
                        angle3d, r_idx_mid, r_ring_pky = per_hand_metrics(hlm2d.landmark, hlm3d.landmark)
                        mid_ext  = finger_extended(hlm2d.landmark, hlm3d.landmark, 9, 10, 11, 12,
                                                   EXT_PIP_MIN_DEG, EXT_DIP_MIN_DEG, EXT_LEN_RATIO_MIN)
                        ring_ext = finger_extended(hlm2d.landmark, hlm3d.landmark,13, 14, 15, 16,
                                                   EXT_PIP_MIN_DEG, EXT_DIP_MIN_DEG, EXT_LEN_RATIO_MIN)
                    else:
                        # 2D fallback (less strict)
                        def _vec2d(tip, mcp):
                            return _unit(np.append(_np_xy(hlm2d.landmark, tip) - _np_xy(hlm2d.landmark, mcp), 0.0))
                        v_mid2  = _vec2d(12, 9)
                        v_ring2 = _vec2d(16, 13)
                        angle3d = angle_between_vectors(v_mid2, v_ring2)
                        palm_w = np.linalg.norm(_np_xy(hlm2d.landmark, 5) - _np_xy(hlm2d.landmark, 17)) or 1.0
                        r_idx_mid  = np.linalg.norm(_np_xy(hlm2d.landmark, 8)  - _np_xy(hlm2d.landmark, 12)) / palm_w
                        r_ring_pky = np.linalg.norm(_np_xy(hlm2d.landmark, 16) - _np_xy(hlm2d.landmark, 20)) / palm_w
                        mid_ext  = True
                        ring_ext = True

                    ext_mid_ok[label]  = mid_ext
                    ext_ring_ok[label] = ring_ext

                    if (last_min_angle is None) or (angle3d < last_min_angle):
                        last_min_angle = angle3d

                    neighbor_ok = True
                    if USE_NEIGHBOR_GAP_GUARD:
                        neighbor_ok = (r_idx_mid <= NEIGHBOR_GAP_MAX_RATIO) and (r_ring_pky <= NEIGHBOR_GAP_MAX_RATIO)

                    inside_in  = (ANGLE_MIN_IN  <= angle3d <= ANGLE_MAX_IN)
                    inside_out = (ANGLE_MIN_OUT <= angle3d <= ANGLE_MAX_OUT)
                    ext_ok     = (mid_ext and ring_ext)

                    # hysteresis + debounce
                    if v_active[label]:
                        if not (inside_out and ext_ok and neighbor_ok):
                            cur_active[label] = False
                            v_ok_frames[label] = 0
                        else:
                            cur_active[label] = True
                    else:
                        if inside_in and ext_ok and neighbor_ok:
                            v_ok_frames[label] = min(10, v_ok_frames[label] + 1)
                            if v_ok_frames[label] >= VFRAMES_REQUIRED:
                                cur_active[label] = True
                                v_ok_frames[label] = 0
                        else:
                            v_ok_frames[label] = 0

                # commit states
                v_active["Left"]  = cur_active["Left"]
                v_active["Right"] = cur_active["Right"]

            if hands_results and hands_results.multi_hand_landmarks:
                for i, label in enumerate(labels):
                    hlm2d = hands_results.multi_hand_landmarks[i]
                    hlm3d = hands_results.multi_hand_world_landmarks[i] if hands_results.multi_hand_world_landmarks else None

                    log_signal(f"hands.{label}.mid_ring_angle3d_deg", last_min_angle if label in v_active else angle3d, t_rel)
                    log_signal(f"hands.{label}.r_idx_mid",  r_idx_mid,  t_rel)
                    log_signal(f"hands.{label}.r_ring_pky", r_ring_pky, t_rel)
                    log_signal(f"hands.{label}.mid_extended",  mid_ext,  t_rel)
                    log_signal(f"hands.{label}.ring_extended", ring_ext, t_rel)
                    log_signal(f"hands.{label}.v_active", v_active.get(label, False), t_rel)

                    # finger joint angles logging
                    fj = finger_joint_angles(hlm2d.landmark, hlm3d.landmark if hlm3d else None)
                    for finger_name, joints in fj.items():
                        for joint_name, deg in joints.items():
                            log_signal(f"hands.{label}.{finger_name}.{joint_name}_deg", deg, t_rel)
            # else: keep v_active as-is (we require both inactive to re-arm anyway)

        # ---- TWO-HANDS latch (single shot while held open)
        both_active = v_active["Left"] and v_active["Right"]
        if both_active:
            city.update(now)  # keep windows blinking even while held
            both_inactive_frames = 0
            if both_armed and (now - last_both_fire_time > FIREBALL_COOLDOWN):
                fb_dir = 1 if player.facing_right else -1
                fireballs.append(Fireball(player.x + (25 * fb_dir), player.y - 10, direction=fb_dir))
                if AUDIO_OK: sfx_fireball.play()
                EVENT_COUNTS["fireball"] += 1
                last_both_fire_time = now
                both_armed = False
        else:
            # re-arm only after BOTH are inactive together for a few frames
            if not v_active["Left"] and not v_active["Right"]:
                both_inactive_frames = min(60, both_inactive_frames + 1)
                if both_inactive_frames >= BOTH_RELEASE_FRAMES:
                    both_armed = True
            else:
                both_inactive_frames = 0

        # ---- Movement from shoulder raises
        left_up = raise_info.get("left_raised", False)
        right_up = raise_info.get("right_raised", False)
        if now - last_move_time > MOVE_COOLDOWN:
            if right_up and not prev_right_raise:
                player.move_forward(); last_move_time = now
                EVENT_COUNTS["move_forward"] += 1
            elif left_up and not prev_left_raise:
                player.move_backward(); last_move_time = now
                EVENT_COUNTS["move_backward"] += 1
        prev_left_raise = left_up; prev_right_raise = right_up

        # ---- Attacks from elbow angles
        left_ready = angles_info.get("left_attack", False)
        right_ready = angles_info.get("right_attack", False)
        combo_ready = angles_info.get("combo", False)
        if now - last_attack_time > ATTACK_COOLDOWN:
            if combo_ready and not (prev_left_attack and prev_right_attack):
                if player.combo_attack(): last_attack_time = now
                EVENT_COUNTS["combo_attack"] += 1
            else:
                if left_ready and not prev_left_attack:
                    if player.punch_left(): last_attack_time = now
                    EVENT_COUNTS["punch_left"] += 1
                elif right_ready and not prev_right_attack:
                    if player.punch_right(): last_attack_time = now
                    EVENT_COUNTS["punch_right"] += 1
        prev_left_attack = left_ready; prev_right_attack = right_ready

        # ---- Update fighters & AI
        player.update(); enemy.update()
        player.facing_right = (enemy.x > player.x)
        enemy.facing_right  = (player.x > enemy.x)

        if frame_count % 120 == 0:
            if abs(player.x - enemy.x) > PUNCH_RANGE:
                (enemy.move_forward() if player.x > enemy.x else enemy.move_backward())
            else:
                random.choice([enemy.punch_left, enemy.punch_right, enemy.combo_attack])()

        # ---- Resolve melee
        resolve_hit(player, enemy); resolve_hit(enemy, player)

        # ---- Fireballs
        for fb in fireballs:
            fb.update()
            enemy_rect = pygame.Rect(int(enemy.x - FIGHTER_BASE_W//2), int(enemy.y - FIGHTER_BASE_H),
                                     FIGHTER_BASE_W, FIGHTER_BASE_H)
            if enemy_rect.collidepoint(int(fb.x), int(fb.y - FIGHTER_BASE_H//2)):
                fb.alive = False; enemy.take_damage(FIREBALL_DAMAGE)
        fireballs = [f for f in fireballs if f.alive]

        # ---- Round end
        if not round_over:
            if player.health <= 0:
                enemy_wins += 1; round_over = True; round_end_time = now
                save_round_summary(round_number)
            elif enemy.health <= 0:
                player_wins += 1; round_over = True; round_end_time = now
                save_round_summary(round_number)

        # =========================
        # Draw
        # =========================
        screen.fill(BLACK)

        # Left panel: camera + HUD overlays
        if ret:
            overlay = frame.copy()

            # Shoulder HUD
            cv2.putText(overlay, f"Left shoulder RAISE: {'YES' if left_up else 'no'} (Back)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if left_up else (200,200,200), 2, cv2.LINE_AA)
            cv2.putText(overlay, f"Right shoulder RAISE: {'YES' if right_up else 'no'} (Forward)",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if right_up else (200,200,200), 2, cv2.LINE_AA)

            # Elbow HUD
            lh = angles_info.get("left_angle"); rh = angles_info.get("right_angle")
            base_y = 85
            if lh is not None:
                cv2.putText(overlay, f"Left elbow angle: {lh:5.1f}°", (10, base_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if lh >= 90 else (0,0,255), 2, cv2.LINE_AA)
            else:
                cv2.putText(overlay, "Left elbow angle: --", (10, base_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2, cv2.LINE_AA)
            if rh is not None:
                cv2.putText(overlay, f"Right elbow angle: {rh:5.1f}°", (10, base_y+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if rh >= 90 else (0,0,255), 2, cv2.LINE_AA)
            else:
                cv2.putText(overlay, "Right elbow angle: --", (10, base_y+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2, cv2.LINE_AA)

            # V-split + extension HUD
            if last_min_angle is not None:
                tgt = f"{ANGLE_MIN_IN:.0f}-{ANGLE_MAX_IN:.0f}° (hyst {ANGLE_MIN_OUT:.0f}-{ANGLE_MAX_OUT:.0f}°)"
                ok_color = (180,255,180) if (ANGLE_MIN_IN <= last_min_angle <= ANGLE_MAX_IN) else (200,200,200)
                cv2.putText(overlay, f"Mid–ring min angle (3D): {last_min_angle:4.1f}°  target {tgt}",
                            (10, base_y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ok_color, 2, cv2.LINE_AA)

            cv2.putText(overlay, f"L EXT (mid,ring): {ext_mid_ok.get('Left', False)}, {ext_ring_ok.get('Left', False)}",
                        (10, base_y+122), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0,255,255) if (ext_mid_ok.get('Left', False) and ext_ring_ok.get('Left', False)) else (200,200,200),
                        2, cv2.LINE_AA)
            cv2.putText(overlay, f"R EXT (mid,ring): {ext_mid_ok.get('Right', False)}, {ext_ring_ok.get('Right', False)}",
                        (10, base_y+144), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0,255,255) if (ext_mid_ok.get('Right', False) and ext_ring_ok.get('Right', False)) else (200,200,200),
                        2, cv2.LINE_AA)

            cv2.putText(overlay, f"Both-hands V-SPLIT active: {both_active}   Armed: {both_armed}",
                        (10, base_y+166), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0,255,0) if both_active and both_armed else (0,255,255) if both_active else (200,200,200),
                        2, cv2.LINE_AA)

            frame_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            frame_surface = pygame.transform.scale(frame_surface, (CAMERA_WIDTH, CAMERA_HEIGHT))
            screen.blit(frame_surface, (0, 0))

            # Instructions
            instructions = [
                "MOVEMENT (Shoulder Raises):",
                "Left shoulder RAISE = Move BACKWARD",
                "Right shoulder RAISE = Move FORWARD",
                "ATTACK (Elbow Angles):",
                "Left <90° = Left Attack (-5) | Right <90° = Right Attack (-5)",
                "Both <90° = COMBO (-15)",
                "SPECIAL (two hands required):",
                "Open/split BOTH hands (mid+ring extended) = FIREBALL (-12)",
                f"Fires once while held; re-arms after both hands relax ({BOTH_RELEASE_FRAMES} frames).",
                "Audio: M mute, [ vol-, ] vol+"
            ]
            for i, line in enumerate(instructions):
                color = YELLOW if i in (0, 3, 6) else WHITE
                text = INST_FONT.render(line, True, color)
                screen.blit(text, (10, CAMERA_HEIGHT - 180 + i * 15))

        # Divider
        pygame.draw.line(screen, WHITE, (CAMERA_WIDTH, 0), (CAMERA_WIDTH, SCREEN_HEIGHT), 2)

        # Right panel background (city with blinking lights)
        city.update(now)
        city.draw(screen)

        # Health bars
        bar_width = 150; bar_height = 20
        pr = player.health / player.max_health; er = enemy.health / enemy.max_health
        pbar = pygame.Rect(CAMERA_WIDTH + 20, 20, bar_width, bar_height)
        ebar = pygame.Rect(CAMERA_WIDTH + GAME_WIDTH - bar_width - 20, 20, bar_width, bar_height)
        pygame.draw.rect(screen, RED, pbar)
        pygame.draw.rect(screen, GREEN, (pbar.x, pbar.y, int(bar_width * pr), bar_height))
        pygame.draw.rect(screen, WHITE, pbar, 2)
        pygame.draw.rect(screen, RED, ebar)
        pygame.draw.rect(screen, GREEN, (ebar.x, ebar.y, int(bar_width * er), bar_height))
        pygame.draw.rect(screen, WHITE, ebar, 2)
        screen.blit(SMALL_FONT.render(f"PLAYER: {player.health}%", True, WHITE), (CAMERA_WIDTH + 20, 45))
        screen.blit(SMALL_FONT.render(f"ENEMY: {enemy.health}%", True, WHITE), (CAMERA_WIDTH + GAME_WIDTH - 120, 45))

        # Round text
        screen.blit(FONT.render(f"ROUND {round_number}", True, YELLOW), (CAMERA_WIDTH + GAME_WIDTH // 2 - 50, 70))
        screen.blit(SMALL_FONT.render(f"Player: {player_wins} | Enemy: {enemy_wins}", True, WHITE),
                    (CAMERA_WIDTH + GAME_WIDTH // 2 - 60, 95))

        # Fighters
        player.draw(screen, CAMERA_WIDTH)
        enemy.draw(screen, CAMERA_WIDTH)

        # Fireballs
        for fb in fireballs: fb.draw(screen, CAMERA_WIDTH)

        # Round overlay
        if round_over:
            overlay = pygame.Surface((GAME_WIDTH, GAME_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            screen.blit(overlay, (CAMERA_WIDTH, 0))
            result_text = "ENEMY WINS!" if player.health <= 0 else "PLAYER WINS!"
            screen.blit(FONT.render(result_text, True, RED if player.health <= 0 else GREEN),
                        (CAMERA_WIDTH + GAME_WIDTH // 2 - 80, GAME_HEIGHT // 2))
            screen.blit(SMALL_FONT.render("Next round in 3s (press R to skip)", True, WHITE),
                        (CAMERA_WIDTH + GAME_WIDTH // 2 - 120, GAME_HEIGHT // 2 + 30))

        pygame.display.flip()
        clock.tick(FPS)

        # Auto next round
        if round_over and (now - round_end_time > 3):
            round_number += 1; reset_game(); round_over = False; round_end_time = None

except Exception as e:
    print(f"Error in main loop: {e}")
    import traceback; traceback.print_exc()
finally:
    print("Cleaning up...")
    cap.release(); pose.close(); hands.close()
    pygame.quit(); cv2.destroyAllWindows()
    print("Game ended!")
