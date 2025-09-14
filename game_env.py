# -*- coding: utf-8 -*-
"""
game_env.py
-----------
Arena Blasters — Wide Arena Edition (Gymnasium + Pygame, single file)
- Large scrolling world with platforms, updrafts (vertical wind), teleporters
- Weapons with re-balanced stats: pistol/SMG/shotgun/rifle/sniper/rocket/laser(beam)
- No-inertia controls, double-jump, dash, drop-through thin platforms
- Pick up / throw with RMB (in human mode)
- Destructible solids are stubbed off by default to keep training stable (can be re-enabled)
- Anti-stuck: physical separation + reward shaping penalty

Dual mode:
1) Import as Gymnasium environment:  env = GameEnv(render_mode=None|'rgb_array'|'human')
2) Run as a human-playable game:     python game_env.py   (uses GameEnv(render_mode='human'))

Requirements satisfied:
- Single file
- Class name is GameEnv
- __main__ starts human mode
- Pure Python physics/collision (no external physics libs)
- Clean hyper-casual visuals, 800x600 window for human play
"""

import os, math, random, time, sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

# -------------------- Optional headless setup for RL --------------------
# Use dummy driver unless human render explicitly requested.
if __name__ != "__main__" and os.environ.get("SDL_VIDEODRIVER") is None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import numpy as np

# Gymnasium is expected for training; for pure human run it's optional but recommended.
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # Minimal fallback for human-only run
    class _Dummy(gym := object):  # type: ignore
        class Env: ...
        class spaces: ...
    print("[WARN] gymnasium not found. Human mode will work; RL import will require gymnasium.", file=sys.stderr)

# -------------------- Window / World --------------------
WIN_W, WIN_H = 800, 600                  # human window size (hyper-casual)
SCALE = 2                                 # internal render scale (AA look)
RW, RH = WIN_W * SCALE, WIN_H * SCALE     # render target
FPS = 60

# Large world (scrolling camera)
WORLD_W, WORLD_H = 2400, 1400

# -------------------- Palette --------------------
HEX = lambda h: tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
BG       = HEX("F0F4F8")
INK      = HEX("222831")
MUTED    = HEX("A0AEC0")
ACCENT1  = HEX("FF6B6B")  # Bot
ACCENT2  = HEX("4ECDC4")  # You
SOLID_COL = (200, 206, 216)
THIN_COL  = (184, 192, 204)
LASER_COL = HEX("35D0FF")
PORTAL_IN = HEX("A66BFF")
PORTAL_OUT= HEX("57E2E5")
WIND_COL  = HEX("C8F7FF")
BREAK_COL = HEX("B0BAC8")

# -------------------- Utility --------------------
def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v
def lerp(a, b, t): return a + (b - a) * t
def draw_rrect(surf, rect, color, r): pygame.draw.rect(surf, color, rect, border_radius=r)

def seg_point_dist(ax, ay, bx, by, px, py):
    abx, aby = bx-ax, by-ay
    apx, apy = px-ax, py-ay
    ab2 = abx*abx + aby*aby + 1e-6
    t = clamp((apx*abx + apy*aby)/ab2, 0.0, 1.0)
    cx, cy = ax + abx*t, ay + aby*t
    dx, dy = px - cx, py - cy
    return math.hypot(dx, dy)

# -------------------- Physics --------------------
GRAVITY = 2200.0
MAX_FALL = 2400.0

TILE_SOLID = 0
TILE_THIN  = 1

@dataclass
class Rect:
    x: float; y: float; w: float; h: float
    @property
    def left(self):   return self.x
    @property
    def right(self):  return self.x + self.w
    @property
    def top(self):    return self.y
    @property
    def bottom(self): return self.y + self.h
    def to_pg(self, cam=(0,0)):
        cx, cy = cam
        return pygame.Rect(int(self.x - cx), int(self.y - cy), int(self.w), int(self.h))
    def intersects(self, other:'Rect'):
        return not (self.right <= other.left or self.left >= other.right or
                    self.bottom <= other.top or self.top >= other.bottom)

# -------------------- Particles --------------------
class Particle:
    def __init__(self, pos, vel, radius, color, life):
        self.x, self.y = pos; self.vx, self.vy = vel
        self.r = radius; self.color = color
        self.life = life; self.age = 0.0
    def update(self, dt):
        self.age += dt
        self.x += self.vx * dt; self.y += self.vy * dt
        self.vy += GRAVITY*0.15 * dt
        self.vx *= 0.985; self.vy *= 0.985
        return self.age < self.life
    def draw(self, surf, cam):
        t = 1.0 - (self.age / self.life)
        c = tuple(int(lerp(self.color[i], BG[i], 1.0 - t)) for i in range(3))
        pygame.draw.circle(surf, c, (int(self.x - cam[0]), int(self.y - cam[1])), max(1, int(self.r * t)))

# -------------------- Weapons --------------------
@dataclass
class GunDef:
    name: str
    color: Tuple[int,int,int]
    dmg: int
    cooldown: float
    speed: float
    recoil: float
    ammo: int
    special: str = ""   # "spread", "burst3", "rocket", "sniper", "beam"
    pellet: int = 1
    spread_deg: float = 0.0

GUNS: List[GunDef] = [
    GunDef("Pistol",  HEX("3FA7D6"), dmg=18, cooldown=0.25, speed=1000, recoil=0, ammo=18),
    GunDef("SMG",     HEX("2BB673"), dmg=9,  cooldown=0.09, speed=950,  recoil=0, ammo=32, special="burst3", spread_deg=5),
    GunDef("Shotgun", HEX("9C27B0"), dmg=24, cooldown=0.55, speed=820,  recoil=0, ammo=16, pellet=7, spread_deg=10, special="spread"),
    GunDef("Rifle",   HEX("00ADB5"), dmg=22, cooldown=0.22, speed=1150, recoil=0, ammo=28),
    GunDef("Sniper",  HEX("FF8F6B"), dmg=50, cooldown=0.85, speed=1700, recoil=0, ammo=6,  special="sniper"),
    GunDef("Rocket",  HEX("FD5E53"), dmg=55, cooldown=0.95, speed=720,  recoil=0, ammo=7,  special="rocket"),
    GunDef("Laser",   LASER_COL,     dmg=70, cooldown=0.95, speed=0,    recoil=0, ammo=3,  special="beam"),
]

class GunEntity:
    """Pickup/throwable gun instance with ammo."""
    def __init__(self, gdef:GunDef, pos:Tuple[float,float], ammo_override:int=None, spawned=True):
        self.gdef = gdef
        self.x, self.y = pos
        self.vx, self.vy = 0.0, 0.0
        self.state = "ground"   # "ground", "tossed", "equipped"
        self.owner = None
        self.radius = 16
        self.ammo = gdef.ammo if ammo_override is None else ammo_override
        self.spawned_pickup = spawned

    def aabb(self): return Rect(self.x-self.radius, self.y-self.radius, self.radius*2, self.radius*2)

    def toss(self, owner, dir_vec, power):
        self.owner = owner
        self.state = "tossed"
        self.vx, self.vy = dir_vec[0]*power, dir_vec[1]*power

    def update(self, dt, level_rects:List[Tuple[Rect,int]]) -> bool:
        """Returns False if should be removed (after toss impacts or out)."""
        if self.state == "ground":
            return True
        self.vy += GRAVITY * 0.9 * dt
        self.vx *= 0.995
        self.vy = min(self.vy, MAX_FALL)

        self.x += self.vx * dt
        wr = self.aabb()
        for r,t in level_rects:
            if wr.intersects(r) and (t == TILE_SOLID or t == TILE_THIN):
                return False
        self.y += self.vy * dt
        wr = self.aabb()
        for r,t in level_rects:
            if t == TILE_THIN:
                if self.vy >= 0 and wr.bottom > r.top and wr.bottom - self.vy*dt <= r.top+2 and wr.right > r.left and wr.left < r.right:
                    return False
            else:
                if wr.intersects(r): return False

        if self.y > WORLD_H + 120 or self.x < -200 or self.x > WORLD_W+200: return False
        return True

    def draw(self, surf, cam):
        pygame.draw.circle(surf, self.gdef.color, (int(self.x - cam[0]), int(self.y - cam[1])), self.radius)
        ring = tuple(int(lerp(self.gdef.color[i], BG[i], 0.6)) for i in range(3))
        pygame.draw.circle(surf, ring, (int(self.x - cam[0]), int(self.y - cam[1])), max(2, self.radius-6))
        pygame.draw.rect(surf, self.gdef.color,
                         pygame.Rect(int(self.x-12 - cam[0]), int(self.y-7 - cam[1]), 24, 14), border_radius=4)
        pygame.draw.rect(surf, INK, pygame.Rect(int(self.x+10 - cam[0]), int(self.y-3 - cam[1]), 10, 4), border_radius=2)

# -------------------- Projectiles --------------------
class Bullet:
    def __init__(self, x, y, vx, vy, dmg, owner, life=0.9, radius=6,
                 explosive=False, pierce_thin=False, proximity:float=0.0, seek:float=0.0):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.dmg = dmg; self.owner = owner
        self.life = life; self.age = 0.0
        self.r = radius
        self.explosive = explosive
        self.pierce_thin = pierce_thin
        self.proximity = float(proximity)
        self.seek = float(seek)
        self._spd = math.hypot(vx, vy) or 1.0

    def aabb(self): return Rect(self.x-self.r, self.y-self.r, self.r*2, self.r*2)

    def update(self, dt, level_rects, players, particles, breakables) -> bool:
        self.age += dt

        # Light homing for sniper
        if self.seek > 0.0:
            best = None; bd2 = 1e12
            for pl in players:
                if not pl.alive or pl is self.owner: continue
                dx, dy = pl.x - self.x, (pl.y - pl.h*0.5) - self.y
                d2 = dx*dx + dy*dy
                if d2 < bd2: bd2 = d2; best = (dx, dy)
            if best:
                bx, by = best
                L = math.hypot(bx, by) or 1.0
                tx, ty = bx/L, by/L
                ax, ay = tx*self._spd, ty*self._spd
                k = clamp(self.seek*dt, 0.0, 0.25)
                self.vx = lerp(self.vx, ax, k)
                self.vy = lerp(self.vy, ay, k)
                sp = math.hypot(self.vx, self.vy) or 1.0
                self.vx *= (self._spd / sp); self.vy *= (self._spd / sp)

        # Move
        self.x += self.vx * dt; self.y += self.vy * dt
        rect = self.aabb()

        # Proximity detonation (rocket)
        if self.explosive and self.proximity > 0.0:
            R2 = self.proximity * self.proximity
            for pl in players:
                if not pl.alive or pl is self.owner: continue
                dx, dy = pl.x - self.x, (pl.y - pl.h*0.5) - self.y
                if dx*dx + dy*dy <= R2:
                    self.explode(particles, players, breakables)
                    return False

        # Level collision
        for r,t in level_rects:
            if t == TILE_THIN and self.pierce_thin:
                continue
            if t == TILE_THIN:
                if rect.bottom > r.top and rect.top < r.top and rect.right > r.left and rect.left < r.right and self.vy > 0:
                    self.explode(particles, players, breakables); return False
            else:
                if rect.intersects(r):
                    self.explode(particles, players, breakables); return False

        # Player collision
        for pl in players:
            if not pl.alive or pl is self.owner: continue
            if rect.intersects(pl.aabb()):
                pl.damage(self.dmg, (math.copysign(200, self.vx), -120), attacker=self.owner, particles=particles)
                self.explode(particles, players, breakables, center=(pl.x, pl.y - pl.h*0.5))
                return False
        return self.age < self.life

    def explode(self, particles, players, breakables, center=None):
        if not self.explosive: return
        cx, cy = (self.x, self.y) if center is None else center
        for _ in range(12):
            ang = random.uniform(0, math.tau); spd = random.uniform(160, 360)
            particles.append(Particle((cx, cy), (math.cos(ang)*spd, math.sin(ang)*spd-80), 5, ACCENT1, 0.35))
        for pl in players:
            if not pl.alive: continue
            dx, dy = pl.x - cx, (pl.y - pl.h*0.5) - cy
            if dx*dx + dy*dy < (180*180):
                kx = 240 * (1 if dx>=0 else -1)
                pl.damage(18, (kx, -160), attacker=self.owner, particles=particles)

    def draw(self, surf, cam):
        pygame.draw.circle(surf, INK, (int(self.x - cam[0]), int(self.y - cam[1])), self.r)

# -------------------- Beam (Laser) --------------------
class Beam:
    """Charged, short-lived thick beam. Passes thin platforms, blocked by solid."""
    def __init__(self, owner, start, angle, level_rects, color=LASER_COL, warmup=0.35, active=0.16, width=38):
        self.owner = owner
        self.sx, self.sy = start
        self.ang = angle
        self.color = color
        self.warmup = warmup
        self.active_time = active
        self.width = width
        self.age = 0.0
        self.hit_once: Set[object] = set()
        c, s = math.cos(self.ang), math.sin(self.ang)
        max_len = WORLD_W*0.9
        step = 14.0
        x, y = self.sx, self.sy
        for _ in range(int(max_len//step)):
            nx, ny = x + c*step, y + s*step
            seg = Rect(min(x,nx), min(y,ny), abs(nx-x)+1, abs(ny-y)+1)
            blocked = False
            for r,t in level_rects:
                if t == TILE_SOLID and seg.intersects(r):
                    blocked = True; break
            x, y = nx, ny
            if blocked: break
        self.ex, self.ey = x, y
    @property
    def active(self): return self.age >= self.warmup and self.age < (self.warmup + self.active_time)
    def update(self, dt, players, particles, breakables):
        self.age += dt
        if self.active:
            for pl in players:
                if not pl.alive or pl is self.owner: continue
                d = seg_point_dist(self.sx, self.sy, self.ex, self.ey, pl.x, pl.y - pl.h*0.5)
                if d <= (self.width*0.5 + 18) and pl not in self.hit_once:
                    dmg = self.owner.holding.gdef.dmg if self.owner.holding else 40
                    pl.damage(dmg, (math.cos(self.ang)*80, -60), attacker=self.owner, particles=particles)
                    self.hit_once.add(pl)
        return self.age < (self.warmup + self.active_time + 0.12)
    def draw(self, surf, cam):
        sx, sy = int(self.sx - cam[0]), int(self.sy - cam[1])
        ex, ey = int(self.ex - cam[0]), int(self.ey - cam[1])
        if self.age < self.warmup:
            t = clamp(self.age/self.warmup, 0, 1)
            col = tuple(int(lerp(BG[i], self.color[i], t*0.8)) for i in range(3))
            pygame.draw.line(surf, col, (sx, sy), (ex, ey), 2)
        else:
            w = int(self.width)
            pygame.draw.line(surf, (240, 248, 255), (sx, sy), (ex, ey), w+6)
            pygame.draw.line(surf, self.color, (sx, sy), (ex, ey), w)

# -------------------- Player --------------------
class Player:
    def __init__(self, x, y, color, name="P1", is_bot=False, game=None):
        self.x, self.y = x, y
        self.w, self.h = 36, 54
        self.vx = self.vy = 0.0
        self.on_ground = False
        self.drop_through_timer = 0.0
        self.coyote = 0.0
        self.jump_buf = 0.0
        self.face = 1
        self.color = color; self.name = name; self.is_bot = is_bot

        self.hp = 100; self.alive = True
        self.kos = 0; self.deaths = 0

        self.holding: Optional[GunEntity] = None
        self.attack_cool = 0.0

        self.invuln = 0.0; self.hitlag = 0.0
        self.last_aim = (x+100, y)

        self.dash_cd = 0.0; self.dash_time = 0.0; self.dashing = False
        self.max_air_jumps = 1; self.air_jumps = 1
        self.tp_cd = 0.0

        self.game = game

    def aabb(self): return Rect(self.x - self.w/2, self.y - self.h, self.w, self.h)
    def feet(self): return (self.x, self.y)

    def damage(self, amount, knock=(0,0), attacker=None, particles=None):
        if self.invuln > 0.0: return
        self.hp -= int(amount)
        self.vx += knock[0]; self.vy += knock[1]
        self.invuln = 0.35; self.hitlag = 0.05
        if self.hp <= 0:
            self.alive = False; self.deaths += 1
            if attacker and attacker is not self: attacker.kos += 1
        if particles:
            for _ in range(8):
                ang = random.uniform(0, math.tau); spd = random.uniform(80, 240)
                particles.append(Particle((self.x, self.y - self.h*0.6), (math.cos(ang)*spd, math.sin(ang)*spd-200), 6, ACCENT1, 0.5))

    def respawn(self, pos):
        self.x, self.y = pos; self.vx = self.vy = 0.0
        self.hp = 100; self.alive = True; self.invuln = 0.8
        self.holding = None; self.attack_cool = 0.0
        self.air_jumps = self.max_air_jumps; self.tp_cd = 0.0

    def update(self, dt, inputs, level_rects, bullets, guns, players, particles, beams, windzones, breakables, teleports):
        self.attack_cool = max(0.0, self.attack_cool - dt)
        self.coyote = max(0.0, self.coyote - dt)
        self.jump_buf = max(0.0, self.jump_buf - dt)
        self.drop_through_timer = max(0.0, self.drop_through_timer - dt)
        self.invuln = max(0.0, self.invuln - dt)
        self.hitlag = max(0.0, self.hitlag - dt)
        self.dash_cd = max(0.0, self.dash_cd - dt)
        self.tp_cd = max(0.0, self.tp_cd - dt)
        if self.dash_time > 0.0: self.dash_time = max(0.0, self.dash_time - dt)
        self.dashing = self.dash_time > 0.0
        if not self.alive: return

        left = inputs.get("left", False); right = inputs.get("right", False)
        down = inputs.get("down", False)
        jump_pressed = inputs.get("jump_pressed", False)
        attack = inputs.get("attack", False)
        rmb_release = inputs.get("rmb_release", False)
        dash_pressed = inputs.get("dash", False)
        aim = inputs.get("aim", (self.x+self.face*100, self.y-40)); mx, my = aim
        self.last_aim = aim

        # Movement (no inertia)
        speed = 440.0
        dirx = (-1.0 if left and not right else 1.0 if right and not left else 0.0)
        if dash_pressed and self.dash_cd <= 0.0:
            dash_dir = -1.0 if dirx < 0 else 1.0 if dirx > 0 else (1.0 if mx>self.x else -1.0)
            self.vx = dash_dir * 950.0; self.vy *= 0.5
            self.dash_time = 0.13; self.dashing = True; self.dash_cd = 0.6
            for _ in range(8): particles.append(Particle(self.feet(), (random.uniform(-260,260), random.uniform(-160, -40)), 4, MUTED, 0.25))
        if not self.dashing: self.vx = dirx * speed

        # Jump buffer + double-jump
        if jump_pressed: self.jump_buf = 0.12
        if self.jump_buf > 0:
            if (self.on_ground or self.coyote>0):
                self.vy = -1000.0; self.on_ground=False; self.jump_buf=0.0; self.air_jumps = self.max_air_jumps
                for _ in range(6): particles.append(Particle(self.feet(), (random.uniform(-120,120), random.uniform(-420,-200)), 5, ACCENT2, 0.4))
            elif self.air_jumps > 0:
                self.vy = -940.0; self.air_jumps -= 1; self.jump_buf=0.0
                for _ in range(5): particles.append(Particle(self.feet(), (random.uniform(-100,100), random.uniform(-380,-160)), 5, ACCENT2, 0.35))

        if down and self.on_ground: self.drop_through_timer = 0.20

        # Gravity + updrafts
        self.vy += GRAVITY * dt
        for wz in windzones:
            r = wz["rect"]
            if (self.x > r.left and self.x < r.right and self.y - self.h*0.5 > r.top and self.y - self.h*0.5 < r.bottom):
                self.vy -= wz["strength"] * dt
        self.vy = min(self.vy, MAX_FALL)

        if self.hitlag > 0.0: return

        # Horizontal
        self.x += self.vx * dt
        rect = self.aabb()
        for r,t in level_rects:
            if t == TILE_THIN: continue
            if rect.intersects(r):
                if self.vx > 0: self.x = r.left - self.w/2
                elif self.vx < 0: self.x = r.right + self.w/2
                self.vx = 0; rect = self.aabb()
        # Vertical
        self.on_ground = False
        self.y += self.vy * dt
        rect = self.aabb()
        for r,t in level_rects:
            if t == TILE_THIN:
                if self.vy >= 0 and self.drop_through_timer <= 0:
                    if rect.bottom > r.top and rect.bottom - self.vy*dt <= r.top + 2 and rect.right > r.left and rect.left < r.right:
                        self.y = r.top; self.vy = 0; self.on_ground = True; self.coyote = 0.08; rect = self.aabb(); self.air_jumps = self.max_air_jumps
            else:
                if rect.intersects(r):
                    if self.vy > 0: self.y = r.top; self.on_ground = True; self.coyote = 0.08; self.air_jumps = self.max_air_jumps
                    else: self.y = r.bottom + self.h
                    self.vy = 0; rect = self.aabb()

        self.face = 1 if mx > self.x else -1

        # Teleporters
        if self.tp_cd <= 0.0:
            for tp in teleports:
                if self.aabb().intersects(tp["rect"]):
                    self.x, self.y = tp["exit"]
                    self.vx *= 0.2; self.vy *= 0.2
                    self.tp_cd = 0.5
                    break

        # RMB: pickup or throw (human)
        if rmb_release:
            picked = False
            if self.holding is None:
                for g in guns:
                    if g.state == "ground":
                        if (abs(g.x - self.x) < 56 and abs(g.y - (self.y - self.h)) < 84):
                            guns.remove(g); g.owner = self; g.state = "equipped"; self.holding = g; picked = True; break
            if not picked and self.holding is not None:
                dx, dy = (mx - self.x), (my - (self.y - self.h*0.6)); d = math.hypot(dx, dy) + 1e-6
                dirv = (dx/d, dy/d)
                ent = self.holding; ent.x, ent.y = self.x + dirv[0]*24, (self.y - self.h*0.6) + dirv[1]*24
                ent.toss(self, dirv, 820.0); ent.spawned_pickup = False; self.holding = None; guns.append(ent)

        # LMB Fire
        if attack and self.attack_cool <= 0.0 and self.holding is not None:
            g = self.holding.gdef
            angle = math.atan2(my - (self.y - self.h*0.6), mx - self.x)

            def fire_one(spread=0.0, speed=g.speed, dmg=g.dmg, life=0.9, radius=6,
                         explosive=False, pierce_thin=False, proximity:float=0.0, seek:float=0.0):
                ang = angle + spread
                vx, vy = math.cos(ang)*speed, math.sin(ang)*speed
                bullets.append(Bullet(self.x, self.y - self.h*0.6, vx, vy, dmg, self,
                                      life=life, radius=radius,
                                      explosive=explosive, pierce_thin=pierce_thin,
                                      proximity=proximity, seek=seek))

            shots_fired = 0
            if g.special == "spread":
                for _ in range(g.pellet):
                    offs = math.radians(random.uniform(-g.spread_deg, g.spread_deg))
                    fire_one(spread=offs, speed=g.speed*0.9, dmg=g.dmg, life=0.6, radius=5); shots_fired += 1
            elif g.special == "burst3":
                for _ in range(3):
                    offs = math.radians(random.uniform(-g.spread_deg, g.spread_deg))
                    fire_one(spread=offs, speed=g.speed*0.95, dmg=g.dmg, life=0.6, radius=5); shots_fired += 1
            elif g.special == "sniper":
                fire_one(life=1.1, radius=6, pierce_thin=True, seek=0.9); shots_fired = 1
            elif g.special == "rocket":
                fire_one(speed=g.speed, dmg=g.dmg, life=1.2, radius=7, explosive=True, proximity=160.0); shots_fired = 1
            elif g.special == "beam":
                beams.append(Beam(self, (self.x, self.y - self.h*0.6), angle, level_rects,
                                  color=LASER_COL, warmup=0.35, active=0.16, width=38))
                shots_fired = 1
                self.attack_cool = g.cooldown
            else:
                fire_one(); shots_fired = 1

            if g.special != "beam":
                self.attack_cool = g.cooldown

            self.holding.ammo -= shots_fired
            if self.holding.ammo <= 0:
                for _ in range(10):
                    ang = random.uniform(0, math.tau); spd = random.uniform(120, 320)
                    particles.append(Particle((self.x, self.y - self.h*0.6),
                                              (math.cos(ang)*spd, math.sin(ang)*spd-80), 5, MUTED, 0.35))
                self.holding = None

        # Ring-out
        if self.y > WORLD_H - 4: self.damage(999, (0,0))

    def draw_gun_in_hand(self, surf, cam):
        if not self.holding: return
        gdef = self.holding.gdef
        hx, hy = (self.x + self.face*14, self.y - self.h*0.65)
        mx, my = self.last_aim
        ang = math.atan2(my - hy, mx - hx)
        c, s = math.cos(ang), math.sin(ang)
        def rect_poly(cx, cy, length, thickness):
            dx, dy = c, s; nx, ny = -dy, dx; L = length; T = thickness/2
            p1 = (cx, cy); p2 = (cx + dx*L, cy + dy*L)
            return [(p1[0] + nx*T, p1[1] + ny*T),
                    (p2[0] + nx*T, p2[1] + ny*T),
                    (p2[0] - nx*T, p2[1] - ny*T),
                    (p1[0] - nx*T, p1[1] - ny*T)]
        def poly(points, color):
            pygame.draw.polygon(surf, color, [(int(x - cam[0]), int(y - cam[1])) for x,y in points])
        body_len, body_th = 26, 12; barrel_len, barrel_th = 14, 5
        if gdef.name == "Shotgun": body_len, body_th = 32, 14; barrel_len, barrel_th = 18, 6
        elif gdef.name == "SMG": body_len, body_th = 24, 10; barrel_len, barrel_th = 12, 4
        elif gdef.name == "Rifle": body_len, body_th = 30, 12; barrel_len, barrel_th = 16, 5
        elif gdef.name == "Sniper": body_len, body_th = 36, 12; barrel_len, barrel_th = 24, 5
        elif gdef.name == "Rocket": body_len, body_th = 34, 14; barrel_len, barrel_th = 20, 8
        elif gdef.name == "Laser": body_len, body_th = 28, 10; barrel_len, barrel_th = 18, 4
        body = rect_poly(hx, hy, body_len, body_th)
        barrel = rect_poly(hx + (body_len-4)*c, hy + (body_len-4)*s, barrel_len, barrel_th)
        poly(body, gdef.color); poly(barrel, INK)

    def draw(self, surf, cam):
        r = pygame.Rect(int(self.x - self.w/2 - cam[0]), int(self.y - self.h - cam[1]), int(self.w), int(self.h))
        draw_rrect(surf, r, self.color, 12)
        visor = pygame.Rect(r.x+6, r.y+10, r.w-12, 8); draw_rrect(surf, visor, BG, 6)
        # HP
        hpw = int(self.w * clamp(self.hp/100.0, 0.0, 1.0))
        bar = pygame.Rect(r.x, r.y-10, int(self.w), 6); draw_rrect(surf, bar, MUTED, 3)
        draw_rrect(surf, pygame.Rect(r.x, r.y-10, hpw, 6), ACCENT2 if self.hp>35 else ACCENT1, 3)
        self.draw_gun_in_hand(surf, cam)

# -------------------- Level --------------------
def build_level():
    rects: List[Tuple[Rect,int]] = []
    # Main floor long
    rects.append((Rect(60, WORLD_H*0.86, WORLD_W-120, 40), TILE_SOLID))
    # Side walls
    rects.append((Rect(0, 0, 40, WORLD_H), TILE_SOLID))
    rects.append((Rect(WORLD_W-40, 0, 40, WORLD_H), TILE_SOLID))

    # Mid/upper platforms across the map
    seg = WORLD_W/5
    rects.append((Rect(seg*0.6, WORLD_H*0.70, seg*1.0, 18), TILE_THIN))
    rects.append((Rect(seg*2.0, WORLD_H*0.62, seg*1.0, 18), TILE_SOLID))  # central SOLID (can be made breakable)
    rects.append((Rect(seg*3.2, WORLD_H*0.54, seg*1.0, 18), TILE_THIN))
    rects.append((Rect(seg*1.0, WORLD_H*0.44, seg*0.9, 18), TILE_THIN))
    rects.append((Rect(seg*2.6, WORLD_H*0.36, seg*0.9, 18), TILE_THIN))
    rects.append((Rect(seg*1.6, WORLD_H*0.26, seg*0.8, 18), TILE_THIN))
    rects.append((Rect(seg*3.6, WORLD_H*0.22, seg*0.8, 18), TILE_THIN))
    return rects

def draw_level(surf, level_rects, cam, breakables, windzones, teleports):
    for r, t in level_rects:
        col = THIN_COL if t==TILE_THIN else SOLID_COL
        draw_rrect(surf, r.to_pg(cam), col, 12)

    for b in breakables:
        if b["down"]: continue
        rr = b["rect"].to_pg(cam)
        draw_rrect(surf, rr, BREAK_COL, 10)

    for wz in windzones:
        rr = wz["rect"].to_pg(cam)
        overlay = pygame.Surface((rr.w, rr.h), pygame.SRCALPHA)
        overlay.fill((*WIND_COL, 60))
        surf.blit(overlay, (rr.x, rr.y))
        pygame.draw.rect(surf, WIND_COL, rr, 2, border_radius=10)
        for x in range(rr.x, rr.right, 16):
            pygame.draw.line(surf, (220,240,255), (x, rr.bottom), (x, rr.top), 1)

    for tp in teleports:
        rr = tp["rect"].to_pg(cam)
        pygame.draw.ellipse(surf, PORTAL_IN, rr, 3)
        ex, ey = tp["exit"]
        pygame.draw.circle(surf, PORTAL_OUT, (int(ex - cam[0]), int(ey - cam[1])), 8, 3)

# -------------------- Wind / Breakables / Portals --------------------
def build_windzones():
    floor_y = WORLD_H * 0.86
    top_margin = 120
    height = floor_y - top_margin
    return [
        {"rect": Rect(WORLD_W*0.35, floor_y-180, 120, 180), "strength": 1800.0},
        {"rect": Rect(WORLD_W*0.62, floor_y-220, 120, 220), "strength": 1600.0},
        {"rect": Rect(80, top_margin, 140, height), "strength": 2300.0},
        {"rect": Rect(WORLD_W-220, top_margin, 140, height), "strength": 2300.0},
    ]

def build_breakables():
    # Disabled for training stability. Keep empty list by default.
    return []

def build_teleports():
    a = Rect(WORLD_W*0.15, WORLD_H*0.40-24, 36, 60)
    b = Rect(WORLD_W*0.85-36, WORLD_H*0.32-24, 36, 60)
    floor_y = WORLD_H * 0.86
    c = Rect(WORLD_W*0.50-18, floor_y-64, 36, 60)
    d = Rect(WORLD_W*0.50-18, WORLD_H*0.22-60, 36, 60)
    return [
        {"rect": a, "exit": (b.x + b.w + 20, b.y + b.h)},
        {"rect": b, "exit": (a.x - 20, a.y + a.h)},
        {"rect": c, "exit": (d.x + d.w + 20, d.y + d.h)},
        {"rect": d, "exit": (c.x - 20, c.y + c.h)},
    ]

# -------------------- Spawners --------------------
SPAWN_INTERVAL_START = 4.0
SPAWN_MIN = 0.9
MAX_GUNS = 12

def spawnable_surfaces(level_rects):
    plats = []
    for r,t in level_rects:
        if t == TILE_THIN:
            plats.append(r)
        elif t == TILE_SOLID:
            if r.w > 200 and r.h <= 60 and r.x >= 0 and r.right <= WORLD_W:
                plats.append(r)
    return plats

# -------------------- Bot AI --------------------
class BotBrain:
    def __init__(self): self.jump_cd = 0.0
    def think(self, bot:'Player', players, guns, dt):
        self.jump_cd = max(0.0, self.jump_cd - dt)
        enemies = [p for p in players if p is not bot and p.alive]
        target = min(enemies, key=lambda p: (p.x-bot.x)**2+(p.y-bot.y)**2) if enemies else None
        inputs = {"left":False,"right":False,"down":False,"jump_pressed":False,
                  "attack":False,"rmb_release":False,
                  "aim":(bot.x+bot.face*200, bot.y-40), "dash":False}
        if target: inputs["aim"] = (target.x, target.y-40)
        if bot.holding is None:
            cands = [g for g in guns if g.state=="ground"]
            if cands:
                near = min(cands, key=lambda g: (g.x-bot.x)**2+(g.y-bot.y)**2)
                inputs["left"] = near.x < bot.x; inputs["right"] = near.x > bot.x
                if self.jump_cd<=0.0 and near.y < bot.y-40 and random.random()<0.6:
                    inputs["jump_pressed"]=True; self.jump_cd=0.25
                if abs(near.x-bot.x)<56 and abs(near.y-(bot.y-bot.h))<84:
                    inputs["rmb_release"]=True
            else:
                if random.random()<0.02: inputs["jump_pressed"]=True
            return inputs
        if target:
            dx = target.x - bot.x
            if abs(dx) > 100:
                inputs["left"] = dx < 0; inputs["right"] = dx > 0
            if abs(target.y - bot.y) > 50 and self.jump_cd<=0.0 and random.random()<0.25:
                inputs["jump_pressed"]=True; self.jump_cd=0.35
            if random.random()<0.35: inputs["attack"]=True
        return inputs

# -------------------- Game (core engine used by both human + env) --------------------
class Game:
    def __init__(self, window_for_human:bool=True):
        pygame.init()
        pygame.display.set_caption("Arena Blasters — Wide Arena Edition")
        self.window_for_human = window_for_human
        if window_for_human:
            self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        else:
            # hidden surface for offscreen rendering
            self.screen = pygame.Surface((WIN_W, WIN_H))
        self.scene = pygame.Surface((RW, RH), pygame.SRCALPHA)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 22)

        self.level = build_level()                # static platforms
        self.windzones = build_windzones()
        self.breakables = build_breakables()
        self.teleports = build_teleports()
        self.plat_list = spawnable_surfaces(self.level)

        self.reset_match()

    def give_random_weapon(self, p:'Player'):
        gdef = random.choice(GUNS)
        ent = GunEntity(gdef, (p.x, p.y - p.h*0.6), spawned=False)
        ent.owner = p; ent.state = "equipped"; p.holding = ent

    def active_level_rects(self):
        rects = list(self.level)
        for b in self.breakables:
            if not b["down"]:
                rects.append((b["rect"], TILE_SOLID))
        return rects

    def reset_match(self):
        self.players: List[Player] = []
        self.particles: List[Particle] = []
        self.bullets: List[Bullet] = []
        self.beams: List[Beam] = []
        self.guns: List[GunEntity] = []

        self.players.append(Player(WORLD_W*0.25, WORLD_H*0.3, ACCENT2, "YOU", is_bot=False, game=self))
        self.players.append(Player(WORLD_W*0.75, WORLD_H*0.3, ACCENT1, "BOT", is_bot=True, game=self))
        self.botbrain = BotBrain()

        for p in self.players: self.give_random_weapon(p)

        self.time_left = 120.0
        self.spawn_interval = SPAWN_INTERVAL_START
        self.spawn_timer = 0.5

        self.respawns = [(WORLD_W*0.18, WORLD_H*0.2), (WORLD_W*0.82, WORLD_H*0.2), (WORLD_W*0.5, WORLD_H*0.2)]

        # input edges (for human)
        self.prev_lmb = self.prev_rmb = self.prev_jump = self.prev_shift = False
        self.curr_lmb = self.curr_rmb = self.curr_jump = self.curr_shift = False

        self.cam = [0.0, 0.0]

        # anti-stuck trackers
        self.stuck_timer = 0.0

    def get_inputs(self, p:Player):
        # Human input -> player control (mouse + keyboard)
        mx, my = pygame.mouse.get_pos()
        mx *= SCALE; my *= SCALE
        mx += self.cam[0]; my += self.cam[1]

        keys = pygame.key.get_pressed()
        mouse = pygame.mouse.get_pressed(num_buttons=3)

        left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
        down = keys[pygame.K_s] or keys[pygame.K_DOWN]

        attack = self.curr_lmb and not self.prev_lmb
        rmb_release = (not self.curr_rmb and self.prev_rmb)

        jump_pressed = (self.curr_jump and not self.prev_jump)
        dash_pressed = (self.curr_shift and not self.prev_shift)

        return {"left":left,"right":right,"down":down,
                "jump_pressed":jump_pressed,
                "attack":attack,
                "rmb_release":rmb_release,
                "aim":(mx, my), "dash":dash_pressed}

    def spawn_gun(self):
        if len(self.guns) >= MAX_GUNS: return
        r = random.choice(self.plat_list + [b["rect"] for b in self.breakables if not b["down"]])
        pad = 40
        x = random.uniform(r.x + pad, r.right - pad)
        y = r.top - 24
        for _ in range(8):
            if any((g.state=="ground" and (g.x-x)**2 + (g.y-y)**2 < (56*56)) for g in self.guns):
                x = random.uniform(r.x + pad, r.right - pad)
            else:
                break
        gdef = random.choice(GUNS)
        ammo = gdef.ammo + random.randint(-2, 2)
        self.guns.append(GunEntity(gdef, (x,y), ammo_override=max(1, ammo), spawned=True))

    def _anti_stuck(self, dt):
        """Push players apart if too close; update a timer for RL penalty."""
        a, b = self.players[0], self.players[1]
        dx, dy = b.x - a.x, (b.y - b.h*0.5) - (a.y - a.h*0.5)
        dist = math.hypot(dx, dy)
        if dist < 32:
            self.stuck_timer += dt
            # Simple horizontal separation
            push = (32 - max(dist, 1.0)) * 0.5
            if dx >= 0:
                a.x -= push; b.x += push
            else:
                a.x += push; b.x -= push
            a.vx *= 0.5; b.vx *= 0.5
        else:
            self.stuck_timer = max(0.0, self.stuck_timer - dt*0.5)

    def update(self, dt, override_inputs_first:Optional[Dict]=None):
        self.dt = dt
        self.time_left = max(0.0, self.time_left - dt)

        # spawn cadence
        self.spawn_timer -= dt
        if self.spawn_timer <= 0.0:
            if len(self.guns) < MAX_GUNS:
                self.spawn_gun()
                self.spawn_interval = max(SPAWN_MIN, self.spawn_interval * 0.96)
            self.spawn_timer = self.spawn_interval

        # input edges (human)
        try:
            self.prev_lmb, self.prev_rmb, self.prev_jump, self.prev_shift = self.curr_lmb, self.curr_rmb, self.curr_jump, self.curr_shift
        except AttributeError:
            self.prev_lmb = self.prev_rmb = self.prev_jump = self.prev_shift = False
            self.curr_lmb = self.curr_rmb = self.curr_jump = self.curr_shift = False

        mouse = pygame.mouse.get_pressed(num_buttons=3)
        keys = pygame.key.get_pressed()
        self.curr_lmb = mouse[0]; self.curr_rmb = mouse[2]
        self.curr_jump = (keys[pygame.K_w] or keys[pygame.K_SPACE])
        self.curr_shift = (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])

        level_now = self.active_level_rects()

        # Players: YOU (index 0) may be overridden by RL action
        for idx, p in enumerate(self.players):
            if idx == 0 and override_inputs_first is not None:
                inputs = override_inputs_first
            else:
                inputs = self.botbrain.think(p, self.players, self.guns, dt) if p.is_bot else self.get_inputs(p)
            p.update(dt, inputs, level_now, self.bullets, self.guns, self.players, self.particles, self.beams,
                     self.windzones, self.breakables, self.teleports)

        # Guns
        for g in self.guns[:]:
            alive = g.update(dt, level_now)
            if not alive and not g.spawned_pickup:
                try: self.guns.remove(g)
                except ValueError: pass

        # Bullets
        for bb in self.bullets[:]:
            alive = bb.update(dt, level_now, self.players, self.particles, self.breakables)
            if not alive:
                try: self.bullets.remove(bb)
                except ValueError: pass

        # Beams
        for bm in self.beams[:]:
            alive = bm.update(dt, self.players, self.particles, self.breakables)
            if not alive: self.beams.remove(bm)

        # Particles
        self.particles[:] = [pa for pa in self.particles if pa.update(dt)]

        # Breakables respawn tick (unused by default)
        for b in self.breakables:
            if b["down"]:
                b["respawn"] -= dt
                if b["respawn"] <= 0:
                    b["down"] = False
                    b["hp"] = b["hp_max"]

        # Respawn players immediately (arcade)
        for p in self.players:
            if not p.alive:
                p.respawn(random.choice(self.respawns))
                self.give_random_weapon(p)

        # Anti-stuck separation
        self._anti_stuck(dt)

        # Camera follow YOU
        you = self.players[0]
        target_x = you.x - RW/2
        target_y = you.y - RH/2
        self.cam[0] = clamp(lerp(self.cam[0], target_x, 0.15), 0, WORLD_W - RW)
        self.cam[1] = clamp(lerp(self.cam[1], target_y, 0.15), 0, WORLD_H - RH)

    def draw_hud(self, surf):
        t = time.gmtime(int(self.time_left))
        txt = f"{t.tm_min:02d}:{t.tm_sec:02d}"
        img = self.font.render(txt, True, INK)
        surf.blit(img, (RW//2 - img.get_width()//2, 12))
        you = self.players[0]
        gun_txt = "None" if not you.holding else f"{you.holding.gdef.name} [{you.holding.ammo}]"
        info = self.font.render(gun_txt, True, INK)
        surf.blit(info, (16, 12))
        x = RW - 16
        for pl in reversed(self.players):
            s = self.font.render(f"{pl.name} KOs:{pl.kos}", True, INK)
            x -= s.get_width()
            surf.blit(s, (x, 12))
            x -= 24

    def render(self) -> pygame.Surface:
        try: pygame.event.pump()
        except: pass
        self.scene.fill(BG)
        cam = self.cam
        draw_level(self.scene, self.active_level_rects(), cam, self.breakables, self.windzones, self.teleports)
        for g in self.guns: g.draw(self.scene, cam)
        for b in self.bullets: b.draw(self.scene, cam)
        for bm in self.beams: bm.draw(self.scene, cam)
        for p in self.players: p.draw(self.scene, cam)
        for pa in self.particles: pa.draw(self.scene, cam)
        self.draw_hud(self.scene)
        scaled = pygame.transform.smoothscale(self.scene, (WIN_W, WIN_H))
        if self.window_for_human:
            self.screen.blit(scaled, (0,0)); pygame.display.flip()
        return scaled

    def human_loop(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: running = False
                    elif event.key == pygame.K_r: self.reset_match()
            self.update(dt)         # human reads real keyboard/mouse
            self.render()
        pygame.quit()

# -------------------- Gymnasium Environment --------------------
class GameEnv(gym.Env):
    """
    Observation:  (float32 Box, shape=(36,))
        [ you.x/W, you.y/H, you.vx/1000, you.vy/1500, you.on_ground(0/1), you.hp/100,
          you.has_gun(0/1), you.ammo/50, you.cooldown, 
          bot.x/W, bot.y/H, bot.vx/1000, bot.vy/1500, bot.hp/100,
          rel_dx/1000, rel_dy/1000, rel_dist/1500, rel_angle/cos, rel_angle/sin,
          you.face(-1/1), bot.face(-1/1),
          nearest_gun_dx/1000, nearest_gun_dy/1000, nearest_gun_dist/1500,
          time_left/120,
          bullets_n(0..1), nearest_bullet_dx/1000, nearest_bullet_dy/1000,
          nearest_bullet_vx/1500, nearest_bullet_vy/1500,
          you.dash_cd, you.air_jumps/1, tp_cd, stuck_timer(0..2), you.kos, bot.kos
        ]

    Action: Discrete(16)  (macro actions)
        0 Idle
        1 Move Left
        2 Move Right
        3 Jump
        4 Dash
        5 Drop-through
        6 Fire
        7 Pickup/Throw
        8 Aim Left
        9 Aim Right
        10 Aim Up
        11 Aim Down
        12 Strafe Left + Fire
        13 Strafe Right + Fire
        14 Jump + Fire
        15 Dash + Fire

    Reward shaping:
        +0.10 per damage dealt
        -0.08 per damage taken
        +1.0 on KO given, -1.0 on self death (already counted via damage, this is bonus)
        +0.0005 * movement_speed each step (encourage activity)
        -0.001 per step time penalty
        -0.01 per step if stuck_timer > 0.5s (anti-camping/overlap)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode:Optional[str]=None, seed:Optional[int]=None, max_steps:int=1800):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        # Create game core (hidden window unless human)
        self._game = Game(window_for_human=(render_mode=="human"))

        # Observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)
        self.action_space = spaces.Discrete(16)

        # Persistent aim target (for RL control)
        self._aim = np.array([self._game.players[0].x+100, self._game.players[0].y-40], dtype=np.float32)

        # Trackers for reward
        self._last_you_hp = 100
        self._last_bot_hp = 100
        self._last_you_kos = 0
        self._last_bot_kos = 0
        self._step_count = 0

    # ------------- Helper: build override inputs from action -------------
    def _action_to_inputs(self, a:int) -> Dict:
        you = self._game.players[0]
        # adjust aim
        aim_step = 120.0
        if a == 8:   self._aim[0] -= aim_step
        if a == 9:   self._aim[0] += aim_step
        if a == 10:  self._aim[1] -= aim_step
        if a == 11:  self._aim[1] += aim_step
        # keep aim within world
        self._aim[0] = clamp(self._aim[0], 0, WORLD_W)
        self._aim[1] = clamp(self._aim[1], 0, WORLD_H)

        left = (a in (1,12))
        right = (a in (2,13))
        jump_pressed = (a in (3,14))
        dash_pressed = (a in (4,15))
        drop = (a == 5)
        fire = (a in (6,12,13,14,15))
        pick_throw = (a == 7)

        return {
            "left": left, "right": right, "down": drop,
            "jump_pressed": jump_pressed,
            "attack": fire,
            "rmb_release": pick_throw,
            "aim": (float(self._aim[0]), float(self._aim[1])),
            "dash": dash_pressed
        }

    # ------------- Helper: observation vector -------------
    def _nearest_gun(self, you:Player):
        best = None; bd2 = 1e20
        for g in self._game.guns:
            if g.state != "ground": continue
            dx, dy = g.x - you.x, g.y - (you.y - you.h*0.5)
            d2 = dx*dx + dy*dy
            if d2 < bd2: bd2 = d2; best = (dx, dy, math.sqrt(d2))
        if best is None: return (0.0, 0.0, 9999.0)
        return best

    def _nearest_bullet(self, you:Player):
        best = None; bd2 = 1e20
        for b in self._game.bullets:
            if b.owner is you: continue
            dx, dy = b.x - you.x, b.y - (you.y - you.h*0.5)
            d2 = dx*dx + dy*dy
            if d2 < bd2: bd2 = d2; best = (dx, dy, b.vx, b.vy, math.sqrt(d2))
        if best is None: return (0.0, 0.0, 0.0, 0.0, 9999.0)
        return best

    def _get_obs(self) -> np.ndarray:
        you = self._game.players[0]; bot = self._game.players[1]
        relx = bot.x - you.x; rely = (bot.y - bot.h*0.5) - (you.y - you.h*0.5)
        dist = math.hypot(relx, rely)
        ang = math.atan2(rely, relx)
        has_gun = float(you.holding is not None)
        ammo = you.holding.ammo if you.holding else 0
        cd = you.attack_cool
        gdx, gdy, gdist = self._nearest_gun(you)
        nb_dx, nb_dy, nb_vx, nb_vy, nb_dist = self._nearest_bullet(you)
        bullets_n = float(len(self._game.bullets) > 0)
        obs = np.array([
            you.x/WORLD_W, you.y/WORLD_H, you.vx/1000.0, you.vy/1500.0,
            float(you.on_ground), you.hp/100.0, has_gun, min(ammo,50)/50.0, cd,
            bot.x/WORLD_W, bot.y/WORLD_H, bot.vx/1000.0, bot.vy/1500.0, bot.hp/100.0,
            relx/1000.0, rely/1000.0, dist/1500.0, math.cos(ang), math.sin(ang),
            float(you.face), float(bot.face),
            gdx/1000.0, gdy/1000.0, gdist/1500.0,
            self._game.time_left/120.0,
            bullets_n, nb_dx/1000.0, nb_dy/1000.0, nb_vx/1500.0, nb_vy/1500.0,
            you.dash_cd, float(you.air_jumps), you.tp_cd, clamp(self._game.stuck_timer,0,2)/2.0,
            float(you.kos), float(bot.kos)
        ], dtype=np.float32)
        return obs

    # ------------- Gym API -------------
    def reset(self, *, seed:Optional[int]=None, options:Optional[dict]=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._game.reset_match()
        # randomize spawn sides a bit
        if self._rng.random() < 0.5:
            self._game.players[0].x, self._game.players[1].x = self._game.players[1].x, self._game.players[0].x
        self._aim = np.array([self._game.players[0].x+100, self._game.players[0].y-40], dtype=np.float32)

        self._last_you_hp = self._game.players[0].hp
        self._last_bot_hp = self._game.players[1].hp
        self._last_you_kos = self._game.players[0].kos
        self._last_bot_kos = self._game.players[1].kos
        self._step_count = 0

        obs = self._get_obs()
        info = {}
        if self.render_mode == "human":
            self._game.render()
        return obs, info
    
    def step(self, action: int):
    # ===== fixed-step sim =====
        self._step_count += 1
        dt = 1.0 / FPS

        # ---- persistent scratch (across steps) ----
        if not hasattr(self, "_prev"):
            self._prev = {
                "gun_dist": None,
                "eng_dev": None,
                "aim_dot": 0.0,
                "threat_dist": None,
            }

        # ---- helpers ----
        def has_los(ax, ay, bx, by) -> bool:
            level = self._game.active_level_rects()
            dx, dy = bx - ax, by - ay
            L = max(1.0, math.hypot(dx, dy))
            steps = int(L / 12.0) + 1
            for i in range(1, steps):
                t = i / steps
                px, py = ax + dx * t, ay + dy * t
                for r, ttype in level:
                    if ttype != TILE_SOLID:
                        continue
                    if r.left <= px <= r.right and r.top <= py <= r.bottom:
                        return False
            return True

        def nearest_gun(player):
            best, bd2 = None, 1e18
            for g in self._game.guns:
                if g.state != "ground":
                    continue
                dx, dy = g.x - player.x, g.y - (player.y - player.h)
                d2 = dx*dx + dy*dy
                if d2 < bd2:
                    bd2, best = d2, g
            return best, (math.sqrt(bd2) if best else 9e9)

        def nearest_threat_bullet(player):
            cx, cy = player.x, player.y - player.h*0.5
            best, bd2 = None, 1e18
            for b in self._game.bullets:
                if b.owner is player:
                    continue
                dx, dy = b.x - cx, b.y - cy
                d2 = dx*dx + dy*dy
                if d2 < bd2:
                    bd2, best = d2, b
            if best is None:
                return None, 9e9, False
            d = math.sqrt(bd2)
            to_me_x, to_me_y = cx - best.x, cy - best.y
            L = max(1e-6, math.hypot(to_me_x, to_me_y))
            approaching = (best.vx * (to_me_x / L) + best.vy * (to_me_y / L) > 0.0)
            return best, d, (approaching and d < 300.0)

        def aim_alignment(aim_xy, you_xy, tar_xy) -> float:
            ax, ay = aim_xy[0] - you_xy[0], aim_xy[1] - you_xy[1]
            tx, ty = tar_xy[0] - you_xy[0], tar_xy[1] - you_xy[1]
            aL = max(1e-6, math.hypot(ax, ay))
            tL = max(1e-6, math.hypot(tx, ty))
            return (ax / aL) * (tx / tL) + (ay / aL) * (ty / tL)  # -1..1

        def weapon_engagement_ring(holding) -> tuple:
            # 무기별 권장 교전 거리 (NEAR, FAR)
            if holding is None:
                return (260.0, 520.0)
            name = holding.gdef.name
            if name == "Shotgun":
                return (140.0, 300.0)
            if name == "SMG":
                return (200.0, 420.0)
            if name == "Pistol":
                return (220.0, 480.0)
            if name == "Rifle":
                return (260.0, 540.0)
            if name == "Sniper":
                return (380.0, 900.0)
            if name == "Rocket":
                return (240.0, 520.0)
            if name == "Laser":
                return (260.0, 560.0)
            return (260.0, 520.0)

        # ---- pre-snapshot ----
        you = self._game.players[0]
        bot  = self._game.players[1]

        you_hp_prev, bot_hp_prev   = you.hp, bot.hp
        you_kos_prev, bot_kos_prev = you.kos, bot.kos

        you_cx_pre, you_cy_pre = you.x, you.y - you.h*0.6
        bot_cx_pre,  bot_cy_pre  = bot.x, bot.y - bot.h*0.6
        dist_pre = math.hypot(bot_cx_pre - you_cx_pre, bot_cy_pre - you_cy_pre)

        aim_pre = np.array(self._aim, dtype=np.float32)
        aim_dot_pre = aim_alignment(aim_pre, (you_cx_pre, you_cy_pre), (bot_cx_pre, bot_cy_pre))
        los_pre = has_los(you_cx_pre, you_cy_pre, bot_cx_pre, bot_cy_pre)

        gun_ent_pre, gun_dist_pre = nearest_gun(you)
        threat_b_pre, threat_dist_pre, threat_flag_pre = nearest_threat_bullet(you)

        has_gun_pre = you.holding is not None
        ammo_pre = (you.holding.ammo if you.holding else 0)
        beam_pre = bool(has_gun_pre and getattr(you.holding.gdef, "special", "") == "beam")
        unarmed_like_pre = (not has_gun_pre) or (ammo_pre <= 0 and not beam_pre)

        NEAR_pre, FAR_pre = weapon_engagement_ring(you.holding)
        dev_pre = 0.0 if NEAR_pre <= dist_pre <= FAR_pre else (NEAR_pre - dist_pre if dist_pre < NEAR_pre else dist_pre - FAR_pre)

        self._prev["gun_dist"] = gun_dist_pre
        self._prev["eng_dev"]  = dev_pre
        self._prev["aim_dot"]  = aim_dot_pre
        self._prev["threat_dist"] = threat_dist_pre

        # ---- build inputs from action ----
        inputs = self._action_to_inputs(int(action))
        fired_this_step = int(action) in (6, 12, 13, 14, 15)

        # ---- advance world ----
        self._game.update(dt, override_inputs_first=inputs)

        # ---- post-snapshot ----
        you = self._game.players[0]
        bot  = self._game.players[1]

        you_cx, you_cy = you.x, you.y - you.h*0.6
        bot_cx,  bot_cy  = bot.x, bot.y - bot.h*0.6
        dist = math.hypot(bot_cx - you_cx, bot_cy - you_cy)
        los  = has_los(you_cx, you_cy, bot_cx, bot_cy)

        aim_now = np.array(self._aim, dtype=np.float32)
        aim_dot = aim_alignment(aim_now, (you_cx, you_cy), (bot_cx, bot_cy))

        gun_ent, gun_dist = nearest_gun(you)
        threat_b, threat_dist, threat_flag = nearest_threat_bullet(you)

        has_gun = you.holding is not None
        ammo = (you.holding.ammo if you.holding else 0)
        beam = bool(has_gun and getattr(you.holding.gdef, "special", "") == "beam")
        unarmed_like = (not has_gun) or (ammo <= 0 and not beam)

        # ---- outcomes ----
        dmg_dealt = max(0, bot_hp_prev - bot.hp)
        dmg_taken = max(0, you_hp_prev - you.hp)
        kos_you   = you.kos - you_kos_prev
        kos_bot   = bot.kos - bot_kos_prev

        # ===== base reward: kill/hit/survive (dominates) =====
        reward  = 0.30 * float(dmg_dealt)
        reward -= 0.25 * float(dmg_taken)
        reward += 20.0 * float(kos_you)
        reward -= 20.0 * float(kos_bot)

        # ===== weapon handling =====
        if unarmed_like_pre or unarmed_like:
            # 진짜 진행(reward for distance reduction to nearest gun)
            if self._prev["gun_dist"] is not None and math.isfinite(self._prev["gun_dist"]) and math.isfinite(gun_dist):
                prog = float(self._prev["gun_dist"] - gun_dist)      # >0 가까워짐
                reward += 0.006 * prog                                # ~+0.06/10px
            # 가만히 있지 않게 약한 압박
            speed = abs(you.vx) + 0.2 * abs(you.vy)
            if speed < 140.0:
                reward -= 0.008 * dt
            # 줍기 성공 보너스
            if unarmed_like_pre and (not unarmed_like):
                reward += 2.0

        # ===== engagement shaping (armed-like) =====
        NEAR, FAR = weapon_engagement_ring(you.holding if has_gun else None)
        dev = 0.0 if NEAR <= dist <= FAR else (NEAR - dist if dist < NEAR else dist - FAR)
        if not unarmed_like:
            # 링 안: 소폭 유지 보상
            if NEAR <= dist <= FAR:
                reward += 0.025 * dt
            # 링에 접근하는 상대적 개선 보상
            if self._prev["eng_dev"] is not None:
                reward += 0.006 * float(self._prev["eng_dev"] - dev)
            # 계속 크게 벗어나면 미세 패널티
            reward -= 0.003 * dt * float(min(1.0, abs(dev) / 700.0))

            # LOS & 조준 정합
            if los:
                reward += 0.02 * max(0.0, aim_dot) * dt
            else:
                # 위협 상황에서 엄폐(LOS 꺼짐)면 소폭 보상
                if threat_flag or (self._prev["threat_dist"] is not None and self._prev["threat_dist"] < 220.0):
                    reward += 0.005 * dt
                else:
                    reward -= 0.003 * dt

            # 발사 유인: 맞출 수 있을 때만 강하게
            if fired_this_step and has_gun:
                if los and aim_dot > 0.70:
                    reward += 0.18
                elif aim_dot < 0.40:
                    reward -= 0.03  # 허공 난사 억제

                # 탄약 관리: 탄 적을 때는 정합 높을 때만 쏘게 유도
                if ammo <= max(3, int((you.holding.gdef.ammo or 6) * 0.2)) and aim_dot < 0.85:
                    reward -= 0.02

            # 조준 향상 자체에 대한 짧은 보상
            if self._prev["aim_dot"] is not None:
                delta_aim = max(0.0, float(aim_dot - self._prev["aim_dot"]))
                reward += min(0.08, 0.12 * delta_aim)

        # ===== threat avoidance (armed/unarmed 공통) =====
        if threat_flag:
            reward -= 0.04 * dt  # 탄이 접근 중이면 압박
        if (self._prev["threat_dist"] is not None
            and math.isfinite(self._prev["threat_dist"])
            and math.isfinite(threat_dist)
            and self._prev["threat_dist"] < 350.0):
            # 위협으로부터 멀어졌으면 보상
            reward += 0.006 * float(threat_dist - self._prev["threat_dist"])
            # 근접 회피 성공(near-miss)
            if self._prev["threat_dist"] < 140.0 and threat_dist > 200.0 and dmg_taken == 0:
                reward += 0.10

        # ===== environment mild penalties =====
        if you.x < 70.0 or you.x > WORLD_W - 70.0:
            reward -= 0.008 * dt
        if self._game.stuck_timer > 0.6:
            reward -= 0.02 * dt

        # faint time pressure
        reward -= 0.0002 * dt

        # ---- termination / truncation ----
        terminated = False
        truncated = (self._step_count >= self.max_steps) or (self._game.time_left <= 0.0)

        # ---- obs & info ----
        obs = self._get_obs()
        info = {
            "dmg_dealt": float(dmg_dealt),
            "dmg_taken": float(dmg_taken),
            "you_kos":   int(you.kos),
            "bot_kos":   int(bot.kos),
            "dist":      float(dist),
            "los":       bool(los),
            "aim_dot":   float(aim_dot),
            "has_gun":   bool(has_gun),
            "ammo":      int(ammo),
            "gun_dist":  float(gun_dist),
        }

        if self.render_mode == "human":
            self._game.render()

        return obs, float(reward), terminated, truncated, info



    def render(self):
        frame = self._game.render()
        if self.render_mode == "rgb_array":
            arr = pygame.surfarray.array3d(frame).swapaxes(0,1).copy()
            return arr
        return None

    def close(self):
        pygame.quit()

# -------------------- Human mode entrypoint --------------------
if __name__ == "__main__":
    # Human playable session via GameEnv
    env = GameEnv(render_mode='human')
    print("Controls: A/D move, W/Space jump, S drop-through, Shift dash, LMB fire, RMB pick/throw, R reset, ESC quit")
    # Use built-in human loop from core game for the best feel (mouse aim, continuous inputs)
    env._game.human_loop()
    env.close()
