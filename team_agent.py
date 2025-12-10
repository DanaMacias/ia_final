import socket
import time
import threading
import json
import os
import re
import random
import math
from heapq import heappush, heappop

HOST = "127.0.0.1"
PORT = 6000
TEAM = "MY_TEAM"
CONF_FILE = "conf_file.conf"
PLAYERS = 11

# --- Parámetros de campo (valores típicos RoboCup 2D) ---
FIELD_HALF_X = 52.5
FIELD_HALF_Y = 34.0

# Globals para coordinación del balón (un solo perseguidor)
ball_chaser = None
ball_chaser_dist = float("inf")
ball_chaser_lock = threading.Lock()
ball_chaser_last_update = 0.0
# cooldown para evitar reasignaciones/robos excesivos del token (segundos)
BALL_CHASER_MIN_CHANGE_DT = 0.08
_last_ball_chaser_change = 0.0

DEBUG = True

# World model compartido
world_lock = threading.Lock()
world_model = {
    "ball": {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "t": 0.0, "seen": False},
    "opponents": [],
    "teammates": {}
}

# Configuraciones
BALL_LOST_TIMEOUT = 4.0
BALL_PROCESS_DT = 0.1

def load_positions():
    default = {
        1: (-50, 0), 2: (-35, -8), 3: (-35, 8), 4: (-30, -12), 5: (-30, 12),
        6: (-10, -8), 7: (-10, 8), 8: (-18, 0), 9: (5, 10), 10: (5, -10), 11: (12, 0)
    }

    if not os.path.exists(CONF_FILE):
        return default

    try:
        with open(CONF_FILE, "r") as f:
            data = json.load(f)

        if "data" in data and isinstance(data["data"], list):
            data = data["data"][0]

        pos = {}
        for k, v in data.items():
            if k.isdigit() and "x" in v and "y" in v:
                pos[int(k)] = (float(v["x"]), float(v["y"]))

        for i in range(1, PLAYERS + 1):
            if i not in pos:
                pos[i] = default[i]

        return pos
    except:
        return default

# ------------------ KALMAN FILTER PARA BALÓN ------------------
class KalmanBall:
    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0):
        self.x = x; self.y = y; self.vx = vx; self.vy = vy
        self.P = [[1.0,0,0,0],[0,1.0,0,0],[0,0,1.0,0],[0,0,0,1.0]]
        self.q = 0.05
        self.r = 0.5
        self.last_t = time.time()

    def predict(self, t=None):
        now = time.time() if t is None else t
        dt = now - self.last_t
        if dt <= 0:
            return
        self.x += self.vx * dt
        self.y += self.vy * dt
        for i in range(4):
            self.P[i][i] += self.q * dt
        self.last_t = now

    def update(self, meas_x, meas_y, t=None):
        now = time.time() if t is None else t
        dt = now - self.last_t
        if dt > 0:
            vx_new = (meas_x - self.x) / dt
            vy_new = (meas_y - self.y) / dt
            alpha = 0.6
            self.vx = alpha * vx_new + (1 - alpha) * self.vx
            self.vy = alpha * vy_new + (1 - alpha) * self.vy
        K_pos = 1.0 / (1.0 + self.r)
        self.x = (1 - K_pos) * self.x + K_pos * meas_x
        self.y = (1 - K_pos) * self.y + K_pos * meas_y
        self.last_t = now

    def state(self):
        return (self.x, self.y, self.vx, self.vy)

# ------------------- PATH PLANNING A* -------------------
class AStarPlanner:
    def __init__(self, x_min=-FIELD_HALF_X, x_max=FIELD_HALF_X, y_min=-FIELD_HALF_Y, y_max=FIELD_HALF_Y, res=1.0):
        self.xmin = x_min; self.xmax = x_max; self.ymin = y_min; self.ymax = y_max
        self.res = res

    def _to_grid(self, x, y):
        gx = int(round((x - self.xmin) / self.res))
        gy = int(round((y - self.ymin) / self.res))
        return gx, gy

    def _to_world(self, gx, gy):
        x = self.xmin + gx * self.res
        y = self.ymin + gy * self.res
        return x, y

    def _neighbors(self, node):
        x,y = node
        steps = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for dx,dy in steps:
            nx,ny = x+dx, y+dy
            wx,wy = self._to_world(nx,ny)
            if self.xmin - 0.1 <= wx <= self.xmax + 0.1 and self.ymin - 0.1 <= wy <= self.ymax + 0.1:
                yield (nx,ny)

    def heuristic(self, a,b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def plan(self, start, goal, obstacles=None, obs_inflation=1.0, max_iter=20000):
        if obstacles is None:
            obstacles = []
        s = self._to_grid(*start)
        g = self._to_grid(*goal)
        occ = set()
        for ox,oy in obstacles:
            og = self._to_grid(ox,oy)
            r = int(math.ceil(obs_inflation / self.res))
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    occ.add((og[0]+dx, og[1]+dy))
        openq = []
        heappush(openq, (0 + self.heuristic(s,g), 0, s, None))
        came = {}
        cost = {s:0}
        it = 0
        while openq and it < max_iter:
            it += 1
            _, c, current, parent = heappop(openq)
            if current in came:
                continue
            came[current] = parent
            if current == g:
                break
            for n in self._neighbors(current):
                if n in occ:
                    continue
                nc = c + math.hypot(n[0]-current[0], n[1]-current[1])
                if n not in cost or nc < cost[n]:
                    cost[n] = nc
                    pr = nc + self.heuristic(n,g)
                    heappush(openq, (pr, nc, n, current))
        if g not in came:
            return None
        path = []
        cur = g
        while cur is not None:
            path.append(self._to_world(*cur))
            cur = came[cur]
        path.reverse()
        path = self._smooth_path(path, obstacles)
        return path

    def _smooth_path(self, path, obstacles):
        if len(path) < 3:
            return path
        sm = [path[0]]
        i = 0
        while i < len(path)-1:
            j = len(path)-1
            while j > i+1:
                if self._line_free(path[i], path[j], obstacles):
                    break
                j -= 1
            sm.append(path[j])
            i = j
        return sm

    def _line_free(self, a,b,obstacles, eps=0.8):
        for ox,oy in obstacles:
            if self._dist_point_segment((ox,oy), a, b) < eps:
                return False
        return True

    def _dist_point_segment(self, p, a, b):
        ax,ay = a; bx,by = b; px,py = p
        if ax==bx and ay==by:
            return math.hypot(px-ax, py-ay)
        t = ((px-ax)*(bx-ax)+(py-ay)*(by-ay))/((bx-ax)**2+(by-ay)**2)
        t = max(0.0, min(1.0, t))
        projx = ax + t*(bx-ax); projy = ay + t*(by-ay)
        return math.hypot(px-projx, py-projy)

# ------------------- JUGADOR BOT ----------------------
used_positions = []
used_positions_lock = threading.Lock()
stop_event = threading.Event()

planner = AStarPlanner(res=1.0)

class Player:
    GK = 1
    DEF = [2, 3, 4, 5]
    MID = [6, 7, 8]
    FWD = [9, 10, 11]

    def __init__(self, n, pos):
        self.idx = n
        self.pos_map = pos
        self.unum = None
        self.side = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", 0))
        self.sock.settimeout(1)
        self.ball = None
        self.own_goal_x = None
        self.opp_goal_x = None
        self.opponents_positions = []
        self.kf = KalmanBall()
        self.last_ball_seen_time = 0.0
        self.current_path = None
        self.path_index = 0
        self.path_replan_ts = 0.0
        self.current_pos = None   # posición actual (se actualizará desde el (see) player)
    
    def get_best_shot_target(self):
        """
        Busca si hay línea de tiro libre al centro, o a los postes.
        Devuelve (target_x, target_y) o None.
        """
        mypos = self.my_abs_pos()
        # Coordenadas de los postes (reducimos un poco para asegurar que entre)
        # El arco suele ir de -7 a 7 en Y. Apuntamos a -6, 0 y 6.
        targets_y = [0.0, 6.0, -6.0, 3.0, -3.0] 
        
        # Filtrar oponentes (ignoramos a los compañeros para tirar, 
        # a veces hay que arriesgar aunque haya un compañero en medio)
        with world_lock:
            opponents = list(world_model.get("opponents", []))

        # Distancia máxima de tiro efectiva (ej. 32 metros)
        dist_to_goal = math.hypot(self.opp_goal_x - mypos[0], 0 - mypos[1])
        if dist_to_goal > 35.0:
            return None # Muy lejos para tirar directo

        best_target = None
        
        # Revisar cada punto del arco
        for ty in targets_y:
            target_pos = (self.opp_goal_x, ty)
            # Verificar si alguien bloquea este disparo específico
            if self.can_pass_without_intercept(mypos, target_pos, ignore_unums=list(range(1, 12))):
                return target_pos # ¡Encontramos un hueco! Disparar aquí.

        return None

    def close_to_ball(self):
        if not self.ball:
            return False
        dist, ang = self.ball
        return dist < 0.9  # levemente más tolerante

    def kick_towards(self, target_x, target_y, power=80):
        myx, myy = self.my_abs_pos()
        dx = target_x - myx
        dy = target_y - myy
        abs_angle = math.degrees(math.atan2(dy, dx))
        facing = 0.0 if self.side == "l" else 180.0
        rel_angle = abs_angle - facing
        while rel_angle > 180: rel_angle -= 360
        while rel_angle < -180: rel_angle += 360
        # si el ángulo relativo es grande, giramos antes
        if abs(rel_angle) > 25:
            self.send(f"(turn {rel_angle:.1f})")
            time.sleep(0.06)
        # enviar kick con ángulo relativo 0 (ya nos orientamos) o con rel_angle pequeño
        send_ang = 0 if abs(rel_angle) <= 10 else rel_angle
        self.send(f"(kick {power:.1f} {send_ang:.1f})")
        # liberamos token tras patear
        self.release_ball_chaser_if_mine()

    def perform_kick_logic(self):
        # Si no tengo el control del balón (soy chaser pero está lejos), no hago nada
        if not self.close_to_ball():
            return
        # Doble chequeo de seguridad de token
        if not (ball_chaser == self.unum):
            return

        mypos = self.my_abs_pos()
        dist_goal = math.hypot(self.opp_goal_x - mypos[0], 0 - mypos[1])

        shot_target = self.get_best_shot_target()
        if shot_target:
            tx, ty = shot_target
            if DEBUG: print(f"[{self.unum}] ¡TIRO A PUERTA! -> ({tx:.1f}, {ty:.1f})")
            self.kick_towards(tx, ty, power=100)
            return

        # --- ESTRATEGIA 2: TIRO DESESPERADO (Cerca del área chica) ---
        # Si estamos MUY cerca (< 12m) y todo está tapado, patea fuerte al centro igual.
        if dist_goal < 12.0:
             self.kick_towards(self.opp_goal_x, 0.0, power=100)
             return

        # --- ESTRATEGIA 3: PASE ---
        # Si no puedo tirar, busco compañero mejor posicionado
        pass_target = self.find_pass_target()
        if pass_target:
            u, pos, d, adv = pass_target
            # Solo pasar si el compañero está más adelantado (adv > 0) o si estoy muy presionado
            if adv > 2.0 or dist_goal > 30.0:
                tx, ty = pos
                if DEBUG: print(f"[{self.unum}] Pase a {u}")
                self.kick_towards(tx, ty, power=70) # Pase un poco más fuerte
                return

        # --- ESTRATEGIA 4: AVANCE / DESPEJE ---
        # Si no hay tiro ni pase, driblar hacia el arco o despejar
        
        # Calcular ángulo hacia el arco rival
        angle_to_goal = math.degrees(math.atan2(0 - mypos[1], self.opp_goal_x - mypos[0]))
        # Ajustar según mi orientación
        facing = 0.0 if self.side == "l" else 180.0
        rel_angle = angle_to_goal - facing
        
        # Patalear balón hacia adelante (dribbling tosco)
        self.send(f"(kick 30 {rel_angle:.1f})")
        # Nota: Al hacer kick suave, el loop principal volverá a tomar el control 
        # y correrá tras el balón en la siguiente iteración ("dribble").

    def do_chase_actions(self):
        if not self.ball:
            return
        self.turn_to_ball_if_needed()
        if self.close_to_ball():
            self.perform_kick_logic()
            return
        dist, ang = self.ball
        self.send(f"(dash {min(100, 60 + dist*5):.1f})")
        if abs(ang) > 5:
            self.send(f"(turn {ang})")

    def turn_to_ball_if_needed(self):
        if not self.ball:
            return
        _, ang = self.ball
        if abs(ang) > 10:
            self.send(f"(turn {ang})")

    def send(self, c):
        try:
            self.sock.sendto(c.encode(), (HOST, PORT))
        except Exception:
            pass

    def configure_field_sides(self):
        if self.side == "l":
            self.own_goal_x = -FIELD_HALF_X
            self.opp_goal_x = FIELD_HALF_X
        else:
            self.own_goal_x = FIELD_HALF_X
            self.opp_goal_x = -FIELD_HALF_X

    def is_in_own_half(self, x_coord):
        if self.own_goal_x is None:
            return None
        if self.own_goal_x < 0:
            return x_coord < 0
        else:
            return x_coord > 0

    def side_of_point(self, x_coord, neutral_margin=0.0):
        if abs(x_coord) <= neutral_margin:
            return "neutral"
        return "own" if self.is_in_own_half(x_coord) else "opp"

    def get_abs_pos_for_unum(self, unum):
        x, y = self.pos_map.get(unum, (0.0, 0.0))
        if self.side == "r":
            return (-x, -y)
        return (x, y)

    def my_abs_pos(self):
        # devuelve la posición actual estimada (actualizada desde (see player ...))
        if self.current_pos is not None:
            return self.current_pos
        # fallback a pos_map de spawn (por compatibilidad)
        return self.get_abs_pos_for_unum(self.unum)


    # token ball_chaser
    def request_ball_chaser(self, dist):
        global ball_chaser, ball_chaser_dist, ball_chaser_last_update, _last_ball_chaser_change
        now = time.time()
        
        # 1. Calcular mi distancia real (usando visión local O modelo global)
        my_comp = float("inf")
        
        # Si 'dist' (visión local) es válida, úsala.
        if dist is not None:
            my_comp = dist
        else:
            # Si no veo el balón, calculo distancia contra el modelo global compartido
            with world_lock:
                bm = world_model.get("ball", {})
                if bm.get("seen", False) and (now - bm.get("t", 0)) < BALL_LOST_TIMEOUT:
                    bx, by = bm["x"], bm["y"]
                    myx, myy = self.my_abs_pos()
                    my_comp = math.hypot(bx - myx, by - myy)

        # Si mi distancia sigue siendo infinita (nadie sabe donde está el balón), salir
        if my_comp == float("inf"):
            return False

        with ball_chaser_lock:
            # A) Si el dueño actual lleva mucho tiempo sin actualizar (se cayó o perdió el balón), liberar
            if ball_chaser is not None and (now - ball_chaser_last_update) > 2.0: # Reduje el timeout para ser más reactivo
                ball_chaser = None
                ball_chaser_dist = float("inf")

            # B) Si no hay dueño, lo tomo
            if ball_chaser is None:
                ball_chaser = self.unum
                ball_chaser_dist = my_comp
                ball_chaser_last_update = now
                _last_ball_chaser_change = now
                return True

            # C) Si YO soy el dueño, actualizo mi distancia y mantengo el turno
            if ball_chaser == self.unum:
                ball_chaser_dist = my_comp
                ball_chaser_last_update = now
                return True

            # D) INTENTO DE ROBO (La corrección importante)
            # Regla: Solo robo si mi distancia es MENOR que la del actual multiplicada por un factor (ej. 0.9)
            # Esto significa que debo estar un 10% más cerca para justificar el cambio.
            
            # Recuperar distancia del actual (con seguridad)
            current_comp = ball_chaser_dist if (ball_chaser_dist is not None) else float("inf")
            
            # CORRECCIÓN DE FÓRMULA:
            # Tu código anterior: if my_comp * 0.85 < current_comp (permitía robar estando lejos)
            # Nuevo código: if my_comp < current_comp * 0.85 (debo estar MUCHO más cerca)
            
            THRESHOLD_REASSIGN = 0.90 # Ajustado a 0.9 para ser más ágil
            
            if (now - _last_ball_chaser_change) > BALL_CHASER_MIN_CHANGE_DT:
                if my_comp < (current_comp * THRESHOLD_REASSIGN):
                    if DEBUG:
                        print(f"[{self.unum}] ROBO a {ball_chaser} (yo={my_comp:.1f} vs el={current_comp:.1f})")
                    ball_chaser = self.unum
                    ball_chaser_dist = my_comp
                    ball_chaser_last_update = now
                    _last_ball_chaser_change = now
                    return True

            return False


    def release_ball_chaser_if_mine(self):
        global ball_chaser, ball_chaser_dist, ball_chaser_last_update
        with ball_chaser_lock:
            if ball_chaser == self.unum:
                ball_chaser = None
                ball_chaser_dist = float("inf")
                ball_chaser_last_update = 0.0
                if DEBUG:
                    print(f"[{self.unum}] released ball_chaser token")

    def place_spawn(self):
        base = self.pos_map[self.unum]
        x, y = base
        if self.side == "r":
            x, y = -x, -y
        x += random.uniform(-0.5, 0.5)
        y += random.uniform(-0.5, 0.5)
        MIN_DIST = 2.0
        with used_positions_lock:
            for px, py in used_positions:
                if math.dist((x, y), (px, py)) < MIN_DIST:
                    x += random.uniform(MIN_DIST, MIN_DIST + 1.0)
                    y += random.uniform(-1.0, 1.0)
            used_positions.append((x, y))
        self.send(f"(move {x:.1f} {y:.1f})")
        # guardamos la posición inicial como current_pos (coordenadas absolutas)
        if self.side == "r":
            self.current_pos = (-x, -y)
        else:
            self.current_pos = (x, y)
        time.sleep(0.3)


    def parse_see_for_ball(self, msg):
        """
        Parser robusto: busca la palabra 'ball' y toma las dos primeras cifras que aparecen
        después de ella. Decide 'rel' vs 'abs' por heurística:
          - si el segundo número está en [-200,200] y el primero >= 0 -> es muy probablemente (dist, ang) relativo
          - si ambos están dentro de los límites del campo -> absoluto
        """
        lm = msg.lower()
        i = lm.find("ball")
        if i != -1:
            # buscar números después de 'ball'
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", msg[i:])
            if len(nums) >= 2:
                try:
                    a = float(nums[0])
                    b = float(nums[1])
                    # heurística: si el segundo está en rango de ángulos -> 'rel'
                    if -200.0 <= b <= 200.0 and a >= 0.0:
                        return (a, b, "rel")
                    # si parecen coordenadas plausibles en el campo -> 'abs'
                    if -FIELD_HALF_X*1.5 <= a <= FIELD_HALF_X*1.5 and -FIELD_HALF_Y*1.5 <= b <= FIELD_HALF_Y*1.5:
                        return (a, b, "abs")
                    # fallback: tratar como relativo si el primer número es distancia razonable
                    if 0.0 <= a <= 200.0:
                        return (a, b, "rel")
                except Exception:
                    pass

        # si no se encontró la palabra "ball" o no había números, intentar formas antiguas por compatibilidad
        m = re.search(r"\(ball\s+([-\d\.]+)\s+([-\d\.]+)\)", msg)
        if m:
            try:
                return (float(m.group(1)), float(m.group(2)), "rel")
            except:
                pass

        m = re.search(r"ball\)\s*([-\d\.]+)\s+([-\d\.]+)", msg)
        if m:
            try:
                return (float(m.group(1)), float(m.group(2)), "rel")
            except:
                pass

        # no hubo parse válido
        return None

    def see_ball(self, msg):
        global ball_chaser_last_update, ball_chaser
        parsed = self.parse_see_for_ball(msg)
        now = time.time()

        if DEBUG:
            # mostrar solo el inicio del message para no llenar la terminal
            preview = msg.replace("\n", " ")[:220]
            if parsed:
                print(f"[{self.unum}] see_ball: parsed={parsed}  preview='{preview}...'")
            else:
                print(f"[{self.unum}] see_ball: parse falló (preview='{preview}...')")

        if parsed:
            v1, v2, kind = parsed
            if kind == "rel":
                try:
                    dist = float(v1); ang = float(v2)
                    self.ball = (dist, ang)
                    facing = 0.0 if self.side == 'l' else 180.0
                    abs_ang = math.radians(facing + ang)
                    myx, myy = self.my_abs_pos()
                    bx = myx + dist * math.cos(abs_ang)
                    by = myy + dist * math.sin(abs_ang)
                    self.kf.update(bx, by)
                    self.last_ball_seen_time = now
                    with world_lock:
                        world_model["ball"]["x"] = bx
                        world_model["ball"]["y"] = by
                        world_model["ball"]["vx"] = self.kf.vx
                        world_model["ball"]["vy"] = self.kf.vy
                        world_model["ball"]["t"] = now
                        world_model["ball"]["seen"] = True
                    with ball_chaser_lock:
                        if ball_chaser == self.unum:
                            ball_chaser_last_update = now
                    return
                except Exception:
                    pass
            if kind == "abs":
                try:
                    bx = float(v1); by = float(v2)
                    self.kf.update(bx, by)
                    self.last_ball_seen_time = now
                    with world_lock:
                        world_model["ball"]["x"] = bx
                        world_model["ball"]["y"] = by
                        world_model["ball"]["vx"] = self.kf.vx
                        world_model["ball"]["vy"] = self.kf.vy
                        world_model["ball"]["t"] = now
                        world_model["ball"]["seen"] = True
                    myx, myy = self.my_abs_pos()
                    dx = bx - myx; dy = by - myy
                    dist = math.hypot(dx, dy)
                    abs_ang_deg = math.degrees(math.atan2(dy, dx))
                    facing = 0.0 if self.side == 'l' else 180.0
                    ang = abs_ang_deg - facing
                    self.ball = (dist, ang)
                    with ball_chaser_lock:
                        if ball_chaser == self.unum:
                            ball_chaser_last_update = now
                    return
                except Exception:
                    pass

        # si llegamos acá no se parseó
        STALE_TOLERANCE = 1.2
        if now - self.last_ball_seen_time <= STALE_TOLERANCE:
            if DEBUG:
                print(f"[{self.unum}] see_ball: parse falló pero keep previous (last seen {now - self.last_ball_seen_time:.2f}s)")
        else:
            if DEBUG:
                print(f"[{self.unum}] see_ball: balón no visto y timeout -> limpiar")
            self.ball = None
            self.release_ball_chaser_if_mine()

        # parseo de oponentes (igual que antes)
        self.opponents_positions = []
        for m2 in re.finditer(r"opp\s+([-\d\.]+)\s+([-\d\.]+)", msg):
            try:
                ox = float(m2.group(1)); oy = float(m2.group(2))
                self.opponents_positions.append((ox, oy))
            except:
                pass
        if self.opponents_positions:
            with world_lock:
                world_model["opponents"] = list(self.opponents_positions)
                # --- parseo de compañeros (player MY_TEAM <unum>) para actualizar posiciones ---
        # formatos observados en logs:
        #   (player MY_TEAM 2) 8.2 27 0 0
        #   (player MY_TEAM) 24.5 3  (a veces sin número)
        # intentamos capturar ambas formas
        for m2 in re.finditer(r"\(player\s+MY_TEAM\s+(\d+)\)\s+([-\d\.]+)\s+([-\d\.]+)", msg):
            try:
                unum = int(m2.group(1))
                px = float(m2.group(2))
                py = float(m2.group(3))
                # convertir a coordenadas absolutas según el lado
                if self.side == "r":
                    ax, ay = -px, -py
                else:
                    ax, ay = px, py
                # actualizar mapa y current_pos si es mi propio unum
                with world_lock:
                    # opcional: mantener world_model["teammates"] si te sirve
                    tm = world_model.get("teammates", {})
                    tm[unum] = (ax, ay)
                    world_model["teammates"] = tm
                # actualizamos el pos_map (opcional) y current_pos para nuestro propio unum
                self.pos_map[unum] = (px, py)
                if unum == self.unum:
                    self.current_pos = (ax, ay)
            except Exception:
                pass

        # fallback simple si aparece player MY_TEAM sin número:
        m3 = re.search(r"\(player\s+MY_TEAM\)\s+([-\d\.]+)\s+([-\d\.]+)", msg)
        if m3:
            try:
                px = float(m3.group(1)); py = float(m3.group(2))
                if self.side == "r":
                    ax, ay = -px, -py
                else:
                    ax, ay = px, py
                if self.unum is not None:
                    self.current_pos = (ax, ay)
            except:
                pass



    # ---------- pases/tiros helpers ----------
    def line_distance_point(self, a, b, p):
        ax, ay = a; bx, by = b; px, py = p
        if ax == bx and ay == by:
            return math.dist(a, p)
        t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)
        t = max(0.0, min(1.0, t))
        projx = ax + t * (bx - ax)
        projy = ay + t * (by - ay)
        return math.dist((projx, projy), p)

    def can_pass_without_intercept(self, from_pos, to_pos, ignore_unums=None):
        if ignore_unums is None:
            ignore_unums = []
        INTERCEPT_DIST = 1.8
        with world_lock:
            opponents = list(world_model.get("opponents", []))
        for (ox, oy) in opponents:
            d = self.line_distance_point(from_pos, to_pos, (ox, oy))
            if d < INTERCEPT_DIST:
                return False
        return True

    def is_path_clear_to_goal(self):
        mypos = self.my_abs_pos()
        goal_pos = (self.opp_goal_x, 0.0)
        with world_lock:
            opponents = list(world_model.get("opponents", []))
        if not opponents:
            return True
        return self.can_pass_without_intercept(mypos, goal_pos, ignore_unums=list(range(1, PLAYERS + 1)))

    def find_pass_target(self):
        mypos = self.my_abs_pos()
        PASS_MAX_DIST = 35.0
        best = None
        best_score = -1e9
        W_X_ADV = 4.0
        W_DIST = 1.0
        for u in range(1, PLAYERS + 1):
            if u == self.unum:
                continue
            target_pos = self.get_abs_pos_for_unum(u)
            dist = math.dist(mypos, target_pos)
            if dist > PASS_MAX_DIST:
                continue
            my_x = mypos[0]; targ_x = target_pos[0]
            if self.opp_goal_x > 0:
                x_adv = targ_x - my_x
            else:
                x_adv = my_x - targ_x
            score = W_X_ADV * x_adv + W_DIST * (PASS_MAX_DIST - dist)
            if not self.can_pass_without_intercept(mypos, target_pos, ignore_unums=[self.unum, u]):
                continue
            if score > best_score:
                best_score = score
                best = (u, target_pos, dist, x_adv)
        if best is None:
            best_adv = None
            best_x = -1e9
            for u in range(1, PLAYERS + 1):
                if u == self.unum:
                    continue
                tx, ty = self.get_abs_pos_for_unum(u)
                if self.opp_goal_x > 0:
                    adv = tx
                else:
                    adv = -tx
                if adv > best_x:
                    best_x = adv
                    best_adv = (u, (tx, ty))
            if best_adv:
                u, pos = best_adv
                if math.dist(mypos, pos) <= PASS_MAX_DIST * 1.4 and self.can_pass_without_intercept(mypos, pos, ignore_unums=[self.unum, u]):
                    return (u, pos, math.dist(mypos, pos), best_x)
            return None
        return best

    def angle_to_point(self, from_pos, to_pos):
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        ang = math.degrees(math.atan2(dy, dx))
        return ang

    # --------------- DECISION: ofensiva / defensiva ----------------
    def has_ball_control(self, dist):
        return dist is not None and dist < 0.9 and ball_chaser == self.unum

    def offensive_decision(self, dist):
        if self.has_ball_control(dist):
            if self.is_path_clear_to_goal():
                return ("shoot", None)
            choice = self.find_pass_target()
            if choice:
                u, pos, d, _ = choice
                return ("pass", (u, pos, d))
            with world_lock:
                ball = world_model["ball"].copy()
            goal = (self.opp_goal_x, 0.0)
            mypos = self.my_abs_pos()
            path = planner.plan(mypos, goal, obstacles=world_model.get("opponents", []), obs_inflation=1.0)
            return ("dribble", path)
        else:
            if dist is not None and dist < 8.0:
                return ("approach", None)
            return ("support", None)

    def defensive_decision(self, dist):
        mypos = self.my_abs_pos()
        with world_lock:
            opponents = list(world_model.get("opponents", []))
            ball = world_model["ball"].copy()
        danger = None
        best_score = 1e9
        for ox,oy in opponents:
            d_goal = math.hypot(ox - self.own_goal_x, oy - 0.0)
            d_ball = math.hypot(ox - ball["x"], oy - ball["y"]) if ball.get("seen") else float('inf')
            score = d_goal + 0.5 * d_ball
            if score < best_score:
                best_score = score
                danger = (ox,oy)
        if danger and math.hypot(danger[0]-mypos[0], danger[1]-mypos[1]) < 20.0:
            return ("mark", danger)
        if ball.get("seen") and self.side_of_point(ball["x"]) == "own":
            if dist is None or dist > 1.5:
                return ("move_to_ball_prediction", None)
            else:
                return ("challenge", None)
        return ("hold", None)

    # ---------------- Acciones resultantes ----------------
    def execute_action(self, action, param, dist, ang):
        if action == "shoot":
            mypos = self.my_abs_pos()
            ang_goal = self.angle_to_point(mypos, (self.opp_goal_x, 0.0))
            self.send(f"(turn {ang_goal:.1f})")
            time.sleep(0.02)
            self.send("(kick 100 0)")
            self.release_ball_chaser_if_mine()
            return
        if action == "pass":
            u, pos, d = param
            ang = self.angle_to_point(self.my_abs_pos(), pos)
            power = min(100, max(30, int(d * 3.0)))
            self.send(f"(turn {ang:.1f})")
            time.sleep(0.02)
            self.send(f"(kick {power} 0)")
            self.release_ball_chaser_if_mine()
            return
        if action == "dribble":
            path = param
            if path and len(path) > 1:
                wx,wy = path[1]
                ang = self.angle_to_point(self.my_abs_pos(), (wx,wy))
                self.send(f"(turn {ang:.1f})")
                time.sleep(0.02)
                self.send("(dash 80)")
            else:
                self.send("(dash 30)")
            return
        if action == "approach":
            if dist is not None and ang is not None:
                if abs(ang) > 10:
                    self.send(f"(turn {ang})")
                else:
                    self.send("(dash 70)")
            else:
                self.send("(dash 50)")
            return
        if action == "support":
            tx, ty = self.get_abs_pos_for_unum(self.unum)
            ang = self.angle_to_point(self.my_abs_pos(), (tx, ty))
            if abs(ang) > 10:
                self.send(f"(turn {ang:.1f})")
            else:
                self.send("(dash 50)")
            return
        if action == "mark":
            ox,oy = param
            ang = self.angle_to_point(self.my_abs_pos(), (ox,oy))
            if abs(ang) > 10:
                self.send(f"(turn {ang})")
            else:
                self.send("(dash 60)")
            return
        if action == "move_to_ball_prediction":
            with world_lock:
                bx = world_model["ball"]["x"]
                by = world_model["ball"]["y"]
            ang = self.angle_to_point(self.my_abs_pos(), (bx,by))
            if abs(ang) > 10:
                self.send(f"(turn {ang})")
            else:
                self.send("(dash 90)")
            return
        if action == "challenge":
            if dist is not None and ang is not None:
                if abs(ang) > 10:
                    self.send(f"(turn {ang})")
                else:
                    self.send("(dash 100)")
            return
        if action == "hold":
            self.send("(turn 20)")
            return

    def chase(self, dist, ang, power):
        if dist < 0.9:
            self.attempt_high_level_action(dist)
            return
        if abs(ang) > 10:
            self.send(f"(turn {ang})")
        else:
            self.send(f"(dash {power})")

    def go_formation(self):
        self.send("(turn 30)")

    def attempt_high_level_action(self, dist):
        # Decisión ofensiva/defensiva basada en la POSICIÓN DEL BALÓN
        with world_lock:
            bm = world_model["ball"].copy()

        if bm.get("seen"):
            ball_side = self.side_of_point(bm["x"])
        else:
            ball_side = "neutral"  # si no vemos balón, no forzamos defensa

        # El portero siempre defiende
        if self.unum == self.GK or ball_side == "own":
            decision, param = self.defensive_decision(dist)
        else:
            decision, param = self.offensive_decision(dist)

        self.execute_action(decision, param, dist, None)

    def act(self):
        # 1) Predicción del filtro de Kalman local
        self.kf.predict()
        px, py, pvx, pvy = self.kf.state()
        now = time.time()

        # Actualizar world_model si tengo info fresca
        with world_lock:
            if now - self.last_ball_seen_time <= BALL_LOST_TIMEOUT:
                world_model["ball"].update({"x": px, "y": py, "vx": pvx, "vy": pvy, "t": now, "seen": True})

        # 2) Obtener datos del balón (local o global)
        dist = None
        ang = None
        
        # Intento usar mi visión local primero
        if self.ball:
            dist, ang = self.ball
        else:
            # Si no, uso el modelo global para calcular donde debería estar
            with world_lock:
                bm = world_model["ball"]
                if bm["seen"] and (now - bm["t"]) < BALL_LOST_TIMEOUT:
                    myx, myy = self.my_abs_pos()
                    dx = bm["x"] - myx
                    dy = bm["y"] - myy
                    dist = math.hypot(dx, dy)
                    # Calculo el ángulo relativo para poder girar hacia él
                    abs_ang = math.degrees(math.atan2(dy, dx))
                    facing = 0.0 if self.side == 'l' else 180.0 # Simplificación, idealmente usar body_angle si se tiene
                    ang = abs_ang - facing 
                    # Normalizar ángulo
                    while ang > 180: ang -= 360
                    while ang < -180: ang += 360

        # 3) Lógica de persecución
        
        # Si no sabemos dónde está el balón, volver a formación
        if dist is None:
            self.release_ball_chaser_if_mine()
            return self.go_formation()

        # Intentar pedir el turno
        is_chaser = self.request_ball_chaser(dist)

        # SI SOY EL PERSEGUIDOR:
        if is_chaser:
            # Lógica especial para portero y defensas (mantener zona)
            if self.unum == self.GK and dist > 15:
                # Si el portero tiene el turno pero el balón está lejos, soltarlo
                self.release_ball_chaser_if_mine()
                return self.go_formation()
            
            # Ejecutar persecución
            if self.close_to_ball():
                 self.perform_kick_logic()
            else:
                 # Si tengo ángulo calculado, lo uso, si no, dash
                 turn_ang = ang if ang is not None else 0
                 self.chase(dist, turn_ang, 100)
            return
        
        if dist < 20.0:
            # Modo "Soporte": Solo girar hacia el balón, no correr
            if ang is not None and abs(ang) > 15:
                self.send(f"(turn {ang:.1f})")
            else:
                # Si ya estoy mirando el balón, esperar (o moverse muy lento para ajustar)
                time.sleep(0.05) 
        else:
            # Si está muy lejos, entonces sí vuelvo a formación
            self.go_formation()
            
        time.sleep(0.03)


    # LOOP PRINCIPAL
    def run(self):
        self.send(f"(init {TEAM})")
        buf = ""
        t = time.time()
        TIMEOUT_INIT = 5.0
        while time.time() - t < TIMEOUT_INIT and not stop_event.is_set():
            try:
                d, _ = self.sock.recvfrom(1024)
                buf += d.decode(errors="ignore")
                m = re.search(r"\(init\s+([lr])\s+(\d+)", buf)
                if m:
                    self.side = m.group(1)
                    self.unum = int(m.group(2))
                    break
            except socket.timeout:
                continue
            except:
                break

        if self.unum is None:
            print(f"[Jugador {self.idx}] ❌ ERROR: Falló la inicialización (no se recibió unum del servidor).")
            try:
                self.sock.close()
            except:
                pass
            return

        self.configure_field_sides()
        print(f"[Jugador {self.unum}] listo lado='{self.side}' | own_goal_x={self.own_goal_x} | opp_goal_x={self.opp_goal_x}")
        self.place_spawn()

        while not stop_event.is_set():
            try:
                d, _ = self.sock.recvfrom(4096)
                msg = d.decode(errors="ignore")
                if "(see" in msg:
                    self.see_ball(msg)
                # cada iteración intentamos actuar
                self.act()
            except socket.timeout:
                if stop_event.is_set():
                    break
                self.send("(turn 20)")
            except:
                break

        try:
            self.sock.close()
        except:
            pass

# ----------------------- MAIN -------------------------
def main():
    pos = load_positions()
    print("\nFormación cargada:\n", pos, "\n")
    threads = []
    for i in range(1, PLAYERS + 1):
        p = Player(i, pos)
        t = threading.Thread(target=p.run, daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.4)
    print("\n🟢 Equipo iniciado y en cancha\n")
    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🟠 Detención solicitada (Ctrl+C). Cerrando equipo...")
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)
        print("✅ Equipo detenido. Saliendo.")

if __name__ == "__main__":
    main()
