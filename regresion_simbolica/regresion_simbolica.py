#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regresión simbólica con algoritmo genético y árboles de expresión.

Uso:
    1) Generar el CSV objetivo (puntos X, Y a los que se quiere llegar):
         python regresion_simbolica.py generar --funcion seno --salida datos.csv

    2) Evolucionar expresiones para aproximar el CSV (con gráfica en vivo):
         python regresion_simbolica.py evolucionar datos.csv

El algoritmo genético está implementado a mano (sin DEAP). Solo se usa numpy
para evaluar rápido y matplotlib para graficar.

Técnicas incluidas:
  - Árboles de expresión (operadores + - * /, sin, cos; terminales x y constantes)
  - Población inicial "ramped half-and-half" (full + grow)
  - Selección por torneo + elitismo
  - Cruza de subárboles y mutación (subárbol, punto, constantes)
  - Poda   : recorta ramas que exceden la profundidad máxima
  - Reparación : corrige aridad inválida, operadores desconocidos y constantes fuera de rango
  - Reconstrucción : simplifica el árbol (plegado de constantes, x*1, x+0, x*0, ...)
  - Errores: MSE, RMSE, NMSE (dB) y R²
"""

import argparse
import csv
import random
import sys

import numpy as np

np.seterr(all="ignore")

LIM = 1e12          # límite para evitar desbordes
CONST_MAX = 100.0   # rango permitido de constantes


# ---------------------------------------------------------------------------
# Operadores
# ---------------------------------------------------------------------------
def div_protegida(a, b):
    ok = np.abs(b) > 1e-6
    return np.where(ok, a / np.where(ok, b, 1.0), 1.0)


OPS2 = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": div_protegida,
}
OPS1 = {
    "sin": np.sin,
    "cos": np.cos,
}


# ---------------------------------------------------------------------------
# Árbol de expresión
# ---------------------------------------------------------------------------
class Nodo:
    __slots__ = ("tipo", "valor", "hijos")  # tipo: 'op' | 'var' | 'const'

    def __init__(self, tipo, valor, hijos=None):
        self.tipo = tipo
        self.valor = valor
        self.hijos = hijos if hijos is not None else []


def const_aleatoria():
    return round(random.uniform(-5, 5), 2)


def terminal():
    if random.random() < 0.7:
        return Nodo("var", "x")
    return Nodo("const", const_aleatoria())


def crear(prof, metodo):
    """Crea un árbol aleatorio. metodo: 'full' o 'grow'."""
    if prof <= 0 or (metodo == "grow" and random.random() < 0.3):
        return terminal()
    if random.random() < 0.85:
        op = random.choice(list(OPS2))
        return Nodo("op", op, [crear(prof - 1, metodo) for _ in range(2)])
    op = random.choice(list(OPS1))
    return Nodo("op", op, [crear(prof - 1, metodo)])


def copiar(n):
    return Nodo(n.tipo, n.valor, [copiar(h) for h in n.hijos])


def evaluar(n, x):
    if n.tipo == "const":
        return np.full_like(x, n.valor, dtype=float)
    if n.tipo == "var":
        return x
    if len(n.hijos) == 2:
        r = OPS2[n.valor](evaluar(n.hijos[0], x), evaluar(n.hijos[1], x))
    else:
        r = OPS1[n.valor](evaluar(n.hijos[0], x))
    return np.clip(np.nan_to_num(r, nan=0.0, posinf=LIM, neginf=-LIM), -LIM, LIM)


def a_cadena(n):
    if n.tipo == "var":
        return "x"
    if n.tipo == "const":
        return f"{n.valor:.4g}"
    if len(n.hijos) == 2:
        return f"({a_cadena(n.hijos[0])} {n.valor} {a_cadena(n.hijos[1])})"
    return f"{n.valor}({a_cadena(n.hijos[0])})"


def tamano(n):
    return 1 + sum(tamano(h) for h in n.hijos)


def profundidad(n):
    return 1 + max((profundidad(h) for h in n.hijos), default=0)


def listar(n, padre=None, i=0, out=None):
    """Lista (padre, índice, nodo) de todos los nodos del árbol."""
    if out is None:
        out = []
    out.append((padre, i, n))
    for j, h in enumerate(n.hijos):
        listar(h, n, j, out)
    return out


def reemplazar(raiz, padre, i, nuevo):
    if padre is None:
        return nuevo
    padre.hijos[i] = nuevo
    return raiz


# ---------------------------------------------------------------------------
# Poda, reparación y reconstrucción
# ---------------------------------------------------------------------------
def podar(n, max_prof, prof=1):
    """PODA: cualquier operador que quede en el nivel máximo se corta y se
    sustituye por un terminal."""
    if n.tipo != "op":
        return n
    if prof >= max_prof:
        return terminal()
    n.hijos = [podar(h, max_prof, prof + 1) for h in n.hijos]
    return n


def reparar(n):
    """REPARACIÓN: asegura aridad correcta, operadores válidos y constantes
    dentro del rango permitido."""
    if n.tipo == "op":
        if n.valor in OPS2:
            aridad = 2
        elif n.valor in OPS1:
            aridad = 1
        else:                       # operador desconocido -> terminal
            return terminal()
        hijos = n.hijos[:aridad]                      # sobran hijos: se truncan
        while len(hijos) < aridad:                    # faltan hijos: se completan
            hijos.append(terminal())
        n.hijos = [reparar(h) for h in hijos]
    elif n.tipo == "const":
        v = n.valor
        if not np.isfinite(v):
            v = const_aleatoria()
        n.valor = float(np.clip(v, -CONST_MAX, CONST_MAX))
    return n


def _es(n, v):
    return n.tipo == "const" and abs(n.valor - v) < 1e-12


def _igual(a, b):
    return (a.tipo == b.tipo and a.valor == b.valor and len(a.hijos) == len(b.hijos)
            and all(_igual(u, v) for u, v in zip(a.hijos, b.hijos)))


def reconstruir(n):
    """RECONSTRUCCIÓN: simplifica el árbol (plegado de constantes e identidades
    algebraicas) y lo vuelve a armar más compacto."""
    if n.tipo != "op":
        return n
    n.hijos = [reconstruir(h) for h in n.hijos]
    if all(h.tipo == "const" for h in n.hijos):
        v = float(evaluar(n, np.array([0.0]))[0])
        return Nodo("const", float(np.clip(v, -CONST_MAX, CONST_MAX)))
    if len(n.hijos) == 2:
        a, b = n.hijos
        op = n.valor
        if op == "+":
            if _es(a, 0): return b
            if _es(b, 0): return a
        elif op == "-":
            if _es(b, 0): return a
            if _igual(a, b): return Nodo("const", 0.0)
        elif op == "*":
            if _es(a, 1): return b
            if _es(b, 1): return a
            if _es(a, 0) or _es(b, 0): return Nodo("const", 0.0)
        elif op == "/":
            if _es(b, 1): return a
            if _igual(a, b) or _es(b, 0): return Nodo("const", 1.0)  # valor de la división protegida
    return n


def normalizar(n, max_prof):
    n = reparar(n)
    n = podar(n, max_prof)
    n = reconstruir(n)
    return n


# ---------------------------------------------------------------------------
# Métricas de error
# ---------------------------------------------------------------------------
EPS = 1e-12


def metricas(y, yh):
    """y = valores esperados (CSV), yh = valores del árbol."""
    e = yh - y
    ss_res = float(np.sum(e ** 2))
    mse = ss_res / len(y)
    rmse = float(np.sqrt(mse))
    # NMSE en dB, según la fórmula: 10 log( sum(|Y|-|X|)^2 / sum|X|^2 )
    nmse = 10.0 * np.log10((np.sum((np.abs(yh) - np.abs(y)) ** 2) + EPS) / (np.sum(y ** 2) + EPS))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + EPS)
    return {"MSE": mse, "RMSE": rmse, "NMSE": float(nmse), "R2": float(r2)}


def puntaje(m, tam, cfg):
    """Valor a MINIMIZAR para seleccionar. Incluye una pequeña penalización por
    tamaño (parsimonia) para evitar árboles gigantes."""
    k = cfg.metrica
    if k == "MSE":
        base = m["MSE"] * (1 + cfg.parsimonia * tam)
    elif k == "RMSE":
        base = m["RMSE"] * (1 + cfg.parsimonia * tam)
    elif k == "R2":
        base = (1.0 - m["R2"]) * (1 + cfg.parsimonia * tam)
    else:  # NMSE (dB, más negativo = mejor)
        base = m["NMSE"] + 50 * cfg.parsimonia * tam
    return base if np.isfinite(base) else float("inf")


# ---------------------------------------------------------------------------
# Operadores genéticos
# ---------------------------------------------------------------------------
def torneo(pob, punt, k):
    idx = random.sample(range(len(pob)), min(k, len(pob)))
    return pob[min(idx, key=lambda i: punt[i])]


def cruza(a, b):
    """Cruza de subárboles: intercambia una rama aleatoria entre los padres."""
    h1, h2 = copiar(a), copiar(b)
    p1, i1, n1 = random.choice(listar(h1))
    p2, i2, n2 = random.choice(listar(h2))
    h1 = reemplazar(h1, p1, i1, n2)
    h2 = reemplazar(h2, p2, i2, n1)
    return h1, h2


def mutar(raiz):
    nodos = listar(raiz)
    padre, i, n = random.choice(nodos)
    r = random.random()
    if r < 0.4:                                   # mutación de subárbol
        raiz = reemplazar(raiz, padre, i, crear(random.randint(0, 2), "grow"))
    elif r < 0.75:                                # mutación de punto
        if n.tipo == "const":
            n.valor += random.gauss(0, 1)
        elif n.tipo == "var":
            n.tipo, n.valor = "const", const_aleatoria()
        elif len(n.hijos) == 2:
            n.valor = random.choice(list(OPS2))
        else:
            n.valor = random.choice(list(OPS1))
    else:                                         # ajuste fino de constantes
        for _, _, m in nodos:
            if m.tipo == "const":
                m.valor += random.gauss(0, 0.3)
    return raiz


def refinar(ind, x, y, cfg, intentos):
    """Ajuste fino de constantes (búsqueda local) sobre un individuo. Acelera
    mucho encontrar valores como 6.87 u 11 una vez que la forma del árbol es buena."""
    consts = [n for _, _, n in listar(ind) if n.tipo == "const"]
    if not consts:
        return ind
    mejor = puntaje(metricas(y, evaluar(ind, x)), tamano(ind), cfg)
    for _ in range(intentos):
        n = random.choice(consts)
        viejo = n.valor
        paso = random.choice([1.0, 0.1, 0.01, 0.001]) * max(1.0, abs(viejo))
        n.valor = float(np.clip(viejo + random.gauss(0, paso), -CONST_MAX, CONST_MAX))
        nuevo = puntaje(metricas(y, evaluar(ind, x)), tamano(ind), cfg)
        if nuevo < mejor:
            mejor = nuevo
        else:
            n.valor = viejo
    return ind


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
FUNCIONES_CSV = {
    "cuadratica": (lambda x: x ** 2 + 2 * x + 1, (-5, 5)),
    "cubica": (lambda x: x ** 3 - 2 * x, (-3, 3)),
    "seno": (lambda x: np.sin(x) + 0.5 * x, (-6, 6)),
    "compleja": (lambda x: x * np.sin(x) + x ** 2 / 10, (-6, 6)),
}


def compilar_expresion(texto):
    """Convierte un texto como '6.87 + 11 * cos(7.23*x^3)' en una función de x.
    Solo permite x, números y funciones matemáticas conocidas (sin eval libre)."""
    permitido = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
                 "sqrt": np.sqrt, "log": np.log, "abs": np.abs, "pi": np.pi}
    texto = texto.replace("^", "**")
    try:
        codigo = compile(texto, "<expresion>", "eval")
    except SyntaxError as err:
        sys.exit(f"Expresión inválida: {err}")
    for nombre in codigo.co_names:
        if nombre != "x" and nombre not in permitido:
            sys.exit(f"'{nombre}' no está permitido. Usa: x, {', '.join(permitido)}")

    def f(x):
        r = eval(codigo, {"__builtins__": {}}, {**permitido, "x": x})
        return np.broadcast_to(np.asarray(r, dtype=float), x.shape).copy()
    return f


def generar_csv(args):
    if args.expresion:
        f, nombre = compilar_expresion(args.expresion), args.expresion
        a, b = args.rango
    else:
        nombre = args.funcion or "seno"
        f, (a, b) = FUNCIONES_CSV[nombre]
        if args.rango_dado:
            a, b = args.rango
    rng = np.random.default_rng(args.semilla)
    x = np.linspace(a, b, args.puntos)
    y = f(x) + rng.normal(0, args.ruido, size=x.shape)
    with open(args.salida, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y"])
        for xi, yi in zip(x, y):
            w.writerow([f"{xi:.6f}", f"{yi:.6f}"])
    print(f"CSV generado: {args.salida} ({args.puntos} puntos, función '{nombre}')")
    if not args.sin_grafica:
        import matplotlib.pyplot as plt
        plt.scatter(x, y, s=12)
        plt.title(f"Objetivo: {nombre}")
        plt.xlabel("x"); plt.ylabel("y"); plt.grid(alpha=.3)
        plt.show()


def leer_csv(ruta):
    xs, ys = [], []
    with open(ruta, newline="") as fh:
        for fila in csv.reader(fh):
            try:
                xs.append(float(fila[0])); ys.append(float(fila[1]))
            except (ValueError, IndexError):
                continue  # encabezado o fila inválida
    if len(xs) < 3:
        sys.exit("El CSV necesita al menos 3 filas numéricas con columnas x,y.")
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Evolución
# ---------------------------------------------------------------------------
def evolucionar(args):
    if args.semilla is not None:
        random.seed(args.semilla)
    if args.sin_grafica:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = leer_csv(args.csv)
    xs_den = np.linspace(x.min(), x.max(), 300)
    orden = np.argsort(x)

    # Población inicial: ramped half-and-half
    pob = []
    for i in range(args.poblacion):
        prof = 2 + i % (args.prof_max - 1)
        pob.append(normalizar(crear(prof, "full" if i % 2 else "grow"), args.prof_max))

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    hist = {"MSE": [], "RMSE": [], "NMSE": [], "R2": []}
    mejor_global, mejor_punt_global = None, float("inf")

    for gen in range(args.generaciones + 1):
        # --- ajuste fino de constantes en los mejores (opcional) ---
        if args.refinar > 0 and gen > 0:
            pts = [puntaje(metricas(y, evaluar(i, x)), tamano(i), args) for i in pob]
            for i in sorted(range(len(pob)), key=lambda k: pts[k])[:args.refinar_top]:
                pob[i] = refinar(pob[i], x, y, args, args.refinar)

        # --- evaluación ---
        ys_h, mets, punt = [], [], []
        for ind in pob:
            yh = evaluar(ind, x)
            m = metricas(y, yh)
            ys_h.append(yh); mets.append(m)
            punt.append(puntaje(m, tamano(ind), args))
        orden_pob = sorted(range(len(pob)), key=lambda i: punt[i])
        b = orden_pob[0]
        if punt[b] < mejor_punt_global:
            mejor_global, mejor_punt_global = copiar(pob[b]), punt[b]
            mejor_m = mets[b]
        for k in hist:
            hist[k].append(mejor_m[k])

        print(f"Gen {gen:3d} | MSE={mejor_m['MSE']:.5g} RMSE={mejor_m['RMSE']:.5g} "
              f"NMSE={mejor_m['NMSE']:.2f} dB R2={mejor_m['R2']:.5f} | "
              f"nodos={tamano(mejor_global)} | {a_cadena(mejor_global)}")

        # --- gráfica en vivo ---
        if gen % args.cada == 0 or gen == args.generaciones:
            ax1.cla(); ax2.cla()
            ax1.scatter(x, y, s=14, color="black", label="CSV (esperado)", zorder=3)
            for i in orden_pob[1:4]:  # mejores siguientes, tenues
                ax1.plot(xs_den, evaluar(pob[i], xs_den), alpha=.25, color="tab:blue", lw=1)
            ax1.plot(xs_den, evaluar(mejor_global, xs_den), color="tab:red", lw=2.2,
                     label="Mejor expresión")
            ax1.set_ylim(y.min() - 0.3 * (np.ptp(y) + 1), y.max() + 0.3 * (np.ptp(y) + 1))
            ax1.set_title(f"Generación {gen}")
            ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.grid(alpha=.3)
            ax1.legend(loc="upper left")
            expr = a_cadena(mejor_global)
            ax1.text(0.01, 0.01, "f(x) = " + (expr if len(expr) < 90 else expr[:87] + "..."),
                     transform=ax1.transAxes, fontsize=8, va="bottom",
                     bbox=dict(boxstyle="round", fc="white", alpha=.8))

            ax2.plot(hist["MSE"], label="MSE", color="tab:red")
            ax2.plot(hist["RMSE"], label="RMSE", color="tab:orange")
            ax2.set_yscale("log"); ax2.set_xlabel("Generación"); ax2.set_ylabel("MSE / RMSE (log)")
            ax2.grid(alpha=.3)
            ax3 = ax2.twinx()
            ax3.plot(hist["R2"], label="R²", color="tab:green", ls="--")
            ax3.set_ylabel("R²"); ax3.set_ylim(min(0, min(hist["R2"])) - .05, 1.05)
            h1, l1 = ax2.get_legend_handles_labels()
            h2, l2 = ax3.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, loc="center right")
            ax2.set_title(f"MSE={mejor_m['MSE']:.4g}  RMSE={mejor_m['RMSE']:.4g}  "
                          f"NMSE={mejor_m['NMSE']:.1f} dB  R²={mejor_m['R2']:.4f}")
            fig.tight_layout()
            if not args.sin_grafica:
                plt.pause(args.pausa)
            ax3.remove()

        if mejor_m["MSE"] <= args.tolerancia or gen == args.generaciones:
            break

        # --- nueva generación ---
        nueva = [copiar(pob[i]) for i in orden_pob[:args.elite]]   # elitismo
        n_inm = int(args.inmigrantes * args.poblacion)      # sangre nueva: evita estancarse
        for _ in range(n_inm):
            nueva.append(normalizar(crear(random.randint(2, args.prof_max), "grow"), args.prof_max))
        while len(nueva) < args.poblacion:
            p1 = torneo(pob, punt, args.torneo)
            p2 = torneo(pob, punt, args.torneo)
            if random.random() < args.p_cruza:
                h1, h2 = cruza(p1, p2)
            else:
                h1, h2 = copiar(p1), copiar(p2)
            for h in (h1, h2):
                if random.random() < args.p_mutacion:
                    h = mutar(h)
                nueva.append(normalizar(h, args.prof_max))          # repara + poda + reconstruye
                if len(nueva) >= args.poblacion:
                    break
        pob = nueva

    print("\n=== Resultado final ===")
    print("Expresión :", a_cadena(mejor_global))
    for k, v in mejor_m.items():
        print(f"{k:5s}: {v:.6g}" + (" dB" if k == "NMSE" else ""))
    fig.savefig(args.imagen, dpi=130)
    print(f"Gráfica final guardada en {args.imagen}")
    if not args.sin_grafica:
        plt.ioff()
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Regresión simbólica con algoritmo genético")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generar", help="genera el CSV objetivo (x,y)")
    g.add_argument("--funcion", choices=FUNCIONES_CSV, default=None,
                   help="función predefinida (si no das --expresion)")
    g.add_argument("--expresion", default=None,
                   help="función propia en x, ej: \"6.87 + 11 * cos(7.23*x^3)\"")
    g.add_argument("--rango", type=float, nargs=2, metavar=("XMIN", "XMAX"), default=None,
                   help="intervalo de x (por defecto -1.5 1.5 con --expresion)")
    g.add_argument("--salida", default="datos.csv")
    g.add_argument("--puntos", type=int, default=200)
    g.add_argument("--ruido", type=float, default=0.0, help="desviación del ruido gaussiano")
    g.add_argument("--semilla", type=int, default=1)
    g.add_argument("--sin-grafica", action="store_true")
    g.set_defaults(fn=generar_csv)

    e = sub.add_parser("evolucionar", help="evoluciona expresiones hacia el CSV")
    e.add_argument("csv")
    e.add_argument("--poblacion", type=int, default=300)
    e.add_argument("--generaciones", type=int, default=100)
    e.add_argument("--prof-max", type=int, default=8, help="profundidad máxima (poda)")
    e.add_argument("--inmigrantes", type=float, default=0.1,
                   help="fracción de la población que se reemplaza por árboles nuevos cada generación")
    e.add_argument("--refinar", type=int, default=40,
                   help="intentos de ajuste fino de constantes por individuo (0 = desactivar)")
    e.add_argument("--refinar-top", type=int, default=10,
                   help="a cuántos de los mejores se les ajustan las constantes")
    e.add_argument("--torneo", type=int, default=5)
    e.add_argument("--elite", type=int, default=3)
    e.add_argument("--p-cruza", type=float, default=0.8)
    e.add_argument("--p-mutacion", type=float, default=0.3)
    e.add_argument("--metrica", choices=["MSE", "RMSE", "NMSE", "R2"], default="MSE",
                   help="error usado para seleccionar (las 4 se reportan siempre)")
    e.add_argument("--parsimonia", type=float, default=0.002, help="penalización por tamaño")
    e.add_argument("--tolerancia", type=float, default=1e-6, help="MSE para detenerse")
    e.add_argument("--cada", type=int, default=1, help="graficar cada N generaciones")
    e.add_argument("--pausa", type=float, default=0.05)
    e.add_argument("--semilla", type=int, default=None)
    e.add_argument("--imagen", default="resultado.png")
    e.add_argument("--sin-grafica", action="store_true", help="no abre ventana (solo guarda PNG)")
    e.set_defaults(fn=evolucionar)

    args = p.parse_args()
    if args.cmd == "generar":
        args.rango_dado = args.rango is not None
        if args.rango is None:
            args.rango = (-1.5, 1.5)
    args.fn(args)


if __name__ == "__main__":
    main()