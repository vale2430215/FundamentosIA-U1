#!/usr/bin/env python3
"""
Intérprete de expresiones matemáticas.
 
A partir de una ecuación escrita como texto, por ejemplo:
 
    sqrt(2*(x^2)+5*x-3)
 
el programa:
  1. Tokeniza la cadena (análisis léxico).
  2. Construye el árbol de expresión con un parser descendente recursivo
     (análisis sintáctico) respetando precedencia y asociatividad.
  3. Muestra el árbol en consola (ASCII) y como figura (matplotlib).
  4. Evalúa la expresión con numpy y grafica la función y = f(x).
 
Uso:
    python interprete.py "sqrt(2*(x^2)+5*x-3)"
    python interprete.py "sin(x)/x" --xmin -20 --xmax 20
    python interprete.py "x^3 - 2x" --save salida.png --no-show
    python interprete.py            # modo interactivo
 
Dependencias: numpy, matplotlib
"""
 
from __future__ import annotations
 
import argparse
import re
import sys
from dataclasses import dataclass, field
 
import numpy as np
 
# --------------------------------------------------------------------------
# Funciones y constantes soportadas
# --------------------------------------------------------------------------
FUNCIONES = {
    "sqrt": np.sqrt,
    "cbrt": np.cbrt,
    "abs": np.abs,
    "exp": np.exp,
    "ln": np.log,
    "log": np.log10,
    "log2": np.log2,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
}
 
CONSTANTES = {
    "pi": np.pi,
    "e": np.e,
}
 
 
class ErrorDeExpresion(Exception):
    """Error de sintaxis o de evaluación en la expresión."""
 
 
# --------------------------------------------------------------------------
# 1. Análisis léxico
# --------------------------------------------------------------------------
@dataclass
class Token:
    tipo: str   # NUM, ID, OP, LP, RP
    valor: str
    pos: int
 
 
_PATRON = re.compile(
    r"\s*(?:(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<id>[A-Za-z_]\w*)"
    r"|(?P<otro>.))"
)
 
 
def tokenizar(texto: str) -> list[Token]:
    texto = texto.replace("**", "^").replace("×", "*").replace("÷", "/")
    tokens: list[Token] = []
    pos = 0
    while pos < len(texto):
        m = _PATRON.match(texto, pos)
        if m is None or m.end() == pos:
            break
        pos = m.end()
        if m.group("num") is not None:
            tokens.append(Token("NUM", m.group("num"), m.start("num")))
        elif m.group("id") is not None:
            tokens.append(Token("ID", m.group("id"), m.start("id")))
        else:
            c = m.group("otro")
            if c.isspace():
                continue
            if c in "+-*/^":
                tokens.append(Token("OP", c, m.start("otro")))
            elif c == "(":
                tokens.append(Token("LP", c, m.start("otro")))
            elif c == ")":
                tokens.append(Token("RP", c, m.start("otro")))
            else:
                raise ErrorDeExpresion(
                    f"Carácter no válido {c!r} en la posición {m.start('otro')}"
                )
    return _multiplicacion_implicita(tokens)
 
 
def _multiplicacion_implicita(tokens: list[Token]) -> list[Token]:
    """Inserta '*' en casos como 2x, 3(x+1), (x+1)(x-1), 2sin(x)."""
    salida: list[Token] = []
    for t in tokens:
        if salida:
            prev = salida[-1]
            prev_es_valor = (
                prev.tipo in ("NUM", "RP")
                or (prev.tipo == "ID" and prev.valor not in FUNCIONES)
            )
            if prev_es_valor and t.tipo in ("NUM", "ID", "LP"):
                salida.append(Token("OP", "*", t.pos))
        salida.append(t)
    return salida
 
 
# --------------------------------------------------------------------------
# 2. Árbol de expresión
# --------------------------------------------------------------------------
@dataclass
class Nodo:
    tipo: str                      # 'num', 'var', 'const', 'bin', 'neg', 'func'
    valor: object                  # número, nombre u operador
    hijos: list["Nodo"] = field(default_factory=list)
 
    @property
    def etiqueta(self) -> str:
        if self.tipo == "num":
            v = float(self.valor)
            return str(int(v)) if v.is_integer() else f"{v:g}"
        if self.tipo == "neg":
            return "-(unario)"
        if self.tipo == "func":
            return f"{self.valor}()"
        return str(self.valor)
 
 
# --------------------------------------------------------------------------
# 3. Análisis sintáctico (descendente recursivo)
#
#   expr   := term (('+' | '-') term)*
#   term   := unary (('*' | '/') unary)*
#   unary  := ('-' | '+') unary | power
#   power  := atom ('^' unary)?          (asociativa a la derecha)
#   atom   := NUM | ID | ID '(' expr ')' | '(' expr ')'
# --------------------------------------------------------------------------
class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.i = 0
 
    def _ver(self) -> Token | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None
 
    def _tomar(self) -> Token:
        t = self._ver()
        if t is None:
            raise ErrorDeExpresion("La expresión terminó de forma inesperada")
        self.i += 1
        return t
 
    def _es_op(self, *ops: str) -> bool:
        t = self._ver()
        return t is not None and t.tipo == "OP" and t.valor in ops
 
    def parsear(self) -> Nodo:
        if not self.tokens:
            raise ErrorDeExpresion("La expresión está vacía")
        nodo = self._expr()
        t = self._ver()
        if t is not None:
            raise ErrorDeExpresion(
                f"Símbolo inesperado {t.valor!r} en la posición {t.pos}"
            )
        return nodo
 
    def _expr(self) -> Nodo:
        nodo = self._term()
        while self._es_op("+", "-"):
            op = self._tomar().valor
            nodo = Nodo("bin", op, [nodo, self._term()])
        return nodo
 
    def _term(self) -> Nodo:
        nodo = self._unary()
        while self._es_op("*", "/"):
            op = self._tomar().valor
            nodo = Nodo("bin", op, [nodo, self._unary()])
        return nodo
 
    def _unary(self) -> Nodo:
        if self._es_op("-"):
            self._tomar()
            return Nodo("neg", "-", [self._unary()])
        if self._es_op("+"):
            self._tomar()
            return self._unary()
        return self._power()
 
    def _power(self) -> Nodo:
        base = self._atom()
        if self._es_op("^"):
            self._tomar()
            exponente = self._unary()   # permite 2^-x y 2^3^2 = 2^(3^2)
            return Nodo("bin", "^", [base, exponente])
        return base
 
    def _atom(self) -> Nodo:
        t = self._tomar()
        if t.tipo == "NUM":
            return Nodo("num", float(t.valor))
        if t.tipo == "ID":
            nombre = t.valor
            if nombre in FUNCIONES:
                sig = self._ver()
                if sig is None or sig.tipo != "LP":
                    raise ErrorDeExpresion(
                        f"La función {nombre!r} requiere paréntesis: {nombre}(...)"
                    )
                self._tomar()
                arg = self._expr()
                self._cerrar_parentesis()
                return Nodo("func", nombre, [arg])
            if nombre in CONSTANTES:
                return Nodo("const", nombre)
            return Nodo("var", nombre)
        if t.tipo == "LP":
            nodo = self._expr()
            self._cerrar_parentesis()
            return nodo
        raise ErrorDeExpresion(
            f"Símbolo inesperado {t.valor!r} en la posición {t.pos}"
        )
 
    def _cerrar_parentesis(self) -> None:
        t = self._ver()
        if t is None or t.tipo != "RP":
            raise ErrorDeExpresion("Falta cerrar un paréntesis ')'")
        self._tomar()
 
 
def construir_arbol(texto: str) -> Nodo:
    return Parser(tokenizar(texto)).parsear()
 
 
# --------------------------------------------------------------------------
# 4. Utilidades sobre el árbol
# --------------------------------------------------------------------------
def variables(nodo: Nodo) -> set[str]:
    if nodo.tipo == "var":
        return {nodo.valor}
    resultado: set[str] = set()
    for h in nodo.hijos:
        resultado |= variables(h)
    return resultado
 
 
def a_texto(nodo: Nodo) -> str:
    """Reconstruye la expresión (completamente parentizada) desde el árbol."""
    if nodo.tipo in ("num", "var", "const"):
        return nodo.etiqueta
    if nodo.tipo == "neg":
        return f"(-{a_texto(nodo.hijos[0])})"
    if nodo.tipo == "func":
        return f"{nodo.valor}({a_texto(nodo.hijos[0])})"
    izq, der = nodo.hijos
    return f"({a_texto(izq)} {nodo.valor} {a_texto(der)})"
 
 
def arbol_ascii(nodo: Nodo, prefijo: str = "", es_ultimo: bool = True,
                es_raiz: bool = True) -> str:
    if es_raiz:
        lineas = [nodo.etiqueta]
        nuevo_prefijo = ""
    else:
        lineas = [prefijo + ("└── " if es_ultimo else "├── ") + nodo.etiqueta]
        nuevo_prefijo = prefijo + ("    " if es_ultimo else "│   ")
    for k, h in enumerate(nodo.hijos):
        lineas.append(
            arbol_ascii(h, nuevo_prefijo, k == len(nodo.hijos) - 1, False)
        )
    return "\n".join(lineas)
 
 
def evaluar(nodo: Nodo, x, var: str = "x"):
    """Evalúa el árbol; x puede ser un escalar o un arreglo de numpy."""
    t = nodo.tipo
    if t == "num":
        return nodo.valor
    if t == "const":
        return CONSTANTES[nodo.valor]
    if t == "var":
        if nodo.valor != var:
            raise ErrorDeExpresion(f"Variable desconocida: {nodo.valor!r}")
        return x
    if t == "neg":
        return -evaluar(nodo.hijos[0], x, var)
    if t == "func":
        return FUNCIONES[nodo.valor](evaluar(nodo.hijos[0], x, var))
    a = evaluar(nodo.hijos[0], x, var)
    b = evaluar(nodo.hijos[1], x, var)
    op = nodo.valor
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b
    if op == "^":
        return np.power(a, b)
    raise ErrorDeExpresion(f"Operador desconocido: {op}")
 
 
# --------------------------------------------------------------------------
# 5. Dibujo del árbol y graficación
# --------------------------------------------------------------------------
_COLORES = {
    "num": "#cfe8ff",
    "var": "#d6f5d6",
    "const": "#e8d9ff",
    "bin": "#ffe3b3",
    "neg": "#ffe3b3",
    "func": "#ffd0d0",
}
 
 
def _posiciones(nodo: Nodo, profundidad: int, siguiente_x: list[float],
                pos: dict[int, tuple[float, int]]) -> float:
    if not nodo.hijos:
        x = siguiente_x[0]
        siguiente_x[0] += 1
    else:
        xs = [_posiciones(h, profundidad + 1, siguiente_x, pos)
              for h in nodo.hijos]
        x = sum(xs) / len(xs)
    pos[id(nodo)] = (x, profundidad)
    return x
 
 
def dibujar_arbol(nodo: Nodo, ax) -> None:
    pos: dict[int, tuple[float, int]] = {}
    _posiciones(nodo, 0, [0.0], pos)
 
    def aristas(n: Nodo):
        x0, y0 = pos[id(n)]
        for h in n.hijos:
            x1, y1 = pos[id(h)]
            ax.plot([x0, x1], [-y0, -y1], color="#666666", lw=1.2, zorder=1)
            aristas(h)
 
    def nodos(n: Nodo):
        x, y = pos[id(n)]
        ax.text(
            x, -y, n.etiqueta, ha="center", va="center", fontsize=10,
            zorder=2,
            bbox=dict(boxstyle="round,pad=0.35", fc=_COLORES[n.tipo],
                      ec="#444444"),
        )
        for h in n.hijos:
            nodos(h)
 
    aristas(nodo)
    nodos(nodo)
    ax.set_title("Árbol de expresión")
    ax.margins(0.12)
    ax.axis("off")
 
 
def graficar_funcion(nodo: Nodo, ax, texto: str, xmin: float, xmax: float,
                     puntos: int = 4000) -> None:
    vars_ = variables(nodo)
    if len(vars_) > 1:
        raise ErrorDeExpresion(
            f"Solo se puede graficar con una variable; se encontraron: "
            f"{', '.join(sorted(vars_))}"
        )
    var = next(iter(vars_), "x")
 
    x = np.linspace(xmin, xmax, puntos)
    with np.errstate(all="ignore"):
        y = evaluar(nodo, x, var)
    y = np.broadcast_to(np.asarray(y, dtype=float), x.shape).copy()
    y[~np.isfinite(y)] = np.nan
    y[np.abs(y) > 1e6] = np.nan        # evita líneas verticales en asíntotas
 
    ax.axhline(0, color="#888888", lw=0.8)
    ax.axvline(0, color="#888888", lw=0.8)
    ax.plot(x, y, color="#1f77b4", lw=2)
    ax.set_xlabel(var)
    ax.set_ylabel(f"f({var})")
    ax.set_title(f"y = {texto}")
    ax.grid(True, alpha=0.3)
 
    finitos = y[np.isfinite(y)]
    if finitos.size == 0:
        ax.text(0.5, 0.5, "La expresión no tiene valores reales\n"
                "en el intervalo indicado", transform=ax.transAxes,
                ha="center", va="center")
        return
    lo, hi = np.percentile(finitos, [1, 99])
    if finitos.max() - finitos.min() > 10 * max(hi - lo, 1e-9):
        pad = 0.5 * (hi - lo) + 1e-9      # recorta para ver bien la forma
        ax.set_ylim(lo - pad, hi + pad)
 
 
def mostrar(texto: str, xmin: float, xmax: float, guardar: str | None,
            mostrar_ventana: bool) -> None:
    import matplotlib.pyplot as plt
 
    nodo = construir_arbol(texto)
    print("\nÁrbol de expresión:")
    print(arbol_ascii(nodo))
    print(f"\nForma parentizada: {a_texto(nodo)}")
 
    fig, (ax_arbol, ax_fun) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.1, 1]}
    )
    dibujar_arbol(nodo, ax_arbol)
    graficar_funcion(nodo, ax_fun, texto, xmin, xmax)
    fig.tight_layout()
 
    if guardar:
        fig.savefig(guardar, dpi=150)
        print(f"Figura guardada en: {guardar}")
    if mostrar_ventana:
        plt.show()
    plt.close(fig)
 
 
# --------------------------------------------------------------------------
# 6. Programa principal
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Árbol de expresión y gráfica de una función de x."
    )
    ap.add_argument("expresion", nargs="?",
                    help='Ej.: "sqrt(2*(x^2)+5*x-3)". Si se omite, modo interactivo.')
    ap.add_argument("--xmin", type=float, default=-10.0)
    ap.add_argument("--xmax", type=float, default=10.0)
    ap.add_argument("--save", metavar="ARCHIVO",
                    help="Guarda la figura (png, pdf, svg...).")
    ap.add_argument("--no-show", action="store_true",
                    help="No abrir ventana (útil sin entorno gráfico).")
    args = ap.parse_args()
 
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
 
    if args.xmin >= args.xmax:
        ap.error("--xmin debe ser menor que --xmax")
 
    if args.expresion:
        try:
            mostrar(args.expresion, args.xmin, args.xmax, args.save,
                    not args.no_show)
        except ErrorDeExpresion as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return
 
    print("Intérprete de expresiones. Escribe una ecuación en x "
          "(o 'salir').")
    print("Funciones:", ", ".join(sorted(FUNCIONES)),
          "| constantes: pi, e | potencia: ^")
    while True:
        try:
            texto = input("\nf(x) = ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if texto.lower() in ("salir", "exit", "quit"):
            break
        if not texto:
            continue
        try:
            mostrar(texto, args.xmin, args.xmax, args.save, True)
        except ErrorDeExpresion as e:
            print(f"Error: {e}")
 
 
if __name__ == "__main__":
    main()
 