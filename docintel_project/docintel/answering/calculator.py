"""Safe arithmetic evaluator (ast based, never uses eval/exec)."""
import ast
import math
import operator

MAX_EXPR_LEN = 500
MAX_EXPONENT = 100


class CalcError(ValueError):
    pass


def pct_change(old, new):
    """Percent change from old to new, e.g. pct_change(78, 92) = 17.948..."""
    if old == 0:
        raise CalcError("pct_change: old value is zero")
    return (new - old) / abs(old) * 100.0


def _round(x, n=0):
    return round(x, int(n))


_FUNCS = {
    "pct_change": pct_change,
    "round": _round,
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise CalcError("only numbers are allowed")
    return v


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        return _num(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        l, r = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(r) > MAX_EXPONENT:
            raise CalcError("exponent too large")
        try:
            return _BIN[type(node.op)](l, r)
        except ZeroDivisionError:
            raise CalcError("division by zero")
        except OverflowError:
            raise CalcError("overflow")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise CalcError("function not allowed")
        if node.keywords:
            raise CalcError("keyword arguments not allowed")
        args = [_eval(a) for a in node.args]
        try:
            return _num(_FUNCS[node.func.id](*args))
        except CalcError:
            raise
        except (TypeError, ValueError, ZeroDivisionError) as e:
            raise CalcError(f"bad call: {e}")
    raise CalcError(f"unsupported syntax: {type(node).__name__}")


def evaluate(expr: str) -> float:
    """Evaluate an arithmetic expression safely. Raises CalcError on anything unsafe."""
    if not isinstance(expr, str) or not expr.strip():
        raise CalcError("empty expression")
    if len(expr) > MAX_EXPR_LEN:
        raise CalcError("expression too long")
    expr = expr.replace(",", "") if _looks_thousands(expr) else expr
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise CalcError(f"syntax error: {e.msg}")
    return _eval(tree)


def _looks_thousands(expr: str) -> bool:
    """Only strip commas for numbers like 1,234,567 when no function-call commas exist."""
    import re
    return "(" not in expr and bool(re.search(r"\d,\d{3}", expr))


def _fmt(x) -> str:
    if isinstance(x, int):
        return str(x)
    if float(x).is_integer() and abs(x) < 1e15:
        return str(int(x))
    return _trim(x)


def _trim(x) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")


def format_calculation(expr: str, result: float) -> str:
    """e.g. 'pct_change(78,92) = 17.95%'; percent sign added for pct_change top-level."""
    e = expr.strip()
    is_pct = e.startswith("pct_change(") and e.endswith(")")
    if is_pct:
        return f"{e} = {round(result, 2):g}%" if not float(result).is_integer() else f"{e} = {int(result)}%"
    return f"{e} = {_fmt(result)}"
