"""An exact reference for the narrow float formats, written in rational arithmetic.

This is the other half of the oracle. The Java implementation in ``io.drift.fmt``
rounds by rescaling a double by a power of two, which is fast and exact but shares
its reasoning with everything downstream of it. This module reaches the same answers
by a different route: values are ``fractions.Fraction``, so a product of two
representable numbers is the true product with no rounding at all, and the only
approximation in the whole file is the single deliberate rounding under test.

Nothing here imports from the Java side or reads its output format definitions. The
field widths come from IEEE 754 and the OCP Microscaling specification. If the two
implementations agree on every code and every pair of codes, the agreement means
something.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Union

# Sentinels for the values a Fraction cannot carry. Negative zero is one of them:
# every format here spends a code on it, and it is produced by ordinary arithmetic
# (a positive zero times any negative number), so leaving it out would make the
# reference disagree with the hardware on 154 of the 65,536 e4m3 products.
NAN = "nan"
POS_INF = "inf"
NEG_INF = "-inf"
NEG_ZERO = "-0"

Value = Union[Fraction, str]

IEEE = "IEEE"
FN = "FN"
FINITE = "FINITE"

SATURATE = "SATURATE"
PROPAGATE = "PROPAGATE"


def pow2(k: int) -> Fraction:
    """Exactly 2**k, for any sign of k."""
    return Fraction(2**k, 1) if k >= 0 else Fraction(1, 2**-k)


def floor_log2(a: Fraction) -> int:
    """The largest e with 2**e <= a, for a > 0. Pure integer reasoning."""
    if a <= 0:
        raise ValueError("floor_log2 needs a positive value")
    # bit_length gives the right answer up to one, so start there and walk.
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while pow2(e) > a:
        e -= 1
    while pow2(e + 1) <= a:
        e += 1
    return e


def round_half_even(x: Fraction) -> int:
    """Nearest integer to an exact rational, ties to even."""
    n, d = x.numerator, x.denominator
    q, r = divmod(n, d)  # Python floors, which is what we want for negatives too
    twice = 2 * r
    if twice > d:
        q += 1
    elif twice == d and (q & 1):
        q += 1
    return q


@dataclass(frozen=True)
class Fmt:
    """A minifloat, described only by its fields and its treatment of the top exponent."""

    name: str
    exp_bits: int
    mant_bits: int
    bias: int
    specials: str
    overflow: str = SATURATE

    @property
    def width(self) -> int:
        return 1 + self.exp_bits + self.mant_bits

    @property
    def codes(self) -> int:
        return 1 << self.width

    @property
    def max_exp_field(self) -> int:
        return (1 << self.exp_bits) - 1

    @property
    def mant_mask(self) -> int:
        return (1 << self.mant_bits) - 1

    @property
    def sign_bit(self) -> int:
        return 1 << (self.exp_bits + self.mant_bits)

    @property
    def min_normal_exp(self) -> int:
        return 1 - self.bias

    @property
    def max_finite(self) -> Fraction:
        if self.specials == IEEE:
            top_exp, top_mant = self.max_exp_field - 1, self.mant_mask
        elif self.specials == FN:
            top_exp, top_mant = self.max_exp_field, self.mant_mask - 1
        else:
            top_exp, top_mant = self.max_exp_field, self.mant_mask
        sig = 1 + Fraction(top_mant, 1 << self.mant_bits)
        return sig * pow2(top_exp - self.bias)

    # ------------------------------------------------------------------ decode

    def decode(self, code: int) -> Value:
        """The exact value of a bit pattern."""
        if code < 0 or code >= self.codes:
            raise ValueError(f"code {code} outside {self.name}")
        mant = code & self.mant_mask
        exp_field = (code >> self.mant_bits) & self.max_exp_field
        negative = bool(code & self.sign_bit)

        if exp_field == self.max_exp_field:
            if self.specials == IEEE:
                if mant == 0:
                    return NEG_INF if negative else POS_INF
                return NAN
            if self.specials == FN and mant == self.mant_mask:
                return NAN

        if exp_field == 0:
            if mant == 0:
                return NEG_ZERO if negative else Fraction(0)
            value = Fraction(mant, 1) * pow2(self.min_normal_exp - self.mant_bits)
        else:
            sig = 1 + Fraction(mant, 1 << self.mant_bits)
            value = sig * pow2(exp_field - self.bias)
        return -value if negative else value

    # ------------------------------------------------------------------ encode

    def encode(self, value: Value) -> int:
        """Round an exact value into this format, nearest with ties to even."""
        if value is NAN or value == NAN:
            return self._nan_code()
        if value == POS_INF:
            return self._overflow_code()
        if value == NEG_INF:
            return self.sign_bit | self._overflow_code()
        if value == NEG_ZERO:
            return self.sign_bit

        assert isinstance(value, Fraction)
        sign = self.sign_bit if value < 0 else 0
        a = -value if value < 0 else value

        if a == 0:
            return sign

        e = floor_log2(a)
        if e < self.min_normal_exp:
            e = self.min_normal_exp

        quantum = pow2(e - self.mant_bits)
        m = round_half_even(a / quantum)

        normal_base = 1 << self.mant_bits
        if m < normal_base:
            # Subnormal; the integer is already the encoding.
            return sign | m
        if m >= normal_base * 2:
            m //= 2
            e += 1

        exp_field = e + self.bias
        mant_field = m - normal_base

        if exp_field > self.max_exp_field or (
            exp_field == self.max_exp_field and not self._top_binade_allows(mant_field)
        ):
            return sign | self._overflow_code()
        return sign | (exp_field << self.mant_bits) | mant_field

    def _top_binade_allows(self, mant_field: int) -> bool:
        if self.specials == IEEE:
            return False
        if self.specials == FN:
            return mant_field < self.mant_mask
        return True

    def _nan_code(self) -> int:
        if self.specials == IEEE:
            return (self.max_exp_field << self.mant_bits) | 1
        if self.specials == FN:
            return (self.max_exp_field << self.mant_bits) | self.mant_mask
        raise ArithmeticError(f"{self.name} has no NaN")

    def _overflow_code(self) -> int:
        if self.overflow == SATURATE:
            return self._max_finite_code()
        if self.specials == IEEE:
            return self.max_exp_field << self.mant_bits
        if self.specials == FN:
            return (self.max_exp_field << self.mant_bits) | self.mant_mask
        return self._max_finite_code()

    def _max_finite_code(self) -> int:
        if self.specials == IEEE:
            return ((self.max_exp_field - 1) << self.mant_bits) | self.mant_mask
        if self.specials == FN:
            return (self.max_exp_field << self.mant_bits) | (self.mant_mask - 1)
        return (self.max_exp_field << self.mant_bits) | self.mant_mask


E4M3 = Fmt("e4m3", 4, 3, 7, FN)
E5M2 = Fmt("e5m2", 5, 2, 15, IEEE)
E2M1 = Fmt("e2m1", 2, 1, 1, FINITE)
E3M2 = Fmt("e3m2", 3, 2, 3, FINITE)
E2M3 = Fmt("e2m3", 2, 3, 1, FINITE)
FP16 = Fmt("fp16", 5, 10, 15, IEEE)
BF16 = Fmt("bf16", 8, 7, 127, IEEE)

ALL = [E4M3, E5M2, E2M1, E3M2, E2M3, FP16, BF16]
BY_NAME = {f.name: f for f in ALL}


def is_nan(v: Value) -> bool:
    return isinstance(v, str) and v == NAN


def is_inf(v: Value) -> bool:
    return isinstance(v, str) and v in (POS_INF, NEG_INF)


def is_zero(v: Value) -> bool:
    return v == NEG_ZERO or (isinstance(v, Fraction) and v == 0)


def signbit(v: Value) -> bool:
    """The sign bit, which for zero is information a magnitude does not carry."""
    if isinstance(v, str):
        return v in (NEG_INF, NEG_ZERO)
    return v < 0


def magnitude(v: Value) -> Fraction:
    """The finite value as a rational, with negative zero flattened to zero."""
    if v == NEG_ZERO:
        return Fraction(0)
    assert isinstance(v, Fraction)
    return v


def add(fmt: Fmt, a: Value, b: Value) -> Value:
    """Exact sum, following IEEE 754 on the cases a rational cannot express."""
    if is_nan(a) or is_nan(b):
        return NAN
    if is_inf(a) or is_inf(b):
        if is_inf(a) and is_inf(b):
            return NAN if a != b else a
        return a if is_inf(a) else b
    if is_zero(a) and is_zero(b):
        # Two zeros sum to negative zero only when both were negative. Under any
        # other rounding mode than nearest this would not be true.
        return NEG_ZERO if (signbit(a) and signbit(b)) else Fraction(0)
    total = magnitude(a) + magnitude(b)
    # An exact cancellation is positive zero in round-to-nearest, whatever the signs.
    return total


def mul(fmt: Fmt, a: Value, b: Value) -> Value:
    """Exact product, following IEEE 754 on the cases a rational cannot express."""
    if is_nan(a) or is_nan(b):
        return NAN
    negative = signbit(a) ^ signbit(b)
    if is_inf(a) or is_inf(b):
        if is_zero(a) or is_zero(b):
            return NAN
        return NEG_INF if negative else POS_INF
    if is_zero(a) or is_zero(b):
        return NEG_ZERO if negative else Fraction(0)
    return magnitude(a) * magnitude(b)
