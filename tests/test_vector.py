"""Unit tests for the Vector class.

This module contains tests for all functionality of the n-dimensional Vector class.
"""

import math
import pickle
from array import array

import pytest

from vector import Vector


class TestVectorBasics:
    """Test basic Vector functionality."""

    def test_constructor(self):
        """Test Vector constructor with different inputs."""
        v1 = Vector([3, 4, 5])
        assert len(v1) == 3
        assert v1._components == array('d', [3.0, 4.0, 5.0])

        v2 = Vector([])
        assert len(v2) == 0

        v3 = Vector(range(10))
        assert len(v3) == 10
        assert v3._components == array('d', list(range(10)))

    def test_repr(self):
        """Test string representation."""
        v = Vector([3, 4, 5])
        assert repr(v) == 'Vector([3.0, 4.0, 5.0])'

        v = Vector(range(10))
        assert 'Vector(' in repr(v)
        assert '[0.0, 1.0, ...' in repr(v)
        assert '9.0]' in repr(v)

    def test_str(self):
        """Test string conversion."""
        v = Vector([3, 4, 5])
        assert str(v) == '(3.0, 4.0, 5.0)'

    def test_len(self):
        """Test length method."""
        v = Vector([3, 4, 5])
        assert len(v) == 3

        v = Vector([])
        assert len(v) == 0

    def test_iter(self):
        """Test iteration over Vector components."""
        v = Vector([3, 4, 5])
        components = list(v)
        assert components == [3.0, 4.0, 5.0]

        # Test unpacking
        x, y, z = v
        assert (x, y, z) == (3.0, 4.0, 5.0)


class TestVectorEquality:
    """Test Vector equality and hashing."""

    def test_equality(self):
        """Test equality between Vectors."""
        v1 = Vector([3, 4, 5])
        v2 = Vector([3, 4, 5])
        v3 = Vector([3, 4])
        v4 = Vector([3, 4, 6])

        assert v1 == v2
        assert v1 != v3
        assert v1 != v4
        assert v1 != (3, 4, 5)  # Different type

    def test_hash(self):
        """Test Vector hashing for dict/set usage."""
        v1 = Vector([3, 4, 5])
        v2 = Vector([3, 4, 5])
        v3 = Vector([5, 4, 3])

        assert hash(v1) == hash(v2)
        assert hash(v1) != hash(v3)

        # Test using Vectors as dict keys
        d = {v1: 'v1', v3: 'v3'}
        assert d[v2] == 'v1'  # v2 is equal to v1


class TestVectorMagnitude:
    """Test Vector magnitude and boolean conversion."""

    def test_abs(self):
        """Test Vector magnitude calculation."""
        v = Vector([3, 4])
        assert abs(v) == 5.0

        v = Vector([3, 4, 5])
        assert abs(v) == math.sqrt(3**2 + 4**2 + 5**2)

    def test_bool(self):
        """Test boolean conversion."""
        v = Vector([3, 4, 5])
        assert bool(v) is True

        v = Vector([0, 0, 0])
        assert bool(v) is False


class TestVectorAttributeAccess:
    """Test access to Vector components via attributes."""

    def test_getitem(self):
        """Test item access."""
        v = Vector([3, 4, 5])
        assert v[0] == Vector([3.0])
        assert v[1] == Vector([4.0])
        assert v[-1] == Vector([5.0])

        with pytest.raises(IndexError):
            _ = v[3]

    def test_slicing(self):
        """Test Vector slicing."""
        v = Vector([3, 4, 5, 6])
        assert v[1:3] == Vector([4.0, 5.0])
        assert v[1:] == Vector([4.0, 5.0, 6.0])
        assert v[:2] == Vector([3.0, 4.0])

    def test_getattr(self):
        """Test attribute access for x, y, z, t."""
        v = Vector([3, 4, 5, 6])
        assert v.x == 3.0
        assert v.y == 4.0
        assert v.z == 5.0
        assert v.t == 6.0

        with pytest.raises(AttributeError):
            _ = v.a  # Not a predefined attribute

        v = Vector([3])
        assert v.x == 3.0
        with pytest.raises(AttributeError):
            _ = v.y  # Out of range


class TestVectorSerialization:
    """Test Vector serialization and deserialization."""

    def test_bytes_roundtrip(self):
        """Test conversion to bytes and back."""
        v1 = Vector([3, 4, 5])
        b = bytes(v1)
        v2 = Vector.frombytes(b)
        assert v1 == v2

    def test_pickle_roundtrip(self):
        """Test pickling and unpickling."""
        v1 = Vector([3, 4, 5])
        data = pickle.dumps(v1)
        v2 = pickle.loads(data)
        assert v1 == v2


class TestVectorAngles:
    """Test Vector angle calculations."""

    def test_angle_2d(self):
        """Test angle calculation in 2D."""
        v = Vector([1.0, 0.0])
        assert v.angle(1) == 0.0

        v = Vector([0.0, 1.0])
        assert v.angle(1) == math.pi / 2

        v = Vector([-1.0, 0.0])
        assert v.angle(1) == math.pi

        v = Vector([0.0, -1.0])
        assert v.angle(1) == 3 * math.pi / 2

    def test_angles(self):
        """Test generating all angles."""
        v = Vector([3.0, 4.0, 5.0])
        angles = list(v.angles())
        assert len(angles) == 2
        assert angles[0] == math.atan2(math.hypot(4.0, 5.0), 3.0)


class TestVectorFormatting:
    """Test Vector formatting."""

    def test_default_format(self):
        """Test default formatting."""
        v = Vector([3, 4])
        assert format(v) == '(3.0, 4.0)'

    def test_custom_format(self):
        """Test formatting with format specifiers."""
        v = Vector([3, 4])
        assert format(v, '.2f') == '(3.00, 4.00)'
        assert format(v, '.0f') == '(3, 4)'

    def test_hyperspherical_format(self):
        """Test hyperspherical format."""
        v = Vector([1, 1])
        # Magnitude is sqrt(2), angle is pi/4
        assert format(v, '.2fh') == '<1.41, 0.79>'


class TestVectorOperators:
    """Test Vector operator overloading."""

    def test_negation(self):
        """Test unary negation."""
        v = Vector([3, 4, 5])
        negative = -v
        assert negative == Vector([-3, -4, -5])

    def test_positive(self):
        """Test unary positive."""
        v = Vector([3, 4, 5])
        assert +v is v  # Should return self

    def test_addition(self):
        """Test Vector addition."""
        v1 = Vector([3, 4, 5])
        v2 = Vector([6, 7, 8])
        assert v1 + v2 == Vector([9, 11, 13])

        # Test with different sizes
        v3 = Vector([6, 7])
        assert v1 + v3 == Vector([9, 11, 5])
        assert v3 + v1 == Vector([9, 11, 5])

        # Test with other iterables
        assert v1 + [6, 7, 8] == Vector([9, 11, 13])
        assert [6, 7, 8] + v1 == Vector([9, 11, 13])  # Tests __radd__

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        v = Vector([3, 4, 5])
        assert v * 2 == Vector([6, 8, 10])
        assert 2 * v == Vector([6, 8, 10])  # Tests __rmul__

        # Test with float
        assert v * 0.5 == Vector([1.5, 2.0, 2.5])

        # Test with invalid types
        with pytest.raises(TypeError):
            _ = v * '2'

    def test_matmul(self):
        """Test matrix multiplication (dot product)."""
        v1 = Vector([3, 4, 5])
        v2 = Vector([6, 7, 8])
        assert v1 @ v2 == 3 * 6 + 4 * 7 + 5 * 8
        assert v2 @ v1 == 3 * 6 + 4 * 7 + 5 * 8  # Tests __rmatmul__

        # Test with other iterables
        assert v1 @ [6, 7, 8] == 3 * 6 + 4 * 7 + 5 * 8
        assert [6, 7, 8] @ v1 == 3 * 6 + 4 * 7 + 5 * 8

        # Test with size mismatch
        with pytest.raises(ValueError):
            _ = v1 @ Vector([6, 7])

        # Test with invalid types
        with pytest.raises(TypeError):
            _ = v1 @ 2


class TestVectorAdvanced:
    """Advanced Vector tests."""

    def test_vector_in_multidimensional_space(self):
        """Test a Vector in a high-dimensional space."""
        dimensions = 10
        components = list(range(dimensions))
        v = Vector(components)

        assert len(v) == dimensions
        assert abs(v) == math.sqrt(sum(i * i for i in range(dimensions)))
