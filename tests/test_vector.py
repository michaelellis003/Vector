"""This module contains unit tests for the Vector class.

The tests cover various aspects of the Vector class, including:
- Initialization and basic properties
- String representations
- Iteration and unpacking
- Equality comparison
- Magnitude calculation
- Boolean evaluation
- Indexing and slicing
- Attribute access (x, y, z, t)
- Angle calculations
- Formatting options
- Serialization and deserialization
- Hashing
- Error conditions

The tests use the pytest framework for assertions and test organization.
"""

import math

import pytest

from vector import Vector


def test_vector_creation():
    """Test vector initialization and basic properties."""
    v = Vector([3.0, 4.0])
    assert len(v) == 2
    assert v.x == 3.0
    assert v.y == 4.0

    # Test creation with different types of iterables
    v1 = Vector((1, 2, 3))
    assert len(v1) == 3
    v2 = Vector(range(5))
    assert len(v2) == 5


def test_vector_representation():
    """Test string representations of vectors."""
    v = Vector([3.0, 4.0, 5.0])
    assert str(v) == '(3.0, 4.0, 5.0)'
    assert repr(v) == 'Vector([3.0, 4.0, 5.0])'

    # Test repr truncation for long vectors
    v_long = Vector(range(10))
    assert '...' in repr(v_long)


def test_vector_iteration():
    """Test vector iteration and unpacking."""
    v = Vector([3.0, 4.0])
    x, y = v
    assert x == 3.0
    assert y == 4.0

    components = list(v)
    assert components == [3.0, 4.0]


def test_vector_equality():
    """Test vector equality comparison."""
    v1 = Vector([3.0, 4.0])
    v2 = Vector([3.0, 4.0])
    v3 = Vector([3.0, 4.0, 0.0])

    assert v1 == v2
    assert v1 != v3
    assert v1 != v3


def test_vector_magnitude():
    """Test vector magnitude calculation."""
    v = Vector([3.0, 4.0])
    assert abs(v) == 5.0

    v_zero = Vector([0.0, 0.0])
    assert abs(v_zero) == 0.0


def test_vector_bool():
    """Test vector boolean evaluation."""
    v = Vector([3.0, 4.0])
    assert bool(v) is True

    v_zero = Vector([0.0, 0.0])
    assert bool(v_zero) is False


def test_vector_indexing():
    """Test vector indexing and slicing."""
    v = Vector([1.0, 2.0, 3.0, 4.0, 5.0])

    # Test individual element access
    assert v[0] == 1.0
    assert v[-1] == 5.0

    # Test slicing
    v_slice = v[1:4]
    assert isinstance(v_slice, Vector)
    assert len(v_slice) == 3
    assert list(v_slice) == [2.0, 3.0, 4.0]


def test_vector_attribute_access():
    """Test vector attribute access (x, y, z, t)."""
    v = Vector([1.0, 2.0, 3.0, 4.0])
    assert v.x == 1.0
    assert v.y == 2.0
    assert v.z == 3.0
    assert v.t == 4.0

    with pytest.raises(AttributeError):
        _ = v.invalid_attribute


def test_vector_angles():
    """Test vector angle calculations."""
    v = Vector([1.0, 1.0])
    assert math.isclose(v.angle(1), math.pi / 4, rel_tol=1e-9)

    v2 = Vector([1.0, 1.0, 1.0])
    angles = list(v2.angles())
    assert len(angles) == 2


def test_vector_formatting():
    """Test vector formatting options."""
    v = Vector([1.0, 2.0])

    # Test default formatting
    assert format(v) == '(1.0, 2.0)'

    # Test hyperspherical coordinates
    v_h = format(v, '.2fh')
    assert '<' in v_h and '>' in v_h


def test_vector_bytes():
    """Test vector serialization and deserialization."""
    v1 = Vector([1.0, 2.0, 3.0])
    b = bytes(v1)
    v2 = Vector.frombytes(b)

    assert v1 == v2


def test_vector_hash():
    """Test vector hashing."""
    v1 = Vector([1.0, 2.0])
    v2 = Vector([1.0, 2.0])
    v3 = Vector([2.0, 1.0])

    # Same vectors should have same hash
    assert hash(v1) == hash(v2)
    # Different vectors should have different hashes
    assert hash(v1) != hash(v3)

    # Test vector as dictionary key
    d = {v1: 'test'}
    assert d[v2] == 'test'  # v2 equals v1, so this should work


def test_vector_errors():
    """Test error conditions."""
    with pytest.raises(TypeError):
        Vector(['not', 'a', 'number'])  # type: ignore

    v = Vector([1.0, 2.0])
    with pytest.raises(IndexError):
        _ = v[2]  # Index out of range

    with pytest.raises(AttributeError):
        _ = v.w  # Invalid attribute


# Optional: Add parametrized tests for testing multiple input cases
@pytest.mark.parametrize(
    'components,expected_magnitude',
    [
        ([3.0, 4.0], 5.0),
        ([1.0, 1.0], math.sqrt(2)),
        ([1.0, 0.0], 1.0),
        ([0.0, 0.0], 0.0),
    ],
)
def test_vector_magnitudes(components, expected_magnitude):
    """Test vector magnitude calculations with various inputs."""
    v = Vector(components)
    assert math.isclose(abs(v), expected_magnitude, rel_tol=1e-9)
