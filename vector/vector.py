"""This module provides a class to represent n-dimensional vectors.

The bulk of this code is from chapter 13 and chapter 16
of the 2nd edition of Fluent Python by Luciano Ramalho.

Classes:
    Vector: A class to represent a n-dimensional vector.
"""

import functools
import itertools
import math
import operator
import reprlib
from array import array
from collections.abc import Iterable, Iterator, Sequence, Sized
from typing import Any


class Vector:
    """A class to represent a mathematical vector.

    The Vector class provides methods to perform various vector operations
    such as iteration, representation, equality check, hashing, magnitude
    calculation, truth value testing, length retrieval, item retrieval,
    attribute retrieval, angle calculation, and formatting. It also supports
    serialization and deserialization to and from bytes.

    Attributes:
        typecode (str): The typecode used for the array of components.
        __match_args__ (tuple): A tuple of attribute names for pattern matching

    Methods:
        __init__(self, components: Iterable[float]) -> None:
            Initialize a vector with the given components.
        __iter__(self) -> Iterator:
            Return an iterator over the vector components.
        __repr__(self) -> str:
            Return a string representation of the vector.
        __str__(self) -> str:
            Return a string representation of the vector.
        __bytes__(self) -> bytes:
            Return a bytes object representing the vector.
        __eq__(self, other: Any) -> bool:
            Check if two vectors are equal.
        __hash__(self) -> int:
            Compute the hash value for the object.
        __abs__(self) -> float:
            Return the magnitude of the vector.
        __bool__(self) -> bool:
            Return the truth value of the vector.
        __len__(self) -> int:
            Return the number of components in the vector.
        __getitem__(self, key: int | slice) -> Any:
            Retrieve an item or a slice from the vector.
        __getattr__(self, name: str) -> float:
            Retrieve the attribute specified by 'name'.
        angle(self, n: int) -> float:
            Calculate the angle at the nth dimension of the vector.
        angles(self) -> Iterator:
            Generate an iterable of angles for each element in the vector.
        __format__(self, format_spec: str='') -> str:
            Format the vector according to the given format specification.
        frombytes(cls, octets: bytes) -> 'Vector':
            Create a Vector instance from a bytes object.
        __neg__(self) -> 'Vector':
            Return the negation of the vector.
        __pos__(self) -> 'Vector':
            Return the vector itself.
        __add__(self, other: 'Vector' | Iterable[float]) -> 'Vector':
            Return the sum of two vectors.
        __radd__(self, other: 'Vector' | Iterable[float]) -> 'Vector':
            Return the sum of two vectors.
        __mul__(self, scalar: float) -> 'Vector':
            Return the product of the vector and a scalar.
        __rmul__(self, scalar: float) -> 'Vector':
            Implements the right multiplication operation for the vector.
        __matmul__(self, other: 'Vector' | Iterable[float]) -> float:
            Implements dot product for two vectors using the @ infix operator.
        __rmatmul__(self, other: 'Vector' | Iterable[float]) -> float:
            Implements dot product for two vectors.
    """

    typecode = 'd'
    __match_args__ = ('x', 'y', 'z', 't')

    # Note that int is compatible with float. So both iterables of ints and
    # floats can be passed.
    def __init__(self, components: Iterable[float]) -> None:
        """Initialize a vector with the given components.

        Args:
            components (iterable): An iterable of numerical values
                    representing the vector components.
        """
        self._components = array(self.typecode, components)

    def __iter__(self) -> Iterator:
        """Return an iterator over the x and y coordinates.

        __iter__ makes a Vector iterable; this is what makes unpacking work
        (e.g., x, y = my_vector).

        Yields:
            The coordinates of the vector.
        """
        return iter(self._components)

    def __repr__(self) -> str:
        """Return a string representation of the vector.

        __repr__ builds a string by interpolating the components with {!r} to
        get their repr; because Vector is iterable. The reprlib.repr() function
        is used to limit the number of items displayed.

        __repr__ supports repr(). Returns a string representing the object as
        a developer wants to see it. It's what you get when the Python console
        or debugger shows an object.

        Returns:
            str: A string representation of the vector.
        """
        components = reprlib.repr(self._components)
        components = components[components.find('[') : -1]
        return f'Vector({components})'

    def __str__(self) -> str:
        """Return a string representation of the vector.

        __str__ supports str(). Returns a string representing the object as
        a user wants to see it. It's what you get when you print an object.

        Returns:
            str: A string representation of the vector.
        """
        return str(tuple(self))

    def __bytes__(self) -> bytes:
        """Return a bytes object representing the vector.

        __bytes__ is analogous to __str__. It is called by bytes() to get the
        object represented as a byte string. In this method the typecode is
        creates a bytes representation of the vector's typecode by converting
        its ASCII value using ord() and then wrapping it in a bytes object.
        For example, bytes([ord('d')]) returns b'd' when printed to the
        console. The method then it appends the binary representation of the
        underlying array data by calling bytes() on the array. The end result
        is a bytes object of the array data prefixed by the typecode.

        Returns:
            bytes: A binary representation of the vector.
        """
        return bytes([ord(self.typecode)]) + bytes(self._components)

    def __eq__(self, other: Any) -> bool:
        """Check if two vectors are equal.

        An example of Goose typing. Relies on runtime type checking of objects
        against ABCs.

        Args:
            other (Any): The vector to compare with.

        Returns:
            bool: True if the vectors have the same length and all
                    corresponding elements are equal, False otherwise.
        """
        if isinstance(other, Vector):
            return all(a == b for a, b in zip(self, other, strict=True))
        else:
            return NotImplemented

    def __hash__(self) -> int:
        """Compute the hash value for the object.

        This method generates a hash value by applying the XOR operation
        to the hash values of the elements in the object.

        This method generates a position-sensitive hash value by combining
        each element's hash with its position in the vector. This ensures
        that vectors with the same elements in different positions have
        different hash values.

        Returns:
            int: The computed hash value.
        """
        hashes = (hash(x) for x in self)
        return functools.reduce(operator.xor, hashes, 0)

    def __abs__(self) -> float:
        """Return the magnitude of the vector.

        __abs__ supports the abs() function. It calculates the magnitude of
        the vector using the Pythagorean theorem.

        Returns:
            float: The magnitude of the vector.
        """
        return math.hypot(*self)

    def __bool__(self) -> bool:
        """Return the truth value of the vector.

        __bool__ supports truth testing. It returns False if the magnitude of
        the vector is zero, True otherwise.

        Returns:
            bool: True if the vector is not the zero vector, False otherwise.
        """
        return bool(abs(self))

    def __len__(self) -> int:
        """Return the number of components in the vector.

        Adding the __len__ and the __getitem__ makes Vector a sequence type,
        which means it supports all the sequence protocol methods. This is
        important because it makes Vector 'sliceable'.

        Returns:
            int: The number of components.
        """
        return len(self._components)

    def __getitem__(self, key: int | slice):
        """Retrieve an item or a slice from the vector.

        Adding the __len__ and the __getitem__ makes Vector a sequence type,
        which means it supports all the sequence protocol methods. This is
        important because it makes Vector 'sliceable'.

        Args:
            key (int or slice): The index or slice to retrieve.

        Returns:
            Vector: a new instance of the vector containing the elements in t
            he specified slice or key.

        Raises:
            TypeError: If the key is not an int or slice.
        """
        cls = type(self)
        if isinstance(key, slice):
            return cls(self._components[key])
        index = operator.index(key)
        return cls([self._components[index]])

    def __getattr__(self, name: str) -> float:
        """Retrieve the attribute specified by 'name'.

        This method enables us to retrieve values using the my_vector.x,
        my_vector.y, my_vector.z, and my_vector.t syntax. A simple explanation
        for why this works is that this method is called when an attribute
        lookup fails. So if a user tries to access an attribute using, for
        example, my_vector.x, and the instance, class or any of its super
        classes do not have an attribute named 'x', then this method is called
        with 'x' as the argument. This method then attempts to find the
        attribute in the '_components' list based on the position specified in
        the '__match_args__' class attribute.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            float: The value of the attribute if found in '_components'.

        Raises:
            AttributeError: If the attribute is not found in '__match_args__'
            or '_components'.
        """
        cls = type(self)
        try:
            pos = cls.__match_args__.index(name)
        except ValueError:
            pos = -1
        if 0 <= pos < len(self._components):
            return self._components[pos]
        msg = f'{cls.__name__!r} object has no attribute {name!r}'
        raise AttributeError(msg)

    def angle(self, n: int) -> float:
        """Calculate the angle at the nth dimension of the vector.

        Args:
            n (int): The dimension at which to calculate the angle.

        Returns:
            float: The angle in radians.

        Notes:
            - Uses the Euclidean norm (hypotenuse) for the calculation.
            - Uses the arctangent function to determine the angle.
            - If n is the last dimension and the last element of the vector is
                negative, the angle is adjusted to be in the correct quadrant.
        """
        r = math.hypot(*self[n:]._components)
        a = math.atan2(r, *self[n - 1]._components)
        if (n == len(self) - 1) and (self[-1]._components[0] < 0):
            return math.pi * 2 - a
        else:
            return a

    def angles(self) -> Iterator:
        """Generate an iterable of angles for each element in the vector.

        Returns:
            generator: A generator that yields the angle for each element
            in the vector.
        """
        return (self.angle(n) for n in range(1, len(self)))

    def __format__(self, format_spec: str = '') -> str:
        """Format the vector according to the given format specification.

        If the format specification ends with 'h' (hyperspherical), the vector
        is formatted in a custom format where the magnitude and angles are
        displayed within angle brackets. Otherwise, the vector is formatted in
        the default format with components displayed within parentheses.

        Args:
            format_spec (str): The format specification string.

        Returns:
            str: The formatted string representation of the vector.
        """
        if format_spec.endswith('h'):
            format_spec = format_spec[:-1]
            coords = itertools.chain([abs(self)], self.angles())
            outer_fmt = '<{}>'
        else:
            coords = self
            outer_fmt = '({})'
        components = (format(c, format_spec) for c in coords)
        return outer_fmt.format(', '.join(components))

    # classmethod decorator modifies the method so that it can be called on
    # the class.
    @classmethod
    def frombytes(cls, octets: bytes) -> 'Vector':
        """Create a Vector instance from a bytes object.

        This class method is used to deserialize a bytes object back into a
        Vector instance. It extracts the typecode from the first byte and
        then reads the remaining bytes as an array of the specified typecode.

        The `memoryview` class is a shared-memory sequence type that lets you
        handle slices of arrays without copying bytes. It allows you to share
        memory between data-structures without first copying. This is very
        important for large datasets.

        The `memoryview.cast` method lets you change the way the bytes are
        read or written as units without moving bits around.

        Args:
            octets (bytes): A bytes object containing the serialized vector.

        Returns:
            Vector: A new Vector instance created from the bytes object.
        """
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)  # type: ignore
        return cls(memv)

    # Operator overloading with unary operators. Special methods implementing
    # unary operators should never change the value of the operands. Always
    # return a new object.
    def __neg__(self) -> 'Vector':
        """Return the negation of the vector.

        This method returns a new vector where each component is the negation
        of the corresponding component in the original vector.

        Returns:
            Vector: A new vector with each component negated.
        """
        return Vector(-x for x in self)

    def __pos__(self) -> 'Vector':
        """Return the vector itself.

        This method returns the vector itself.

        Returns:
            Vector: The vector itself.
        """
        return self

    # Operator overloading with infix operators. Special methods implementing
    # unary operators should never change the value of the operands. Always
    # return a new object.
    def __add__(self, other: 'Vector' | Sequence[float]) -> 'Vector':
        """Return the sum of two vectors.

        Args:
            other (Vector | Sequence): The vector to add.

        Returns:
            Vector: A new vector with the sum of the two vectors.
        """
        try:
            pairs = itertools.zip_longest(self, other, fillvalue=0.0)
            return Vector(a + b for a, b in pairs)
        except TypeError:
            return NotImplemented

    def __radd__(self, other: 'Vector' | Sequence[float]) -> 'Vector':
        """Return the sum of two vectors.

        Logic for when __radd__ is called:

        1. If a has __add__ method, call a.__add__(b) and return the result
        unless the result is NotImplemented.

        2. If a does not have __add__ method or calling it returns
        NotImplemented, check if b has __radd__ method, and call b.__radd__(a)
        and return the result unless the result is NotImplemented.

        3. If b doesn't have __radd__ method or calling it returns
        NotImplemented, raise TypeError with an unsupported operand
        type message.

        Args:
            other (Vector | Sequence): The vector to add.

        Returns:
            Vector: A new vector with the sum of the two vectors.
        """
        return self + other

    def __mul__(self, scalar: float) -> 'Vector':
        """Return the product of the vector and a scalar.

        Args:
            scalar (float): The scalar to multiply by.

        Returns:
            Vector: A new vector with each component multiplied by
                    the scalar.
        """
        try:
            factor = float(scalar)
        except ValueError:
            return NotImplemented
        return Vector(x * factor for x in self)

    def __rmul__(self, scalar: float) -> 'Vector':
        """Implements the right multiplication operation for the vector.

        Args:
            scalar (float or int): The scalar value to multiply with the
                    vector.

        Returns:
            Vector: A new vector that is the result of the scalar
                multiplication.
        """
        return self * scalar

    def __matmul__(self, other: 'Vector' | Sequence[float]) -> float:
        """Implements dot product for two vectors using the @ infix operator.

        Uses "Goose typing". Relies on runtime type checking of objects
        against ABCs.

        Args:
            other (Vector | Sequence): The vector to multiply with.

        Returns:
            float: The result the dot product of the two vectors.
        """
        if isinstance(other, Sized) and isinstance(other, Iterable):
            return sum(a * b for a, b in zip(self, other, strict=True))
        else:
            return NotImplemented

    def __rmatmul__(self, other: 'Vector' | Sequence[float]) -> float:
        """Implements dot product for two vectors.

        Uses "Goose typing". Relies on runtime type checking of objects
        against ABCs.

        Args:
            other (Vector | Sequence): The vector to multiply with.

        Returns:
            float: The result the dot product of the two vectors.
        """
        return self @ other
