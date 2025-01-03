"""This module contains `SafeArray` code."""


from ._info import VERSION, AUTHOR

__all__ = ["SafeArray"]
__version__ = VERSION
__author__ = AUTHOR


from typing import Any


class SafeArray:
    """Typed array that will never exceed its capacity."""
    def __init__(self, dtype: Any, capacity: int = 2):
        self.capacity = capacity
        self.dtype = dtype
        self.items = []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]

    def __repr__(self):
        return (f"DynamicArray(capacity={self.capacity!r}, "
                f"dtype={self.dtype!r}, items={self.items!r})")

    def append(self, item: Any):
        """Remove first item if its capacity is exceeded. Then, append `item`
        to the end of the array.

        Parameters
        ----------
        item : Any
            Item to be appended.

        Raises
        ------
        ValueError
            - If the datatype of item to be appended is not an instance of
            `self.dtype`.
        """
        if not isinstance(item, self.dtype):
            raise ValueError(f"Datatype of new item must `{self.dtype}`, "
                             f"found `{type(item)}`.")

        if len(self.items) >= self.capacity:
            self.items.pop(0)

        self.items.append(item)
