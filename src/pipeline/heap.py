import logging
import pandas as pd

log = logging.getLogger(__name__)


class MinHeap:
    """
    Binary Min-Heap implemented from scratch (no heapq).
    Each item is a tuple: (priority: float, label: str)
    """

    def __init__(self) -> None:
        self._data: list[tuple[float, str]] = []

    def insert(self, item: tuple[float, str]) -> None:
        """Insert (priority, label) and restore heap property."""
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"Expected (float, str) tuple, got {type(item)}")
        self._data.append(item)
        self._sift_up(len(self._data) - 1)
        self._assert_heap_property()

    def extract_min(self) -> tuple[float, str]:
        """Remove and return the item with the smallest priority."""
        if self.size() == 0:
            raise IndexError("extract_min called on empty heap")
        self._data[0], self._data[-1] = self._data[-1], self._data[0]
        minimum = self._data.pop()
        if self._data:
            self._sift_down(0)
        return minimum

    def peek(self) -> tuple[float, str]:
        """Return (but don't remove) the minimum-priority item."""
        if self.size() == 0:
            raise IndexError("peek called on empty heap")
        return self._data[0]

    def size(self) -> int:
        """Return number of elements in the heap."""
        return len(self._data)

    @staticmethod
    def _parent(i: int) -> int:
        return (i - 1) // 2

    @staticmethod
    def _left(i: int) -> int:
        return 2 * i + 1

    @staticmethod
    def _right(i: int) -> int:
        return 2 * i + 2

    def _sift_up(self, i: int) -> None:
        """Bubble element at index i upward until heap property holds."""
        while i > 0:
            parent = self._parent(i)
            if self._data[i][0] < self._data[parent][0]:
                self._data[i], self._data[parent] = self._data[parent], self._data[i]
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        """Push element at index i downward until heap property holds."""
        n = self.size()
        while True:
            smallest = i
            left = self._left(i)
            right = self._right(i)

            if left < n and self._data[left][0] < self._data[smallest][0]:
                smallest = left
            if right < n and self._data[right][0] < self._data[smallest][0]:
                smallest = right

            if smallest != i:
                self._data[i], self._data[smallest] = self._data[smallest], self._data[i]
                i = smallest
            else:
                break

    def _assert_heap_property(self) -> None:
        """Assert parent ≤ both children for every node."""
        n = self.size()
        for i in range(n):
            left = self._left(i)
            right = self._right(i)
            if left < n:
                assert self._data[i][0] <= self._data[left][0], (
                    f"Heap violation: node[{i}]={self._data[i]} > left[{left}]={self._data[left]}"
                )
            if right < n:
                assert self._data[i][0] <= self._data[right][0], (
                    f"Heap violation: node[{i}]={self._data[i]} > right[{right}]={self._data[right]}"
                )

    def __repr__(self) -> str:
        return f"MinHeap(size={self.size()}, min={self._data[0] if self._data else None})"


def build_inventory_priority_queue(df: pd.DataFrame) -> list[tuple[float, str]]:
    """
    Task 2 — compute restock priority list using MinHeap
    """
    log.info("--- Starting Inventory Priority Queue ---")

    product_stats = (
        df.groupby("product")
        .agg(total_qty=("qty", "sum"), avg_price=("unit_price", "mean"))
        .reset_index()
    )
    product_stats["restock_score"] = (
        product_stats["total_qty"] / product_stats["avg_price"]
    ).round(6)

    log.info("Restock scores:\n%s", product_stats[[
             "product", "restock_score"]].to_string(index=False))

    heap = MinHeap()
    for _, row in product_stats.iterrows():
        heap.insert((row["restock_score"], row["product"]))

    log.info("Heap size after all insertions: %d", heap.size())

    priority_list: list[tuple[float, str]] = []
    while heap.size() > 0:
        priority_list.append(heap.extract_min())

    print("\n" + "=" * 50)
    print("  INVENTORY RESTOCKING PRIORITY (lowest score first)")
    print("=" * 50)
    for rank, (score, product) in enumerate(priority_list, start=1):
        print(f"  {rank}. {product:<10}  restock_score = {score:.6f}")
    print("=" * 50 + "\n")

    return priority_list
